#include "search/search.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "tactics/engine.h"
#include "tactics/vcf.h"

namespace dz::search {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline float sign_of(PlayerId p) { return p == PlayerId::P0 ? 1.0f : -1.0f; }

// Expected score for the actor from W/D/L logits: P(win) - P(loss).
float wdl_value(const std::array<float, 3>& wdl) {
  const float m = std::max({wdl[0], wdl[1], wdl[2]});
  const float a = std::exp(wdl[0] - m), b = std::exp(wdl[1] - m), c = std::exp(wdl[2] - m);
  return (a - c) / (a + b + c);
}

tactics::Rules rules_of(const GameState& g) { return {g.profile().black, g.profile().pass_allowed}; }

}  // namespace

float action_logit(const Evaluation& e, const Action& a) {
  switch (a.type) {
    case ActionType::PLACE: return e.policy[a.point];
    case ActionType::PASS: return e.policy[features::kPassIndex];
    case ActionType::SWAP: return e.swap[0];
    case ActionType::KEEP_COLOR: return e.swap[1];
    case ActionType::CHOOSE_DIRECT_FIFTH: return e.branch[0];
    case ActionType::CHOOSE_TEN_PROPOSALS: return e.branch[1];
    case ActionType::PROPOSE_FIFTH: return e.propose[a.point];
    case ActionType::SELECT_PROPOSED_FIFTH: return e.select[a.point];
  }
  return 0;
}

Tree::Tree(const GameState& root, const SearchConfig& cfg, const features::FeatureConfig& fcfg)
    : root_state_(root), work_(root), cfg_(cfg), fcfg_(fcfg), rng_(cfg.seed), start_(std::chrono::steady_clock::now()) {
  nodes_.reserve(static_cast<size_t>(std::min(cfg.simulations, 100000)) + cfg.max_inflight + 16);
  new_node(-1);
  if (cfg.proof_threads > 0) proof_box_ = std::make_shared<ProofMailbox>();
  if (root.is_terminal() || root.phase() != OpeningPhase::NORMAL_RENJU) return;
  const tactics::Rules r = rules_of(root);
  std::vector<Point> five;
  tactics::five_points(root.board(), root.side_to_move(), r, five);
  if (!five.empty()) {  // exact immediate win: no search needed
    root_proven_by_tactics_ = 1;
    root_tactics_move_ = Action::place(five.front());
    return;
  }
  if (cfg.root_tactics_nodes > 0) {
    const uint64_t n = cfg.root_tactics_nodes;
    const double tactics_ms = cfg.max_ms > 0 ? std::max(1.0, cfg.max_ms * cfg.root_tactics_time_share) : 0.0;
    const tactics::TacticalDecision d = tactics::analyze_tactics(root, {n, n, n / 2, 40, tactics_ms});
    if (d.result == tactics::ProofResult::WIN && d.move) {
      root_proven_by_tactics_ = 1;
      root_tactics_move_ = Action::place(*d.move);
    } else if (d.result == tactics::ProofResult::LOSS) {
      root_proven_by_tactics_ = -1;  // keep searching for the most resistant move
    }
  }
}

Tree::~Tree() {
  if (proof_box_) {
    std::lock_guard<std::mutex> lk(proof_box_->mu);
    proof_box_->closed = true;  // queued jobs are dropped, running ones discard their result
  }
}

GameState Tree::state_of(int32_t id) const {
  std::vector<Action> acts;
  for (int32_t c = id; nodes_[c].parent >= 0; c = nodes_[c].parent) {
    const Node& p = nodes_[nodes_[c].parent];
    for (uint32_t i = 0; i < p.edge_count; ++i) {
      if (edges_[p.edge_begin + i].child == c) {
        acts.push_back(edges_[p.edge_begin + i].action);
        break;
      }
    }
  }
  GameState g = root_state_;
  for (auto it = acts.rbegin(); it != acts.rend(); ++it) {
    [[maybe_unused]] const bool ok = g.apply(*it);
    assert(ok && "tree edge must be legal");
  }
  return g;
}

// Apply finished proof attempts. A proof is exact (independent of budgets): WIN = the node's actor
// has a forced win, LOSS = it cannot stop an immediate win. UNKNOWN changes nothing but the
// attempt count (never treated as a loss).
void Tree::drain_proofs() {
  if (!proof_box_ || proof_outstanding_ == 0) return;
  std::vector<ProofOutcome> got;
  {
    std::lock_guard<std::mutex> lk(proof_box_->mu);
    got.swap(proof_box_->done);
  }
  for (const ProofOutcome& o : got) {
    --proof_outstanding_;
    ++proof_jobs_;
    proof_nodes_ += o.nodes;
    Node& n = nodes_[o.node];
    n.proof_busy = false;
    if (!o.skipped && n.proof_level < 255) ++n.proof_level;
    if (o.result == tactics::ProofResult::UNKNOWN) continue;
    const bool win = o.result == tactics::ProofResult::WIN;
    if (o.node == 0) {  // root: a win needs its move; the search then stops and plays it
      if (win && o.move != kNoPoint && root_proven_by_tactics_ != 1) {
        root_proven_by_tactics_ = 1;
        root_tactics_move_ = Action::place(o.move);
        ++proof_found_;
      } else if (!win && root_proven_by_tactics_ == kUnknown) {
        root_proven_by_tactics_ = -1;  // keep searching for the most resistant move
        ++proof_found_;
      }
      continue;
    }
    if (n.proven != kUnknown) continue;
    const float s = sign_of(n.actor);
    n.proven = static_cast<int8_t>(win ? s : -s);
    ++proof_found_;
    for (int32_t p = n.parent; p >= 0; p = nodes_[p].parent) {
      update_proven(p);
      if (nodes_[p].proven == kUnknown) break;
    }
  }
}

// Queue attempts for the most-visited unsolved nodes within proof_max_depth, at most
// proof_threads outstanding per tree.
void Tree::schedule_proofs() {
  if (!proof_box_) return;
  if (proof_last_scan_ >= 0 && completed_ - proof_last_scan_ < cfg_.proof_scan_every) return;
  proof_last_scan_ = completed_;
  int slots = cfg_.proof_threads - proof_outstanding_;
  if (slots <= 0) return;
  if (cfg_.max_ms > 0 && elapsed_ms() >= cfg_.max_ms) return;
  struct Cand {
    int visits;
    int32_t id;
  };
  std::vector<Cand> cands;
  std::vector<std::pair<int32_t, int>> stack{{0, 0}};
  while (!stack.empty()) {
    const auto [id, depth] = stack.back();
    stack.pop_back();
    const Node& n = nodes_[id];
    if (n.state != NodeState::EXPANDED || n.visits < cfg_.proof_min_visits) continue;
    const uint64_t budget = cfg_.proof_vct_nodes << (2 * std::min<int>(n.proof_level, 20));
    if (n.proven == kUnknown && !n.proof_busy && budget <= cfg_.proof_max_vct_nodes &&
        !(id == 0 && root_proven_by_tactics_ != kUnknown)) {
      cands.push_back({n.visits, id});
    }
    if (n.proven != kUnknown || depth >= cfg_.proof_max_depth) continue;
    for (uint32_t i = 0; i < n.edge_count; ++i) {
      const int32_t c = edges_[n.edge_begin + i].child;
      if (c >= 0) stack.push_back({c, depth + 1});
    }
  }
  std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.visits > y.visits; });
  ProofPool& pool = ProofPool::instance(cfg_.proof_threads);
  for (const Cand& c : cands) {
    if (slots <= 0) break;
    Node& n = nodes_[c.id];
    GameState g = state_of(c.id);
    if (g.phase() != OpeningPhase::NORMAL_RENJU || g.is_terminal()) {
      n.proof_level = 255;  // never provable by the solver
      continue;
    }
    ProofJob job{proof_box_, c.id, std::move(g), cfg_.proof_vcf_nodes,
                 cfg_.proof_vct_nodes << (2 * std::min<int>(n.proof_level, 20)), {}};
    if (cfg_.max_ms > 0) {
      job.deadline = start_ + std::chrono::microseconds(static_cast<int64_t>(cfg_.max_ms * 1000.0));
    }
    n.proof_busy = true;
    ++proof_outstanding_;
    --slots;
    pool.submit(std::move(job));
  }
}

int32_t Tree::new_node(int32_t parent) {
  Node n;
  n.parent = parent;
  nodes_.push_back(n);
  return static_cast<int32_t>(nodes_.size() - 1);
}

bool Tree::done() const {
  if (root_state_.is_terminal()) return true;
  if (root_proven_by_tactics_ == 1 && root_tactics_move_) return true;
  if (nodes_[0].proven != kUnknown) return true;
  if (nodes_[0].state == NodeState::EXPANDED && nodes_[0].edge_count == 1 && completed_ >= 1) return true;
  if (cfg_.max_ms > 0 && completed_ >= 1 && elapsed_ms() >= cfg_.max_ms) return true;
  return completed_ >= cfg_.simulations;
}

double Tree::elapsed_ms() const {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
}

std::vector<Action> Tree::leaf_actions(const GameState& g, int8_t& proven_p0) const {
  proven_p0 = kUnknown;
  std::vector<Action> acts = g.legal_actions();
  if (!cfg_.leaf_tactics || g.phase() != OpeningPhase::NORMAL_RENJU) return acts;
  const float s = sign_of(g.actor());
  const Color me = g.side_to_move();
  const tactics::Rules r = rules_of(g);
  std::vector<Point> pts;
  tactics::five_points(g.board(), me, r, pts);
  if (!pts.empty()) {
    proven_p0 = static_cast<int8_t>(s);
    return {};
  }
  tactics::five_points(g.board(), opposite(me), r, pts);
  if (pts.size() >= 2 || (pts.size() == 1 && !tactics::is_legal_move(g.board(), pts[0], me, r))) {
    proven_p0 = static_cast<int8_t>(-s);
    return {};
  }
  if (pts.size() == 1) return {Action::place(pts[0])};  // the only non-losing move
  if (cfg_.leaf_vcf_nodes > 0) {
    auto pos = tactics::TPos::from_state(g);
    if (pos && tactics::solve_vcf(*pos, {cfg_.leaf_vcf_nodes, 30}).status == tactics::VcfResult::Status::FOUND) {
      proven_p0 = static_cast<int8_t>(s);
      return {};
    }
  }
  return acts;
}

float Tree::q_actor(const Node& c, PlayerId actor) const {
  const int n = c.visits + c.inflight;
  if (n == 0) return 0;
  return (c.value_sum * sign_of(actor) - cfg_.virtual_loss * static_cast<float>(c.inflight)) / static_cast<float>(n);
}

int Tree::choose_edge(int32_t id) {
  const Node& node = nodes_[id];
  const PlayerId actor = node.actor;
  const float a = sign_of(actor);
  const float parent_n = static_cast<float>(std::max(1, node.visits + node.inflight));
  const float sqrt_n = std::sqrt(parent_n);
  const float parent_q = node.visits > 0 ? node.value_sum * a / static_cast<float>(node.visits) : 0.0f;
  float visited_prior = 0;
  for (uint32_t i = 0; i < node.edge_count; ++i) {
    const Edge& e = edges_[node.edge_begin + i];
    if (e.child >= 0 && nodes_[e.child].visits > 0) visited_prior += e.prior;
  }
  const float fpu = parent_q - cfg_.fpu_reduction * std::sqrt(visited_prior);
  int best = -1;
  float best_score = kNegInf;
  for (uint32_t i = 0; i < node.edge_count; ++i) {
    const Edge& e = edges_[node.edge_begin + i];
    float q = fpu;
    int n = 0;
    if (e.child >= 0) {
      const Node& c = nodes_[e.child];
      if (c.proven != kUnknown) {
        const float pv = static_cast<float>(c.proven) * a;
        if (pv > 0.5f) return static_cast<int>(i);  // proven win for the actor
        if (pv < -0.5f) continue;                   // proven loss: never choose while alternatives exist
        q = 0.0f;
        n = c.visits + c.inflight;
      } else {
        if (c.state == NodeState::PENDING) continue;  // evaluation in flight
        n = c.visits + c.inflight;
        if (n > 0) q = q_actor(c, actor);
      }
    }
    const float score = q + cfg_.c_puct * e.prior * sqrt_n / (1.0f + static_cast<float>(n));
    if (score > best_score) {
      best_score = score;
      best = static_cast<int>(i);
    }
  }
  return best;
}

float Tree::sigma(float qhat) const {
  int max_n = 0;
  const Node& root = nodes_[0];
  for (uint32_t i = 0; i < root.edge_count; ++i) {
    const Edge& e = edges_[root.edge_begin + i];
    if (e.child >= 0) max_n = std::max(max_n, nodes_[e.child].visits);
  }
  return (cfg_.gumbel_c_visit + static_cast<float>(max_n)) * cfg_.gumbel_c_scale * qhat;
}

void Tree::init_gumbel() {
  const Node& root = nodes_[0];
  const int k = root.edge_count;
  gumbel_.assign(k, 0.0f);
  if (cfg_.gumbel_noise) {
    std::uniform_real_distribution<double> u(1e-12, 1.0);
    for (int i = 0; i < k; ++i) gumbel_[i] = static_cast<float>(-std::log(-std::log(u(rng_))));
  }
  std::vector<int> order(k);
  for (int i = 0; i < k; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
    return gumbel_[x] + edges_[root.edge_begin + x].logit > gumbel_[y] + edges_[root.edge_begin + y].logit;
  });
  const int m = std::min(cfg_.gumbel_m, k);
  gumbel_set_.assign(order.begin(), order.begin() + m);
  const int phases = std::max(1, static_cast<int>(std::ceil(std::log2(static_cast<double>(std::max(2, m))))));
  gumbel_phase_budget_ = std::max(1, cfg_.simulations / phases);
  gumbel_target_.assign(k, 0);
  const int per = std::max(1, gumbel_phase_budget_ / m);
  for (int i : gumbel_set_) gumbel_target_[i] = per;
  gumbel_ready_ = true;
}

int Tree::choose_root_gumbel() {
  const Node& root = nodes_[0];
  const PlayerId actor = root.actor;
  auto score = [&](int i) {
    const Edge& e = edges_[root.edge_begin + i];
    float q = 0;
    if (e.child >= 0 && nodes_[e.child].visits > 0) q = (q_actor(nodes_[e.child], actor) + 1.0f) * 0.5f;
    else q = (root.visits > 0 ? (root.value_sum * sign_of(actor) / root.visits + 1.0f) * 0.5f : 0.5f);
    return gumbel_[i] + e.logit + sigma(q);
  };
  for (int guard = 0; guard < 64; ++guard) {
    int best = -1, best_n = std::numeric_limits<int>::max();
    for (int i : gumbel_set_) {
      const Edge& e = edges_[root.edge_begin + i];
      int n = 0;
      if (e.child >= 0) {
        const Node& c = nodes_[e.child];
        if (c.proven != kUnknown && static_cast<float>(c.proven) * sign_of(actor) < -0.5f) continue;
        if (c.state == NodeState::PENDING) continue;
        n = c.visits + c.inflight;
      }
      if (n < gumbel_target_[i] && n < best_n) {
        best_n = n;
        best = i;
      }
    }
    if (best >= 0) return best;
    // Every candidate reached its target (or is unavailable): next halving phase.
    bool any_pending = false;
    for (int i : gumbel_set_) {
      const Edge& e = edges_[root.edge_begin + i];
      if (e.child >= 0 && nodes_[e.child].state == NodeState::PENDING) any_pending = true;
    }
    if (any_pending) return -1;  // wait for evaluations before ranking
    // Every candidate is a proven loss: refill the set with the remaining non-losing root moves
    // (best g + logit first), otherwise the search would stall with legal alternatives left.
    bool all_losing = true;
    for (int i : gumbel_set_) {
      const Edge& e = edges_[root.edge_begin + i];
      if (!(e.child >= 0 && nodes_[e.child].proven != kUnknown &&
            static_cast<float>(nodes_[e.child].proven) * sign_of(actor) < -0.5f)) {
        all_losing = false;
      }
    }
    if (all_losing) {
      std::vector<int> rest;
      for (int i = 0; i < static_cast<int>(root.edge_count); ++i) {
        if (std::find(gumbel_set_.begin(), gumbel_set_.end(), i) != gumbel_set_.end()) continue;
        const Edge& e = edges_[root.edge_begin + i];
        if (e.child >= 0 && nodes_[e.child].proven != kUnknown &&
            static_cast<float>(nodes_[e.child].proven) * sign_of(actor) < -0.5f) {
          continue;
        }
        rest.push_back(i);
      }
      if (rest.empty()) return choose_edge(0);
      std::stable_sort(rest.begin(), rest.end(), [&](int x, int y) {
        return gumbel_[x] + edges_[root.edge_begin + x].logit > gumbel_[y] + edges_[root.edge_begin + y].logit;
      });
      rest.resize(std::min<size_t>(rest.size(), static_cast<size_t>(std::max(1, cfg_.gumbel_m))));
      gumbel_set_ = rest;
      const int per = std::max(1, gumbel_phase_budget_ / static_cast<int>(gumbel_set_.size()));
      for (int i : gumbel_set_) gumbel_target_[i] = (gumbel_target_[i] > 0 ? gumbel_target_[i] : 0) + per;
      continue;
    }
    if (gumbel_set_.size() > 1) {
      std::stable_sort(gumbel_set_.begin(), gumbel_set_.end(), [&](int x, int y) { return score(x) > score(y); });
      gumbel_set_.resize((gumbel_set_.size() + 1) / 2);
    }
    const int per = std::max(1, gumbel_phase_budget_ / static_cast<int>(gumbel_set_.size()));
    for (int i : gumbel_set_) gumbel_target_[i] += per;
  }
  return -1;
}

void Tree::expand(int32_t id, const std::vector<Action>& actions, const Evaluation& e) {
  Node& node = nodes_[id];
  node.edge_begin = static_cast<uint32_t>(edges_.size());
  node.edge_count = static_cast<uint16_t>(actions.size());
  float mx = kNegInf;
  for (const Action& a : actions) mx = std::max(mx, action_logit(e, a));
  float sum = 0;
  for (const Action& a : actions) {
    Edge ed;
    ed.action = a;
    ed.logit = action_logit(e, a);
    ed.prior = std::exp(ed.logit - mx);
    sum += ed.prior;
    edges_.push_back(ed);
  }
  for (uint32_t i = 0; i < node.edge_count; ++i) edges_[node.edge_begin + i].prior /= sum;
  node.state = NodeState::EXPANDED;
  if (id == 0 && cfg_.root == RootMode::GUMBEL) init_gumbel();
}

void Tree::backup(const std::vector<int32_t>& path, float value_p0, bool had_inflight) {
  for (int32_t id : path) {
    Node& n = nodes_[id];
    n.visits += 1;
    n.value_sum += value_p0;
    if (had_inflight) n.inflight -= 1;
  }
  if (nodes_[path.back()].proven != kUnknown) {
    for (int k = static_cast<int>(path.size()) - 2; k >= 0; --k) {
      update_proven(path[k]);
      if (nodes_[path[k]].proven == kUnknown) break;
    }
  }
}

void Tree::update_proven(int32_t id) {
  Node& node = nodes_[id];
  if (node.state != NodeState::EXPANDED || node.proven != kUnknown) return;
  const float a = sign_of(node.actor);
  bool all = node.edge_count > 0;
  float best = -2;
  for (uint32_t i = 0; i < node.edge_count; ++i) {
    const Edge& e = edges_[node.edge_begin + i];
    if (e.child < 0 || nodes_[e.child].proven == kUnknown) {
      all = false;
      continue;
    }
    const float pv = static_cast<float>(nodes_[e.child].proven) * a;
    if (pv > 0.5f) {
      node.proven = nodes_[e.child].proven;
      return;
    }
    best = std::max(best, pv);
  }
  if (all) node.proven = static_cast<int8_t>(best * a);
}

int Tree::descend(const std::function<void(const GameState&, Pending&)>& on_leaf) {
  drain_proofs();
  if (done()) return -2;
  schedule_proofs();
  if (static_cast<int>(pending_.size()) >= cfg_.max_inflight) return -2;
  std::vector<int32_t> path{0};
  int applied = 0;
  auto unwind = [&]() {
    for (int i = 0; i < applied; ++i) work_.undo();
  };
  int32_t id = 0;
  while (true) {
    Node& node = nodes_[id];
    if (node.proven != kUnknown) {  // re-visit of a solved node: back up its exact value
      backup(path, static_cast<float>(node.proven), false);
      ++completed_;
      unwind();
      return -1;
    }
    if (node.state == NodeState::PENDING) {
      unwind();
      return -2;
    }
    if (node.state == NodeState::NEW) {
      node.actor = work_.actor();
      if (work_.is_terminal()) {
        const auto w = work_.winner();
        node.proven = w ? static_cast<int8_t>(sign_of(*w)) : 0;
        backup(path, static_cast<float>(node.proven), false);
        ++completed_;
        unwind();
        return -1;
      }
      int8_t proven = kUnknown;
      std::vector<Action> acts = leaf_actions(work_, proven);
      if (proven != kUnknown) {
        node.proven = proven;
        backup(path, static_cast<float>(proven), false);
        ++completed_;
        unwind();
        return -1;
      }
      node.state = NodeState::PENDING;
      int ticket;
      if (!free_slots_.empty()) {
        ticket = free_slots_.back();
        free_slots_.pop_back();
      } else {
        ticket = static_cast<int>(pending_slots_.size());
        pending_slots_.emplace_back();
      }
      Pending& slot = pending_slots_[ticket];
      slot.path = path;
      slot.actions = std::move(acts);
      on_leaf(work_, slot);
      for (int32_t p : path) nodes_[p].inflight += 1;
      pending_.push_back(ticket);
      unwind();
      return ticket;
    }
    // EXPANDED
    const int e = (id == 0 && cfg_.root == RootMode::GUMBEL && gumbel_ready_) ? choose_root_gumbel() : choose_edge(id);
    if (e < 0) {
      unwind();
      return -2;
    }
    const uint32_t ei = nodes_[id].edge_begin + static_cast<uint32_t>(e);
    if (edges_[ei].child < 0) {
      const int32_t c = new_node(id);
      edges_[ei].child = c;
    }
    [[maybe_unused]] const bool ok = work_.apply(edges_[ei].action);
    assert(ok && "tree edge must be legal");
    ++applied;
    id = edges_[ei].child;
    path.push_back(id);
  }
}

int Tree::select(uint8_t* features_out) {
  return descend([&](const GameState& g, Pending&) { features::encode(g, fcfg_, features_out); });
}

void Tree::complete(int ticket, const Evaluation& e) {
  Pending& slot = pending_slots_[ticket];
  const int32_t leaf = slot.path.back();
  expand(leaf, slot.actions, e);
  const float v = wdl_value(e.wdl) * sign_of(nodes_[leaf].actor);
  backup(slot.path, v, true);
  ++completed_;
  pending_.erase(std::find(pending_.begin(), pending_.end(), ticket));
  free_slots_.push_back(ticket);
}

SearchResult Tree::run(Evaluator& eval) {
  while (!done()) {
    Evaluation ev;
    const int t = descend([&](const GameState& g, Pending&) { ev = eval.evaluate(g); });
    if (t >= 0) complete(t, ev);
    else if (t == -2) {
      stalled_ = true;  // nothing selectable although not done: a search bug, surfaced in result()
      break;
    }
  }
  return result();
}

int Tree::final_root_edge() const {
  const Node& root = nodes_[0];
  if (root.state != NodeState::EXPANDED || root.edge_count == 0) return -1;
  const PlayerId actor = root.actor;
  const float a = sign_of(actor);
  // A proven win is always played.
  for (uint32_t i = 0; i < root.edge_count; ++i) {
    const Edge& e = edges_[root.edge_begin + i];
    if (e.child >= 0 && nodes_[e.child].proven != kUnknown && nodes_[e.child].proven * a > 0.5f) return static_cast<int>(i);
  }
  auto losing = [&](int i) {
    const Edge& e = edges_[root.edge_begin + i];
    return e.child >= 0 && nodes_[e.child].proven != kUnknown && nodes_[e.child].proven * a < -0.5f;
  };
  if (cfg_.root == RootMode::GUMBEL && gumbel_ready_) {
    int best = -1;
    float best_s = kNegInf;
    for (int i : gumbel_set_) {
      if (losing(i) && gumbel_set_.size() > 1) continue;
      const Edge& e = edges_[root.edge_begin + i];
      const float q = (e.child >= 0 && nodes_[e.child].visits > 0)
                          ? (static_cast<float>(nodes_[e.child].value_sum) * a / nodes_[e.child].visits + 1.0f) * 0.5f
                          : 0.5f;
      const float s = gumbel_[i] + e.logit + sigma(q);
      if (s > best_s) { best_s = s; best = i; }
    }
    if (best >= 0) return best;
  }
  int best = -1;
  int best_n = -1;
  float best_q = kNegInf;
  for (uint32_t i = 0; i < root.edge_count; ++i) {
    const Edge& e = edges_[root.edge_begin + i];
    const int n = e.child >= 0 ? nodes_[e.child].visits : 0;
    const float q = n > 0 ? nodes_[e.child].value_sum * a / n : kNegInf;
    const bool lose = losing(static_cast<int>(i));
    // Prefer non-losing moves, then most visits, then higher Q.
    const int key_n = lose ? -1 : n;
    if (key_n > best_n || (key_n == best_n && q > best_q)) {
      best = static_cast<int>(i);
      best_n = key_n;
      best_q = q;
    }
  }
  return best;
}

std::string Tree::debug_string() const {
  std::string s = root_state_.to_string();
  const Node& root = nodes_[0];
  s += "done=" + std::to_string(done()) + " completed=" + std::to_string(completed_) + " pending=" +
       std::to_string(pending_.size()) + " root_state=" + std::to_string(static_cast<int>(root.state)) +
       " root_proven=" + std::to_string(root.proven) + " root_inflight=" + std::to_string(root.inflight) +
       " edges=" + std::to_string(root.edge_count) + " gumbel_ready=" + std::to_string(gumbel_ready_) + "\n";
  if (gumbel_ready_) {
    s += "gumbel_set:";
    for (int i : gumbel_set_) s += " " + to_string(edges_[root.edge_begin + i].action) + "(t=" + std::to_string(gumbel_target_[i]) + ")";
    s += "\n";
  }
  for (uint32_t i = 0; i < root.edge_count && i < 40; ++i) {
    const Edge& e = edges_[root.edge_begin + i];
    s += "  " + to_string(e.action) + " p=" + std::to_string(e.prior);
    if (e.child >= 0) {
      const Node& c = nodes_[e.child];
      s += " n=" + std::to_string(c.visits) + " inflight=" + std::to_string(c.inflight) + " proven=" +
           std::to_string(c.proven) + " state=" + std::to_string(static_cast<int>(c.state));
    }
    s += "\n";
  }
  return s;
}

bool Tree::check_invariants(std::string* why) const {
  auto fail = [&](const std::string& m) {
    if (why) *why = m;
    return false;
  };
  if (pending_.empty() && nodes_[0].visits != completed_) {
    return fail("root visits " + std::to_string(nodes_[0].visits) + " != completed " + std::to_string(completed_));
  }
  for (size_t id = 0; id < nodes_.size(); ++id) {
    const Node& n = nodes_[id];
    if (n.inflight < 0) return fail("negative inflight at node " + std::to_string(id));
    if (n.state != NodeState::EXPANDED) continue;
    int child_visits = 0;
    float prior_sum = 0;
    for (uint32_t i = 0; i < n.edge_count; ++i) {
      const Edge& e = edges_[n.edge_begin + i];
      prior_sum += e.prior;
      if (e.child >= 0) child_visits += nodes_[e.child].visits;
    }
    if (n.edge_count > 0 && std::abs(prior_sum - 1.0f) > 1e-3f) return fail("priors do not sum to 1");
    // A solved node may be re-visited without descending (its own visits grow, children's do not).
    if (pending_.empty() && n.proven == kUnknown && n.visits != 1 + child_visits) {
      return fail("node " + std::to_string(id) + " visits " + std::to_string(n.visits) + " != 1 + " +
                  std::to_string(child_visits));
    }
    if (n.proven != kUnknown && n.visits < 1 + child_visits) return fail("solved node lost visits");
  }
  return true;
}

SearchResult Tree::result() const {
  SearchResult r;
  r.stalled = stalled_;
  r.simulations = completed_;
  r.nodes = static_cast<int>(nodes_.size());
  r.proof_jobs = proof_jobs_;
  r.proof_found = proof_found_;
  r.proof_nodes = proof_nodes_;
  r.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
  r.actor = root_state_.actor();
  const float a = sign_of(r.actor);
  const Node& root = nodes_[0];
  r.value = root.visits > 0 ? root.value_sum * a / root.visits : 0.0f;
  if (root.proven != kUnknown) {
    r.proven = static_cast<int>(std::lround(root.proven * a));
    r.proof_source = "tree";
  }
  if (root_proven_by_tactics_ != kUnknown) {
    r.proven = root_proven_by_tactics_;
    r.proof_source = "root_tactics";
  }
  if (root_proven_by_tactics_ == 1 && root_tactics_move_) {
    r.action = *root_tactics_move_;
    r.value = 1.0f;
    r.pv = {r.action};
    return r;
  }
  const int fe = final_root_edge();
  if (fe < 0) {
    // Root solved before expansion (e.g. no defense): play the block if one exists.
    const auto acts = root_state_.legal_actions();
    if (!acts.empty()) r.action = acts.front();
    if (root_state_.phase() == OpeningPhase::NORMAL_RENJU) {
      std::vector<Point> pts;
      tactics::five_points(root_state_.board(), opposite(root_state_.side_to_move()), rules_of(root_state_), pts);
      for (Point p : pts) {
        if (root_state_.is_legal(Action::place(p))) { r.action = Action::place(p); break; }
      }
    }
    return r;
  }
  r.action = edges_[root.edge_begin + fe].action;
  if (cfg_.root == RootMode::GUMBEL) {
    const Edge& chosen = edges_[root.edge_begin + fe];
    if (chosen.child >= 0 && nodes_[chosen.child].visits > 0) {
      r.value = nodes_[chosen.child].value_sum * a / nodes_[chosen.child].visits;
    }
  }
  std::vector<int> order(root.edge_count);
  for (int i = 0; i < root.edge_count; ++i) order[i] = i;
  auto visits = [&](int i) {
    const Edge& e = edges_[root.edge_begin + i];
    return e.child >= 0 ? nodes_[e.child].visits : 0;
  };
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return visits(x) > visits(y); });
  int total = 0;
  for (int i = 0; i < root.edge_count; ++i) total += visits(i);
  std::vector<float> improved;
  if (cfg_.root == RootMode::GUMBEL && gumbel_ready_) {
    const float vmix = (r.value + 1.0f) * 0.5f;
    improved.resize(root.edge_count);
    float mx = kNegInf;
    for (int i = 0; i < root.edge_count; ++i) {
      const Edge& e = edges_[root.edge_begin + i];
      const float q = visits(i) > 0 ? (nodes_[e.child].value_sum * a / visits(i) + 1.0f) * 0.5f : vmix;
      improved[i] = e.logit + sigma(q);
      mx = std::max(mx, improved[i]);
    }
    float s = 0;
    for (float& x : improved) { x = std::exp(x - mx); s += x; }
    for (float& x : improved) x /= s;
  }
  for (int i : order) {
    const Edge& e = edges_[root.edge_begin + i];
    ChildStat c;
    c.action = e.action;
    c.prior = e.prior;
    c.visits = visits(i);
    if (c.visits > 0) c.q = nodes_[e.child].value_sum * a / c.visits;
    if (e.child >= 0 && nodes_[e.child].proven != kUnknown) c.proven = static_cast<int>(std::lround(nodes_[e.child].proven * a));
    r.children.push_back(c);
    r.policy_target.push_back(improved.empty() ? (total > 0 ? static_cast<float>(c.visits) / total : 0.0f) : improved[i]);
  }
  // Principal variation: most visited child at each level.
  int32_t id = 0;
  for (int depth = 0; depth < 40; ++depth) {
    const Node& n = nodes_[id];
    if (n.state != NodeState::EXPANDED || n.edge_count == 0) break;
    int best = -1, bn = 0;
    for (uint32_t i = 0; i < n.edge_count; ++i) {
      const Edge& e = edges_[n.edge_begin + i];
      if (e.child >= 0 && nodes_[e.child].visits > bn) { bn = nodes_[e.child].visits; best = static_cast<int>(i); }
    }
    if (best < 0) break;
    const Edge& e = edges_[n.edge_begin + best];
    r.pv.push_back(e.action);
    id = e.child;
  }
  return r;
}

}  // namespace dz::search
