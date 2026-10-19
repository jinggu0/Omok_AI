// MCTS with PUCT or Gumbel root (ARCHITECTURE.md §3.10).
//
// Value convention: every value stored in the tree is from P0's perspective in [-1, 1]
// (P(win) - P(loss) for P0). Selection converts to the acting player's perspective at each node.
// Opening phases let the same player act several times in a row (e.g. CHOICE_AFTER_4 ->
// SWAP_BEFORE_DIRECT_5 -> BLACK_DIRECT_MOVE_5), so no negamax sign flipping is ever used.
//
// Two ways to drive a search:
//  * synchronous: Tree::run(Evaluator&) evaluates each leaf immediately (tests, C++ evaluators);
//  * batched: Tree::select() returns leaves whose features are written to a caller buffer, the
//    caller evaluates many leaves (from many trees) at once and calls Tree::complete().
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "features/features.h"
#include "game/game_state.h"

namespace dz::search {

enum class RootMode : uint8_t { PUCT, GUMBEL };

struct SearchConfig {
  int simulations = 200;
  float c_puct = 1.5f;
  float fpu_reduction = 0.25f;
  RootMode root = RootMode::PUCT;
  int gumbel_m = 16;             // root candidates for Sequential Halving
  float gumbel_c_visit = 50.0f;
  float gumbel_c_scale = 1.0f;
  bool gumbel_noise = true;      // false: g = 0 (deterministic, for evaluation)
  bool leaf_tactics = true;      // five points / forced block / no defense at every leaf
  uint64_t leaf_vcf_nodes = 0;   // optional VCF probe at leaves (0 = off)
  uint64_t root_tactics_nodes = 20000;  // analyze_tactics at the root before searching (0 = off)
  int max_inflight = 8;          // leaves pending evaluation per tree (virtual loss)
  float virtual_loss = 1.0f;
  uint64_t seed = 1;
};

// Raw network outputs; the tree picks the head that matches each node's action types.
struct Evaluation {
  std::array<float, features::kNumPolicyActions> policy{};  // PLACE points + PASS
  std::array<float, 3> wdl{};                                // actor's W/D/L logits
  std::array<float, 2> swap{};                               // SWAP, KEEP_COLOR
  std::array<float, 2> branch{};                             // CHOOSE_DIRECT_FIFTH, CHOOSE_TEN
  std::array<float, kNumPoints> propose{};
  std::array<float, kNumPoints> select{};
};

// Synchronous evaluator interface (C++ evaluators, tests).
class Evaluator {
 public:
  virtual ~Evaluator() = default;
  virtual Evaluation evaluate(const GameState& g) = 0;
};

class UniformEvaluator final : public Evaluator {
 public:
  Evaluation evaluate(const GameState&) override { return {}; }  // equal logits, value 0
};

class LambdaEvaluator final : public Evaluator {
 public:
  explicit LambdaEvaluator(std::function<Evaluation(const GameState&)> f) : f_(std::move(f)) {}
  Evaluation evaluate(const GameState& g) override { return f_(g); }
 private:
  std::function<Evaluation(const GameState&)> f_;
};

struct ChildStat {
  Action action;
  float prior = 0;
  int visits = 0;
  float q = 0;          // mean value for the root actor, 0 if unvisited
  int proven = 2;       // value for the root actor: +1 win, 0 draw, -1 loss, 2 unknown
};

struct SearchResult {
  Action action;                     // chosen action
  PlayerId actor = PlayerId::P0;     // root actor
  // Root value for the root actor: PUCT = mean over root visits; GUMBEL = Q of the chosen action
  // (Sequential Halving spreads visits on purpose, so the root mean is not a value estimate).
  float value = 0;
  int proven = 2;                    // root proven value for the root actor (+1/0/-1), 2 unknown
  std::string proof_source;          // "", "root_tactics", "tree"
  std::vector<ChildStat> children;   // sorted by visits (desc)
  std::vector<float> policy_target;  // visit distribution (PUCT) or improved policy (Gumbel), per child order
  std::vector<Action> pv;
  int simulations = 0;
  int nodes = 0;
  bool stalled = false;              // synchronous run stopped with nothing selectable (must never happen)
  double elapsed_ms = 0;
};

class Tree {
 public:
  Tree(const GameState& root, const SearchConfig& cfg, const features::FeatureConfig& fcfg = {});

  // Synchronous search to completion.
  SearchResult run(Evaluator& eval);

  // Batched search. select() runs one selection; if the leaf needs a network evaluation its
  // features are written to `features_out` (num_planes*225 bytes) and a ticket >= 0 is returned.
  // Returns -1 if the simulation was resolved without the network (terminal / proven / tactics),
  // and -2 if no leaf can be selected right now (search done, or every path is in flight).
  int select(uint8_t* features_out);
  void complete(int ticket, const Evaluation& e);
  bool done() const;
  int inflight() const { return static_cast<int>(pending_.size()); }

  SearchResult result() const;
  int completed() const { return completed_; }
  // Tree invariants (tests): with nothing in flight, every expanded node has
  // visits == 1 + sum(child visits) (+ re-visits of solved nodes), the root has `completed` visits,
  // priors sum to 1, and no node has negative in-flight counts.
  bool check_invariants(std::string* why) const;
  // Root diagnostics (state, flags, Gumbel set, per-child visits/inflight/proven/state).
  std::string debug_string() const;

 private:
  static constexpr int8_t kUnknown = 2;
  enum class NodeState : uint8_t { NEW, PENDING, EXPANDED, SOLVED };

  struct Edge {
    Action action;
    float logit = 0;
    float prior = 0;
    int32_t child = -1;
  };
  struct Node {
    int32_t parent = -1;
    uint32_t edge_begin = 0;
    uint16_t edge_count = 0;
    int32_t visits = 0;
    int32_t inflight = 0;
    float value_sum = 0;  // P0 perspective
    PlayerId actor = PlayerId::P0;
    NodeState state = NodeState::NEW;
    int8_t proven = kUnknown;  // P0 perspective
  };
  struct Pending {
    std::vector<int32_t> path;        // node ids root..leaf
    std::vector<Action> actions;      // legal (possibly tactically restricted) actions at the leaf
  };

  // Shared selection; on a leaf needing evaluation `on_leaf` gets the leaf state (state is
  // restored afterwards). Returns ticket / -1 / -2 as select().
  int descend(const std::function<void(const GameState&, Pending&)>& on_leaf);
  void expand(int32_t node, const std::vector<Action>& actions, const Evaluation& e);
  void backup(const std::vector<int32_t>& path, float value_p0, bool had_inflight);
  void update_proven(int32_t node);
  int choose_edge(int32_t node);
  int choose_root_gumbel();
  void init_gumbel();
  float q_actor(const Node& child, PlayerId actor) const;
  float sigma(float qhat) const;
  int32_t new_node(int32_t parent);
  std::vector<Action> leaf_actions(const GameState& g, int8_t& proven_p0) const;
  int final_root_edge() const;

  GameState root_state_;
  GameState work_;
  SearchConfig cfg_;
  features::FeatureConfig fcfg_;
  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<Pending> pending_slots_;
  std::vector<int> free_slots_;
  std::vector<int> pending_;  // active ticket ids
  int completed_ = 0;
  std::mt19937_64 rng_;
  std::chrono::steady_clock::time_point start_;
  int root_proven_by_tactics_ = kUnknown;  // root actor perspective
  bool stalled_ = false;
  std::optional<Action> root_tactics_move_;

  // Gumbel root state
  bool gumbel_ready_ = false;
  std::vector<float> gumbel_;           // per root edge
  std::vector<int> gumbel_set_;         // current candidate edge indices (relative to root edges)
  std::vector<int> gumbel_target_;      // visit target per root edge in the current schedule
  int gumbel_phase_budget_ = 0;
};

// Policy over a node's actions from network heads (softmax over the matching logits).
float action_logit(const Evaluation& e, const Action& a);

}  // namespace dz::search
