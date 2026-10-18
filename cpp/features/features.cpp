#include "features/features.h"

#include <algorithm>
#include <cstring>

#include "tactics/threats.h"

namespace dz::features {
namespace {

constexpr int N = kNumPoints;

void set_points(uint8_t* plane, const std::vector<Point>& pts) {
  for (Point p : pts) plane[p] = 1;
}

void fill(uint8_t* plane, uint8_t v) { std::memset(plane, v, N); }

tactics::Rules rules_of(const GameState& g) {
  return tactics::Rules{g.profile().black, g.profile().pass_allowed};
}

// Threat masks of color c: five points, four moves, three-only moves.
void threat_masks(const GameState& g, Color c, uint8_t* five, uint8_t* four, uint8_t* three) {
  const tactics::Rules r = rules_of(g);
  Board b = g.board();
  std::vector<Point> pts;
  tactics::five_points(b, c, r, pts);
  set_points(five, pts);
  tactics::four_moves(b, c, r, pts);
  set_points(four, pts);
  tactics::threat_moves(b, c, r, pts);
  for (Point p : pts) {
    if (!four[p]) three[p] = 1;
  }
}

void forbidden_mask(const GameState& g, uint8_t* out) {
  Board::Cells scratch = g.board().cells();
  for (int i = 0; i < N; ++i) {
    const Point p = static_cast<Point>(i);
    if (!g.board().empty_at(p)) continue;
    out[i] = is_forbidden(black_verdict_cells(scratch, Board::index_of(p), g.profile().black)) ? 1 : 0;
  }
}

}  // namespace

int FeatureConfig::num_planes() const { return static_cast<int>(plane_names().size()); }

std::vector<std::string> FeatureConfig::plane_names() const {
  std::vector<std::string> n = {"actor_stones", "opponent_stones", "black_stones", "white_stones", "last_move"};
  for (int k = 0; k < history_planes; ++k) n.push_back("history_" + std::to_string(k + 1));
  n.push_back("legal");
  if (forbidden_plane) n.push_back("black_forbidden");
  n.insert(n.end(), {"actor_is_black", "placement_region", "offered", "chosen_fifth"});
  for (int i = 0; i < kNumOpeningPhases; ++i) {
    n.push_back(std::string("phase_") + phase_name(static_cast<OpeningPhase>(i)));
  }
  if (tactical_planes) {
    n.insert(n.end(), {"actor_five", "actor_four", "actor_three", "opp_five", "opp_four", "opp_three"});
  }
  return n;
}

uint64_t FeatureConfig::hash() const {
  uint64_t h = 0xCBF29CE484222325ULL;
  auto mix = [&](const std::string& s) {
    for (unsigned char ch : s) { h ^= ch; h *= 0x100000001B3ULL; }
    h ^= 0xFF;
    h *= 0x100000001B3ULL;
  };
  mix("v" + std::to_string(version));
  for (const auto& s : plane_names()) mix(s);
  return h;
}

void legal_policy_mask(const GameState& g, uint8_t* out) {
  std::memset(out, 0, kNumPolicyActions);
  for (const Action& a : g.legal_actions()) {
    if (a.has_point()) out[a.point] = 1;
    if (a.type == ActionType::PASS) out[kPassIndex] = 1;
  }
}

void encode(const GameState& g, const FeatureConfig& cfg, uint8_t* out) {
  std::memset(out, 0, static_cast<size_t>(cfg.num_planes()) * N);
  uint8_t* plane = out;
  auto next = [&]() {
    uint8_t* p = plane;
    plane += N;
    return p;
  };
  const Color actor_color = g.assignment().color_of(g.actor());
  const Board& b = g.board();
  uint8_t* actor = next();
  uint8_t* opp = next();
  uint8_t* black = next();
  uint8_t* white = next();
  for (int i = 0; i < N; ++i) {
    const Cell c = b.at(static_cast<Point>(i));
    if (c == EMPTY) continue;
    const Color cc = c == BLACK_STONE ? Color::BLACK : Color::WHITE;
    (cc == actor_color ? actor : opp)[i] = 1;
    (cc == Color::BLACK ? black : white)[i] = 1;
  }
  uint8_t* last = next();
  if (g.last_move() != kNoPoint) last[g.last_move()] = 1;
  if (cfg.history_planes > 0) {
    std::vector<Point> placed;
    for (const Action& a : g.action_history()) {
      if (a.type == ActionType::PLACE || a.type == ActionType::SELECT_PROPOSED_FIFTH) placed.push_back(a.point);
    }
    // placed.back() is the last move (already encoded); history_k = k-th placement before it.
    for (int k = 0; k < cfg.history_planes; ++k) {
      uint8_t* h = next();
      const int idx = static_cast<int>(placed.size()) - 2 - k;
      if (idx >= 0) h[placed[idx]] = 1;
    }
  }
  {
    uint8_t mask[kNumPolicyActions];
    legal_policy_mask(g, mask);
    std::memcpy(next(), mask, N);  // point actions only; pass is not a spatial feature
  }
  if (cfg.forbidden_plane) forbidden_mask(g, next());
  fill(next(), actor_color == Color::BLACK ? 1 : 0);
  {
    uint8_t* region = next();
    int n = 0;
    if (g.phase() == OpeningPhase::OFFER_TEN_5TH_MOVES) n = g.profile().taraguchi.proposal_region;
    else if (g.phase() != OpeningPhase::NORMAL_RENJU) n = placement_region(g.phase(), g.profile().taraguchi);
    for (int i = 0; i < N; ++i) region[i] = (n == 0 || in_central(static_cast<Point>(i), n)) ? 1 : 0;
  }
  {
    uint8_t* offered = next();
    if (g.phase() == OpeningPhase::OFFER_TEN_5TH_MOVES || g.phase() == OpeningPhase::SELECT_ONE_OF_TEN) {
      for (Point q : g.offered()) offered[q] = 1;
    }
  }
  {
    uint8_t* chosen = next();
    if (g.chosen_fifth() != kNoPoint) chosen[g.chosen_fifth()] = 1;
  }
  for (int i = 0; i < kNumOpeningPhases; ++i) {
    uint8_t* ph = next();
    if (static_cast<int>(g.phase()) == i) fill(ph, 1);
  }
  if (cfg.tactical_planes) {
    uint8_t* af = next();
    uint8_t* a4 = next();
    uint8_t* a3 = next();
    uint8_t* of = next();
    uint8_t* o4 = next();
    uint8_t* o3 = next();
    if (g.phase() == OpeningPhase::NORMAL_RENJU && !g.is_terminal()) {
      threat_masks(g, actor_color, af, a4, a3);
      threat_masks(g, opposite(actor_color), of, o4, o3);
    }
  }
}

void aux_targets(const GameState& g, uint8_t* out) {
  std::memset(out, 0, static_cast<size_t>(kNumAuxPlanes) * N);
  forbidden_mask(g, out + 4 * N);
  if (g.phase() != OpeningPhase::NORMAL_RENJU || g.is_terminal()) return;
  const Color me = g.side_to_move();
  const tactics::Rules r = rules_of(g);
  Board b = g.board();
  std::vector<Point> pts;
  tactics::five_points(b, me, r, pts);
  set_points(out, pts);
  tactics::five_points(b, opposite(me), r, pts);
  set_points(out + N, pts);
  uint8_t unused[N] = {};
  threat_masks(g, me, unused, out + 2 * N, out + 3 * N);
}

}  // namespace dz::features
