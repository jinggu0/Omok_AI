#include "search/bots.h"

#include <cstdlib>
#include <vector>

#include "tactics/threats.h"

namespace dz::search {

Action random_action(const GameState& g, std::mt19937_64& rng) {
  const auto acts = g.legal_actions();
  return acts[rng() % acts.size()];
}

Action heuristic_action(const GameState& g, std::mt19937_64& rng) {
  const auto acts = g.legal_actions();
  if (g.phase() != OpeningPhase::NORMAL_RENJU) {
    if (!is_placement_phase(g.phase())) return acts[rng() % acts.size()];
    // Opening placement: closest to the centre, random tie-break.
    int best_d = 1 << 30;
    std::vector<Action> best;
    for (const Action& a : acts) {
      if (a.type != ActionType::PLACE) continue;
      const int d = std::abs(row_of(a.point) - 7) + std::abs(col_of(a.point) - 7);
      if (d < best_d) { best_d = d; best.clear(); }
      if (d == best_d) best.push_back(a);
    }
    return best.empty() ? acts.front() : best[rng() % best.size()];
  }
  const Color me = g.side_to_move(), opp = opposite(me);
  const tactics::Rules r{g.profile().black, g.profile().pass_allowed};
  const Board& b = g.board();
  std::vector<Point> pts;
  tactics::five_points(b, me, r, pts);
  if (!pts.empty()) return Action::place(pts.front());
  tactics::five_points(b, opp, r, pts);
  for (Point p : pts) {
    if (g.is_legal(Action::place(p))) return Action::place(p);
  }
  if (b.stone_count() == 0) return Action::place(kCenterPoint);
  std::uniform_real_distribution<double> noise(0.0, 1.0);
  double best_score = -1;
  Action best = acts.front();
  for (const Action& a : acts) {
    if (a.type != ActionType::PLACE) continue;
    const Point p = a.point;
    int own_adj = 0, opp_adj = 0;
    for (int dr = -2; dr <= 2; ++dr) {
      for (int dc = -2; dc <= 2; ++dc) {
        const int rr = row_of(p) + dr, cc = col_of(p) + dc;
        if (!on_board(rr, cc) || (dr == 0 && dc == 0)) continue;
        const Cell c = b.at(make_point(rr, cc));
        if (c == cell_of(me)) ++own_adj;
        if (c == cell_of(opp)) ++opp_adj;
      }
    }
    if (own_adj + opp_adj == 0) continue;
    double s = 1000.0 * count_fours(b, p, me) + 100.0 * count_threes(b, p, me, r.black) +
               800.0 * count_fours(b, p, opp) + 80.0 * count_threes(b, p, opp, r.black) + 2.0 * own_adj + opp_adj;
    s += noise(rng);
    if (s > best_score) {
      best_score = s;
      best = a;
    }
  }
  return best;
}

}  // namespace dz::search
