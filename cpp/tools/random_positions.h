// Random position generator biased towards dense black clusters, shared by tests and tools
// (differential fuzzing of the forbidden checker).
#pragma once

#include <random>

#include "board/board.h"
#include "game/game_state.h"

namespace dz::tools {

inline Board random_dense_board(std::mt19937_64& rng) {
  Board b;
  std::uniform_int_distribution<int> coin(0, 99);
  std::uniform_int_distribution<int> any(0, kNumPoints - 1);
  const int cr = 3 + static_cast<int>(rng() % 9), cc = 3 + static_cast<int>(rng() % 9);
  const int radius = 2 + static_cast<int>(rng() % 4);
  const int n_black = 4 + static_cast<int>(rng() % 22);
  const int n_white = static_cast<int>(rng() % (n_black / 2 + 3));
  auto place_near = [&](Color c) {
    for (int tries = 0; tries < 50; ++tries) {
      int r, col;
      if (coin(rng) < 85) {
        r = cr + static_cast<int>(rng() % (2 * radius + 1)) - radius;
        col = cc + static_cast<int>(rng() % (2 * radius + 1)) - radius;
      } else {
        const Point p = static_cast<Point>(any(rng));
        r = row_of(p);
        col = col_of(p);
      }
      if (!on_board(r, col)) continue;
      const Point p = make_point(r, col);
      if (!b.empty_at(p)) continue;
      b.place(p, c);
      return;
    }
  };
  for (int i = 0; i < n_black; ++i) place_near(Color::BLACK);
  for (int i = 0; i < n_white; ++i) place_near(Color::WHITE);
  return b;
}

// Empty point with a black stone within Chebyshev distance 2 (others are trivially LEGAL).
inline bool near_black(const Board& b, Point p) {
  for (int dr = -2; dr <= 2; ++dr) {
    for (int dc = -2; dc <= 2; ++dc) {
      const int r = row_of(p) + dr, c = col_of(p) + dc;
      if (on_board(r, c) && b.at(make_point(r, c)) == BLACK_STONE) return true;
    }
  }
  return false;
}

// NORMAL_RENJU game after random play biased to moves near existing stones (tactically rich).
// Returns the last non-terminal state reached.
inline GameState random_midgame(std::mt19937_64& rng, ProfilePtr profile, int min_plies, int max_plies) {
  GameState g(std::move(profile));
  const int plies = min_plies + static_cast<int>(rng() % (max_plies - min_plies + 1));
  std::vector<Action> legal, near;
  for (int i = 0; i < plies; ++i) {
    g.legal_actions(legal);
    if (legal.empty()) break;
    // Opening decisions (swap / branch / proposals / selection): uniformly random.
    if (g.phase() != OpeningPhase::NORMAL_RENJU && !is_placement_phase(g.phase())) {
      g.apply(legal[rng() % legal.size()]);
      --i;  // decisions do not count as plies
      continue;
    }
    near.clear();
    for (const Action& a : legal) {
      if (a.type != ActionType::PLACE) continue;
      bool close = g.board().stone_count() == 0 && a.point == kCenterPoint;
      for (int dr = -2; dr <= 2 && !close; ++dr) {
        for (int dc = -2; dc <= 2 && !close; ++dc) {
          const int r = row_of(a.point) + dr, c = col_of(a.point) + dc;
          close = on_board(r, c) && !g.board().empty_at(make_point(r, c));
        }
      }
      if (close) near.push_back(a);
    }
    if (near.empty()) break;
    const Action a = near[rng() % near.size()];
    g.apply(a);
    if (g.is_terminal()) {
      g.undo();
      break;
    }
  }
  return g;
}

}  // namespace dz::tools
