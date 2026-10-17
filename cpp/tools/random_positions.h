// Random position generator biased towards dense black clusters, shared by tests and tools
// (differential fuzzing of the forbidden checker).
#pragma once

#include <random>

#include "board/board.h"

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

}  // namespace dz::tools
