// FROZEN reference copy of cpp/tactics/threats.cpp at ee1d011 (differential oracle, Phase 8 E3b).
// Do not optimize or edit; regenerate only with an explicit decision.
#include "tactics/reference/ref.h"

#include <algorithm>

#include "board/zobrist.h"

namespace dz::tactics::ref {
namespace {

// Own stones adjacent to (empty) idx on both sides along step, i.e. run length minus one if a
// stone were placed at idx.
inline int neighbours(const uint8_t* c, int idx, int step, uint8_t own) {
  int n = 0;
  for (int i = idx + step; c[i] == own; i += step) ++n;
  for (int i = idx - step; c[i] == own; i -= step) ++n;
  return n;
}

// Own stones in the 9-cell window centred on idx along step (excluding idx).
inline int window_count(const uint8_t* c, int idx, int step, uint8_t own) {
  int n = 0;
  for (int k = 1; k <= 4; ++k) {
    n += c[idx + k * step] == own ? 1 : 0;
    n += c[idx - k * step] == own ? 1 : 0;
  }
  return n;
}

inline bool window_has(const uint8_t* c, int idx, uint8_t own, int min_stones) {
  for (int d = 0; d < kNumDirections; ++d) {
    if (window_count(c, idx, Board::kStep[d], own) >= min_stones) return true;
  }
  return false;
}

bool default_black_options(const BlackRuleOptions& o) {
  return o.exact_five_precedence && o.five_beats_overline_other_dir;
}

}  // namespace





bool is_legal_move(const Board& b, Point p, Color c, const Rules& r) {
  if (!b.empty_at(p)) return false;
  return c == Color::WHITE || !is_forbidden(black_verdict(b, p, r.black));
}

bool wins_with(const Board& b, Point p, Color c, const Rules& r) {
  if (!b.empty_at(p)) return false;
  const uint8_t* cells = b.cells().data();
  const int idx = Board::index_of(p);
  const uint8_t own = cell_of(c);
  bool exact = false;
  for (int d = 0; d < kNumDirections; ++d) {
    const int len = 1 + neighbours(cells, idx, Board::kStep[d], own);
    if (c == Color::WHITE && len >= 5) return true;
    if (len == 5) exact = true;
  }
  if (c == Color::WHITE || !exact) return false;
  if (default_black_options(r.black)) return true;  // exact five wins (RULES.md §6)
  return black_verdict(b, p, r.black) == MoveVerdict::WIN_FIVE;
}

void five_points(const Board& b, Color c, const Rules& r, std::vector<Point>& out) {
  out.clear();
  const uint8_t* cells = b.cells().data();
  const uint8_t own = cell_of(c);
  for (int i = 0; i < kNumPoints; ++i) {
    const Point p = static_cast<Point>(i);
    if (!b.empty_at(p)) continue;
    const int idx = Board::index_of(p);
    bool candidate = false;
    for (int d = 0; d < kNumDirections && !candidate; ++d) {
      candidate = neighbours(cells, idx, Board::kStep[d], own) >= 4;
    }
    if (candidate && ref::wins_with(b, p, c, r)) out.push_back(p);
  }
}

namespace {

// Five points of `own` on the line through idx (stone of `own` at idx) in one direction, judged
// in that direction only. Valid when `own` had no five point before the stone at idx: a new five
// point must then complete a five through idx, i.e. along this very line.
inline int line_five_points(const uint8_t* c, int idx, int step, uint8_t own, bool white, int* out) {
  int n = 0;
  for (int k = -4; k <= 4; ++k) {
    if (k == 0) continue;
    const int e = idx + k * step;
    if (c[e] != EMPTY) continue;
    const int len = 1 + neighbours(c, e, step, own);
    if (white ? len >= 5 : len == 5) out[n++] = e;
  }
  return n;
}

// New five points created by the stone at idx (cells must already contain it), deduplicated.
// Black with non-default options re-checks each candidate with the full verdict.
int new_five_points(Board::Cells& cells, int idx, Color c, const Rules& r, int* out) {
  const uint8_t own = cell_of(c);
  const bool white = c == Color::WHITE;
  int n = 0;
  int buf[8];
  for (int d = 0; d < kNumDirections; ++d) {
    const int step = Board::kStep[d];
    if (window_count(cells.data(), idx, step, own) < 3) continue;  // fewer than 4 stones in any five
    const int k = line_five_points(cells.data(), idx, step, own, white, buf);
    for (int j = 0; j < k; ++j) {
      if (std::find(out, out + n, buf[j]) != out + n) continue;
      if (!white && !default_black_options(r.black) &&
          black_verdict_cells(cells, buf[j], r.black) != MoveVerdict::WIN_FIVE) {
        continue;
      }
      out[n++] = buf[j];
    }
  }
  return n;
}

}  // namespace

void five_points_near(const Board& b, Point center, Color c, const Rules& r, std::vector<Point>& out) {
  out.clear();
  Board::Cells cells = b.cells();
  int pts[32];
  const int n = new_five_points(cells, Board::index_of(center), c, r, pts);
  for (int i = 0; i < n; ++i) out.push_back(Board::point_of_index(pts[i]));
}

namespace {

// Moves of c with at least `min_points` new five points; sorted by that count, descending.
// Works on one scratch copy of the cells (no Board updates, no hashing).
void moves_with_five_points(Board& b, Color c, const Rules& r, int min_points, std::vector<Point>& out) {
  out.clear();
  std::vector<std::pair<int, Point>> scored;
  const uint8_t own = cell_of(c);
  Board::Cells cells = b.cells();
  int pts[32];
  for (int i = 0; i < kNumPoints; ++i) {
    const Point m = static_cast<Point>(i);
    const int idx = Board::index_of(m);
    if (cells[idx] != EMPTY) continue;
    if (!window_has(cells.data(), idx, own, 3)) continue;
    if (ref::wins_with(b, m, c, r)) continue;  // a win, not a four
    cells[idx] = own;
    const int n = new_five_points(cells, idx, c, r, pts);
    cells[idx] = EMPTY;
    if (n < min_points) continue;
    if (c == Color::BLACK && is_forbidden(black_verdict_cells(cells, idx, r.black))) continue;
    scored.emplace_back(n, m);
  }
  std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b2) { return a.first > b2.first; });
  for (const auto& [n, m] : scored) out.push_back(m);
}

}  // namespace

void four_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out) {
  moves_with_five_points(b, c, r, 1, out);
}

void double_threat_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out) {
  moves_with_five_points(b, c, r, 2, out);
}

void threat_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out) {
  ref::four_moves(b, c, r, out);
  const uint8_t own = cell_of(c);
  for (int i = 0; i < kNumPoints; ++i) {
    const Point m = static_cast<Point>(i);
    if (!b.empty_at(m)) continue;
    if (!window_has(b.cells().data(), Board::index_of(m), own, 2)) continue;
    if (std::find(out.begin(), out.end(), m) != out.end()) continue;
    if (ref::wins_with(b, m, c, r) || !ref::is_legal_move(b, m, c, r)) continue;
    if (count_threes(b, m, c, r.black) >= 1) out.push_back(m);
  }
}

}  // namespace dz::tactics::ref
