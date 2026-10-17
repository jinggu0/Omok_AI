#include "rules/reference_rules.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <vector>

namespace dz::ref {
namespace {

enum : uint8_t { kE = 0, kB = 1, kW = 2, kOff = 3 };

struct RefBoard {
  std::array<uint8_t, kNumPoints> c{};
  uint8_t at(int row, int col) const { return on_board(row, col) ? c[row * kBoardSize + col] : kOff; }
  void set(int row, int col, uint8_t v) { c[row * kBoardSize + col] = v; }
};

RefBoard from_board(const Board& b) {
  RefBoard rb;
  for (int p = 0; p < kNumPoints; ++p) {
    const Cell cell = b.at(static_cast<Point>(p));
    rb.c[p] = cell == BLACK_STONE ? kB : cell == WHITE_STONE ? kW : kE;
  }
  return rb;
}

// Length of the black run through (row, col) along direction d; (row, col) must be black.
int black_run(const RefBoard& b, int row, int col, int d) {
  const int dr = kDirDRow[d], dc = kDirDCol[d];
  int n = 1;
  for (int i = 1; b.at(row + i * dr, col + i * dc) == kB; ++i) ++n;
  for (int i = 1; b.at(row - i * dr, col - i * dc) == kB; ++i) ++n;
  return n;
}

using StoneSet = std::vector<int>;

// FOUR (RULES.md §3.3), literal window form: every 5-cell window on the line through (row, col)
// that contains (row, col) and holds exactly 4 black + 1 empty, where filling the empty cell
// gives an exact five. Returns stone set S -> its completion points.
std::map<StoneSet, std::set<int>> black_four_sets(const RefBoard& b, int row, int col, int d) {
  std::map<StoneSet, std::set<int>> out;
  const int dr = kDirDRow[d], dc = kDirDCol[d];
  for (int s = -4; s <= 0; ++s) {
    int blacks = 0, empties = 0, er = -1, ec = -1;
    bool valid = true;
    StoneSet stones;
    for (int i = 0; i < 5; ++i) {
      const int r = row + (s + i) * dr, c = col + (s + i) * dc;
      const uint8_t v = b.at(r, c);
      if (v == kB) { ++blacks; stones.push_back(r * kBoardSize + c); }
      else if (v == kE) { ++empties; er = r; ec = c; }
      else { valid = false; break; }
    }
    if (!valid || blacks != 4 || empties != 1) continue;
    RefBoard filled = b;
    filled.set(er, ec, kB);
    if (black_run(filled, er, ec, d) != 5) continue;  // exact five only
    std::sort(stones.begin(), stones.end());
    out[stones].insert(er * kBoardSize + ec);
  }
  return out;
}

MoveVerdict verdict_impl(const RefBoard& b, int row, int col, const BlackRuleOptions& opt);

// STRAIGHT FOUR through both p and e in direction d (black stones at both already placed):
// some four set containing p (and e) with two distinct completion points.
bool straight_four(const RefBoard& b, int pr, int pc, int er, int ec, int d) {
  const int p = pr * kBoardSize + pc;
  for (const auto& [stones, completions] : black_four_sets(b, er, ec, d)) {
    if (completions.size() >= 2 && std::find(stones.begin(), stones.end(), p) != stones.end()) {
      return true;
    }
  }
  return false;
}

// Straight-four extension points of the three through (row, col) in direction d, on board b
// that already has black at (row, col). Directions with a five/overline/four are not threes.
std::vector<std::pair<int, int>> extensions(const RefBoard& b, int row, int col, int d) {
  std::vector<std::pair<int, int>> out;
  if (black_run(b, row, col, d) >= 5) return out;
  if (!black_four_sets(b, row, col, d).empty()) return out;  // ASSUMPTIONS A-R3
  const int dr = kDirDRow[d], dc = kDirDCol[d];
  for (int k = -4; k <= 4; ++k) {
    if (k == 0) continue;
    const int er = row + k * dr, ec = col + k * dc;
    if (b.at(er, ec) != kE) continue;
    RefBoard b3 = b;
    b3.set(er, ec, kB);
    if (straight_four(b3, row, col, er, ec, d)) out.emplace_back(er, ec);
  }
  return out;
}

// THREE (RULES.md §3.5): an extension that is a LEGAL black move on the board with p placed.
bool real_three(const RefBoard& with_p, int row, int col, int d, const BlackRuleOptions& opt) {
  for (const auto& [er, ec] : extensions(with_p, row, col, d)) {
    if (verdict_impl(with_p, er, ec, opt) == MoveVerdict::LEGAL) return true;
  }
  return false;
}

MoveVerdict verdict_impl(const RefBoard& b, int row, int col, const BlackRuleOptions& opt) {
  RefBoard with_p = b;
  with_p.set(row, col, kB);

  bool five = false, over = false;
  for (int d = 0; d < kNumDirections; ++d) {
    const int n = black_run(with_p, row, col, d);
    if (n == 5) five = true;
    if (n >= 6) over = true;
  }
  if (opt.exact_five_precedence) {
    if (five && (opt.five_beats_overline_other_dir || !over)) return MoveVerdict::WIN_FIVE;
    if (over) return MoveVerdict::FORBIDDEN_OVERLINE;
  } else if (over) {
    return MoveVerdict::FORBIDDEN_OVERLINE;
  }

  size_t fours = 0;
  for (int d = 0; d < kNumDirections; ++d) fours += black_four_sets(with_p, row, col, d).size();
  if (fours >= 2) return MoveVerdict::FORBIDDEN_DOUBLE_FOUR;

  int threes = 0;
  for (int d = 0; d < kNumDirections; ++d) {
    if (real_three(with_p, row, col, d, opt)) ++threes;
  }
  if (threes >= 2) return MoveVerdict::FORBIDDEN_DOUBLE_THREE;
  if (!opt.exact_five_precedence && five) return MoveVerdict::WIN_FIVE;
  return MoveVerdict::LEGAL;
}

}  // namespace

MoveVerdict black_verdict(const Board& board, Point p, const BlackRuleOptions& opt) {
  return verdict_impl(from_board(board), row_of(p), col_of(p), opt);
}

int count_black_fours(const Board& board, Point p) {
  RefBoard b = from_board(board);
  b.set(row_of(p), col_of(p), kB);
  size_t n = 0;
  for (int d = 0; d < kNumDirections; ++d) n += black_four_sets(b, row_of(p), col_of(p), d).size();
  return static_cast<int>(n);
}

int count_black_real_threes(const Board& board, Point p, const BlackRuleOptions& opt) {
  RefBoard b = from_board(board);
  b.set(row_of(p), col_of(p), kB);
  int n = 0;
  for (int d = 0; d < kNumDirections; ++d) n += real_three(b, row_of(p), col_of(p), d, opt) ? 1 : 0;
  return n;
}

int count_black_apparent_threes(const Board& board, Point p) {
  RefBoard b = from_board(board);
  b.set(row_of(p), col_of(p), kB);
  int n = 0;
  for (int d = 0; d < kNumDirections; ++d) n += extensions(b, row_of(p), col_of(p), d).empty() ? 0 : 1;
  return n;
}

}  // namespace dz::ref
