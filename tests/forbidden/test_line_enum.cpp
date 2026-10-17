// Exhaustive one-line enumeration (ARCHITECTURE.md §7): every pattern of empty/black/white cells
// within offsets [-5, 5] of the move on one horizontal line, for every column (so board edges are
// covered). Checks fast == oracle and gathers evidence for ASSUMPTIONS A-R3 and A-R4.
#include <gtest/gtest.h>

#include <set>

#include "common/test_util.h"
#include "rules/forbidden.h"
#include "rules/reference_rules.h"

namespace dz {
namespace {

constexpr int kRow = 7;

// Length-4 run through q and e (both black) with both ends empty and no black beyond: a straight
// four for black. Returns the run's stone columns or empty if not straight. Board must hold q, e.
std::vector<int> straight_four_cols(const Board& b, int qcol, int ecol) {
  auto black = [&](int c) { return on_board(kRow, c) && b.at(make_point(kRow, c)) == BLACK_STONE; };
  auto empty = [&](int c) { return on_board(kRow, c) && b.at(make_point(kRow, c)) == EMPTY; };
  int lo = ecol, hi = ecol;
  while (black(lo - 1)) --lo;
  while (black(hi + 1)) ++hi;
  if (hi - lo + 1 != 4 || qcol < lo || qcol > hi) return {};
  if (!empty(lo - 1) || !empty(hi + 1) || black(lo - 2) || black(hi + 2)) return {};
  std::vector<int> cols;
  for (int c = lo; c <= hi; ++c) cols.push_back(c);
  return cols;
}

TEST(LineEnumeration, AllOneLinePatternsWithinFive) {
  long long patterns = 0, mismatches = 0;
  long long ar3_four_and_three = 0, ar4_two_threes = 0;
  std::string ar3_example, ar4_example;

  for (int qcol = 0; qcol < kBoardSize; ++qcol) {
    std::vector<int> cols;
    for (int o = -5; o <= 5; ++o) {
      if (o != 0 && on_board(kRow, qcol + o)) cols.push_back(qcol + o);
    }
    long long total = 1;
    for (size_t i = 0; i < cols.size(); ++i) total *= 3;
    const Point q = make_point(kRow, qcol);
    for (long long code = 0; code < total; ++code) {
      Board b;
      long long x = code;
      for (int c : cols) {
        const int v = static_cast<int>(x % 3);
        x /= 3;
        if (v == 1) b.place(make_point(kRow, c), Color::BLACK);
        if (v == 2) b.place(make_point(kRow, c), Color::WHITE);
      }
      ++patterns;
      const ForbiddenInfo info = explain_black_move(b, q);
      const MoveVerdict ref = ref::black_verdict(b, q);
      if (info.verdict != ref || info.verdict != black_verdict(b, q) ||
          info.four_count() != ref::count_black_fours(b, q) ||
          info.apparent_three_count() != ref::count_black_apparent_threes(b, q)) {
        if (++mismatches <= 5) {
          ADD_FAILURE() << "mismatch at " << to_notation(q) << "\n" << b.to_string() << info.to_string()
                        << "oracle: " << verdict_name(ref);
        }
        continue;
      }

      // Evidence for A-R3 / A-R4 (evaluated without the A-R3 skip).
      if (info.verdict == MoveVerdict::WIN_FIVE || info.verdict == MoveVerdict::FORBIDDEN_OVERLINE) continue;
      Board withq = b;
      withq.place(q, Color::BLACK);
      std::set<std::vector<int>> three_sets;
      for (int k = -3; k <= 3; ++k) {
        const int ecol = qcol + k;
        if (k == 0 || !on_board(kRow, ecol) || !withq.empty_at(make_point(kRow, ecol))) continue;
        Board withe = withq;
        withe.place(make_point(kRow, ecol), Color::BLACK);
        if (makes_five(withq, make_point(kRow, ecol), Color::BLACK)) continue;
        auto run = straight_four_cols(withe, qcol, ecol);
        if (run.empty()) continue;
        run.erase(std::find(run.begin(), run.end(), ecol));
        three_sets.insert(run);
      }
      if (info.four_count() > 0 && !three_sets.empty()) {
        if (++ar3_four_and_three == 1) ar3_example = b.to_string() + "move " + to_notation(q);
      }
      if (three_sets.size() > 1) {
        if (++ar4_two_threes == 1) ar4_example = b.to_string() + "move " + to_notation(q);
      }
    }
  }
  EXPECT_EQ(mismatches, 0);
  EXPECT_GT(patterns, 300000);
  std::cout << "[line-enum] patterns=" << patterns
            << " A-R3(four+apparent three in one line)=" << ar3_four_and_three
            << " A-R4(two distinct threes in one line)=" << ar4_two_threes << "\n";
  if (!ar3_example.empty()) std::cout << "[line-enum] A-R3 example:\n" << ar3_example << "\n";
  if (!ar4_example.empty()) std::cout << "[line-enum] A-R4 example:\n" << ar4_example << "\n";
  RecordProperty("patterns", std::to_string(patterns));
  RecordProperty("ar3_count", std::to_string(ar3_four_and_three));
  RecordProperty("ar4_count", std::to_string(ar4_two_threes));
}

}  // namespace
}  // namespace dz
