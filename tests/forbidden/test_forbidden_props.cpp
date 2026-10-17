// Property tests: fast checker vs reference oracle (differential), explain consistency,
// D4 invariance. Short versions of what tools/fuzz_rules.cpp runs at scale.
#include <gtest/gtest.h>

#include <random>

#include "common/test_util.h"
#include "rules/forbidden.h"
#include "rules/reference_rules.h"
#include "tools/random_positions.h"

namespace dz {
namespace {

#ifdef NDEBUG
constexpr int kBoards = 3000;
#else
constexpr int kBoards = 300;
#endif

using tools::near_black;

void differential(uint64_t seed, const BlackRuleOptions& opt, int boards) {
  std::mt19937_64 rng(seed);
  int compared = 0;
  for (int i = 0; i < boards; ++i) {
    const Board b = test::random_dense_board(rng);
    for (int p = 0; p < kNumPoints; ++p) {
      const Point q = static_cast<Point>(p);
      if (!b.empty_at(q) || !near_black(b, q)) continue;
      const MoveVerdict fast = black_verdict(b, q, opt);
      const MoveVerdict ref = ref::black_verdict(b, q, opt);
      ++compared;
      ASSERT_EQ(fast, ref) << "seed " << seed << " board " << i << " point " << to_notation(q) << "\n"
                           << b.to_string() << explain_black_move(b, q, opt).to_string();
    }
  }
  EXPECT_GT(compared, boards * 10);
}

TEST(ForbiddenProps, FastMatchesOracleDefaultOptions) { differential(1, {}, kBoards); }
TEST(ForbiddenProps, FastMatchesOracleNoFivePrecedence) {
  differential(2, {/*exact_five_precedence=*/false, true}, kBoards / 3);
}
TEST(ForbiddenProps, FastMatchesOracleOverlineBeatsFive) {
  differential(3, {true, /*five_beats_overline_other_dir=*/false}, kBoards / 3);
}

TEST(ForbiddenProps, ExplainMatchesFastAndOracleCounts) {
  std::mt19937_64 rng(4);
  for (int i = 0; i < kBoards / 3; ++i) {
    const Board b = test::random_dense_board(rng);
    for (int p = 0; p < kNumPoints; ++p) {
      const Point q = static_cast<Point>(p);
      if (!b.empty_at(q) || !near_black(b, q)) continue;
      const ForbiddenInfo info = explain_black_move(b, q);
      ASSERT_EQ(info.verdict, black_verdict(b, q)) << b.to_string() << info.to_string();
      ASSERT_EQ(info.four_count(), ref::count_black_fours(b, q)) << b.to_string() << info.to_string();
      ASSERT_EQ(info.apparent_three_count(), ref::count_black_apparent_threes(b, q))
          << b.to_string() << info.to_string();
      ASSERT_EQ(info.real_three_count(), ref::count_black_real_threes(b, q)) << b.to_string() << info.to_string();
      ASSERT_EQ(count_fours(b, q, Color::BLACK), info.four_count());
    }
  }
}

TEST(ForbiddenProps, VerdictInvariantUnderD4) {
  std::mt19937_64 rng(5);
  for (int i = 0; i < kBoards / 3; ++i) {
    const Board b = test::random_dense_board(rng);
    std::array<Board, kNumTransforms> tb;
    for (int t = 0; t < kNumTransforms; ++t) tb[t] = b.transformed(t);
    for (int p = 0; p < kNumPoints; ++p) {
      const Point q = static_cast<Point>(p);
      if (!b.empty_at(q) || !near_black(b, q)) continue;
      const MoveVerdict v = black_verdict(b, q);
      for (int t = 1; t < kNumTransforms; ++t) {
        ASSERT_EQ(black_verdict(tb[t], transform_point(t, q)), v) << transform_name(t) << "\n" << b.to_string();
      }
    }
  }
}

TEST(ForbiddenProps, WhitePatternQueries) {
  // White: overline counts as five; no forbidden moves.
  Board b = test::make_board({}, {"D8", "E8", "F8", "G8", "I8"});
  EXPECT_TRUE(makes_five(b, test::P("H8"), Color::WHITE));
  EXPECT_TRUE(makes_overline(b, test::P("H8"), Color::WHITE));
  Board bb = test::make_board({"D8", "E8", "F8", "G8", "I8"});
  EXPECT_FALSE(makes_five(bb, test::P("H8"), Color::BLACK));
  EXPECT_TRUE(makes_overline(bb, test::P("H8"), Color::BLACK));
  // White double four and double three are just patterns.
  Board w44 = test::make_board({}, {"F8", "G8", "H8", "I5", "I6", "I7"});
  EXPECT_EQ(count_fours(w44, test::P("I8"), Color::WHITE), 2);
  Board w33 = test::make_board({}, {"F8", "G8", "H6", "H7"});
  EXPECT_EQ(count_threes(w33, test::P("H8"), Color::WHITE), 2);
}

}  // namespace
}  // namespace dz
