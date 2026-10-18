#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>

#include "common/test_util.h"
#include "tactics/threats.h"
#include "tools/random_positions.h"

namespace dz::tactics {
namespace {

using test::make_board;
using test::P;

std::set<std::string> names(const std::vector<Point>& pts) {
  std::set<std::string> s;
  for (Point p : pts) s.insert(to_notation(p));
  return s;
}

TEST(Threats, FivePoints) {
  Rules r;
  std::vector<Point> out;
  Board b = make_board({"F8", "G8", "H8", "I8"});
  five_points(b, Color::BLACK, r, out);
  EXPECT_EQ(names(out), (std::set<std::string>{"E8", "J8"}));
  five_points(b, Color::WHITE, r, out);
  EXPECT_TRUE(out.empty());

  // Black: H8 would be an overline, C8 is an exact five. White: both win.
  Board over = make_board({"D8", "E8", "F8", "G8", "I8"});
  five_points(over, Color::BLACK, r, out);
  EXPECT_EQ(names(out), (std::set<std::string>{"C8"}));
  Board wover = make_board({}, {"D8", "E8", "F8", "G8", "I8"});
  five_points(wover, Color::WHITE, r, out);
  EXPECT_EQ(names(out), (std::set<std::string>{"C8", "H8"}));
}

TEST(Threats, FivePointsHonourProfileOptions) {
  // H8 = exact five horizontally + overline vertically.
  Board b = make_board({"D8", "E8", "F8", "G8", "H4", "H5", "H6", "H7", "H9"});
  Rules r;
  EXPECT_TRUE(wins_with(b, P("H8"), Color::BLACK, r));
  r.black.five_beats_overline_other_dir = false;
  EXPECT_FALSE(wins_with(b, P("H8"), Color::BLACK, r));
}

TEST(Threats, FourAndDoubleThreatMoves) {
  Rules r;
  std::vector<Point> out;
  Board b = make_board({"F8", "G8", "H8"});
  four_moves(b, Color::BLACK, r, out);
  EXPECT_EQ(names(out), (std::set<std::string>{"D8", "E8", "I8", "J8"}));
  EXPECT_TRUE(out[0] == P("E8") || out[0] == P("I8")) << "straight fours first";
  double_threat_moves(b, Color::BLACK, r, out);
  EXPECT_EQ(names(out), (std::set<std::string>{"E8", "I8"}));

  // Black 4-4 at I8 is forbidden: not a four move. White may play it.
  Board bb = make_board({"F8", "G8", "H8", "I5", "I6", "I7"}, {"E8", "I4"});
  four_moves(bb, Color::BLACK, r, out);
  EXPECT_EQ(std::count(out.begin(), out.end(), P("I8")), 0);
  Board ww = make_board({"E8", "I4"}, {"F8", "G8", "H8", "I5", "I6", "I7"});
  double_threat_moves(ww, Color::WHITE, r, out);
  EXPECT_EQ(names(out), (std::set<std::string>{"I8"}));
}

TEST(Threats, ThreatMovesIncludeThrees) {
  Rules r;
  std::vector<Point> out;
  Board b = make_board({"F8", "G8"});
  threat_moves(b, Color::BLACK, r, out);
  const auto s = names(out);
  for (const char* p : {"D8", "E8", "H8", "I8"}) EXPECT_TRUE(s.count(p)) << p;
  EXPECT_FALSE(s.count("C8"));
  // Black 3-3 point is not a threat move (illegal).
  Board f = make_board({"F8", "G8", "H6", "H7"});
  threat_moves(f, Color::BLACK, r, out);
  EXPECT_EQ(std::count(out.begin(), out.end(), P("H8")), 0);
}

TEST(Threats, FastBlackRealThreesMatchExplain) {
  std::mt19937_64 rng(31);
  for (int i = 0; i < 400; ++i) {
    const Board b = tools::random_dense_board(rng);
    for (int p = 0; p < kNumPoints; ++p) {
      const Point q = static_cast<Point>(p);
      if (!b.empty_at(q) || !tools::near_black(b, q)) continue;
      ASSERT_EQ(count_threes(b, q, Color::BLACK), explain_black_move(b, q).real_three_count())
          << b.to_string() << to_notation(q);
    }
  }
}

}  // namespace
}  // namespace dz::tactics
