#include <gtest/gtest.h>

#include <random>
#include <set>

#include "board/board.h"
#include "board/transform.h"
#include "common/test_util.h"

namespace dz {
namespace {

TEST(Coord, NotationRoundTripAllPoints) {
  for (int p = 0; p < kNumPoints; ++p) {
    const std::string s = to_notation(static_cast<Point>(p));
    auto back = parse_point(s);
    ASSERT_TRUE(back) << s;
    EXPECT_EQ(*back, p);
  }
}

TEST(Coord, Conventions) {
  EXPECT_EQ(kCenterPoint, 112);
  EXPECT_EQ(to_notation(kCenterPoint), "H8");
  EXPECT_EQ(*parse_point("A1"), 0);
  EXPECT_EQ(*parse_point("O15"), 224);
  EXPECT_EQ(*parse_point("I9"), make_point(8, 8));  // column I is not skipped
  EXPECT_EQ(*parse_point("h8"), kCenterPoint);
  EXPECT_EQ(row_of(*parse_point("C12")), 11);
  EXPECT_EQ(col_of(*parse_point("C12")), 2);
}

TEST(Coord, RejectsMalformed) {
  for (const char* s : {"", "H", "H0", "H16", "P8", "8H", "H08", "HH", "A100", "@1"}) {
    EXPECT_FALSE(parse_point(s)) << s;
  }
}

TEST(Coord, CentralRegions) {
  int counts[16] = {};
  for (int p = 0; p < kNumPoints; ++p) {
    for (int n = 1; n <= 15; n += 2) counts[n] += in_central(static_cast<Point>(p), n) ? 1 : 0;
  }
  for (int n = 1; n <= 15; n += 2) EXPECT_EQ(counts[n], n * n) << n;
  EXPECT_TRUE(in_central(*parse_point("G7"), 3));
  EXPECT_TRUE(in_central(*parse_point("I9"), 3));
  EXPECT_FALSE(in_central(*parse_point("J9"), 3));
  EXPECT_FALSE(in_central(*parse_point("F8"), 3));
  EXPECT_TRUE(in_central(*parse_point("D4"), 9));
  EXPECT_TRUE(in_central(*parse_point("L12"), 9));
  EXPECT_FALSE(in_central(*parse_point("C8"), 9));
}

TEST(Transform, MatchesRulesTable) {
  // RULES.md §1.5 on the point (dr, dc) = (1, 2) relative to H8, i.e. J9.
  const Point j9 = *parse_point("J9");
  const char* expected[kNumTransforms] = {"J9", "G10", "F7", "I6", "F9", "G6", "J7", "I10"};
  for (int t = 0; t < kNumTransforms; ++t) {
    EXPECT_EQ(to_notation(transform_point(t, j9)), expected[t]) << transform_name(t);
  }
}

TEST(Transform, GroupAxioms) {
  for (int t = 0; t < kNumTransforms; ++t) {
    EXPECT_EQ(transform_point(t, kCenterPoint), kCenterPoint);
    std::set<Point> image;
    for (int p = 0; p < kNumPoints; ++p) image.insert(transform_point(t, static_cast<Point>(p)));
    EXPECT_EQ(image.size(), static_cast<size_t>(kNumPoints)) << "bijection";
    EXPECT_EQ(compose_transforms(t, inverse_transform(t)), 0);
    for (int u = 0; u < kNumTransforms; ++u) {
      const int c = compose_transforms(t, u);
      ASSERT_GE(c, 0) << "closure";
      for (int p = 0; p < kNumPoints; ++p) {
        ASSERT_EQ(transform_point(c, static_cast<Point>(p)),
                  transform_point(t, transform_point(u, static_cast<Point>(p))));
      }
    }
  }
  // rot90^4 = id, mirror^2 = id, T(4+k) = rot^k after mirror.
  EXPECT_EQ(compose_transforms(1, compose_transforms(1, compose_transforms(1, 1))), 0);
  EXPECT_EQ(compose_transforms(4, 4), 0);
  for (int k = 0; k < 4; ++k) EXPECT_EQ(compose_transforms(k, 4), 4 + k);
}

TEST(Board, PlaceRemoveRestoresHashAndCounts) {
  std::mt19937_64 rng(11);
  for (int trial = 0; trial < 200; ++trial) {
    Board b;
    std::vector<Point> placed;
    for (int i = 0; i < 60; ++i) {
      const Point p = static_cast<Point>(rng() % kNumPoints);
      if (!b.empty_at(p)) continue;
      b.place(p, (rng() & 1) ? Color::BLACK : Color::WHITE);
      placed.push_back(p);
    }
    // Hash is independent of placement order.
    Board rebuilt;
    for (auto it = placed.rbegin(); it != placed.rend(); ++it) {
      rebuilt.place(*it, b.at(*it) == BLACK_STONE ? Color::BLACK : Color::WHITE);
    }
    for (int t = 0; t < kNumTransforms; ++t) EXPECT_EQ(b.hash(t), rebuilt.hash(t));
    // Transformed board's plain hash equals the original's hash under that transform.
    for (int t = 0; t < kNumTransforms; ++t) EXPECT_EQ(b.transformed(t).hash(0), b.hash(t));
    for (Point p : placed) b.remove(p);
    EXPECT_EQ(b, Board());
    for (int t = 0; t < kNumTransforms; ++t) EXPECT_EQ(b.hash(t), Board().hash(t));
    EXPECT_EQ(b.stone_count(), 0);
  }
}

TEST(Board, PaddingIsWall) {
  Board b;
  int walls = 0;
  for (uint8_t c : b.cells()) walls += c == WALL ? 1 : 0;
  EXPECT_EQ(walls, Board::kPaddedSize - kNumPoints);
  for (int p = 0; p < kNumPoints; ++p) {
    EXPECT_EQ(Board::point_of_index(Board::index_of(static_cast<Point>(p))), p);
  }
}

}  // namespace
}  // namespace dz
