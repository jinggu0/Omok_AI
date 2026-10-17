// Ten-proposal symmetry (RULES.md §10).
#include <gtest/gtest.h>

#include "common/test_util.h"
#include "opening/symmetry.h"

namespace dz {
namespace {

using test::P;

TEST(Symmetry, Stabilizers) {
  EXPECT_EQ(stabilizer_mask(Board()), 0xFF);
  EXPECT_EQ(stabilizer_mask(test::make_board({"H8"})), 0xFF);
  // H9 = (dr, dc) = (1, 0): fixed by identity and the left-right mirror only.
  EXPECT_EQ(stabilizer_mask(test::make_board({"H8"}, {"H9"})), 0x11);
  EXPECT_EQ(stabilizer_mask(test::make_board({"H8", "G10"}, {"I9", "K5"})), 0x01);
}

TEST(Symmetry, MirrorSymmetricPosition) {
  const Board b = test::make_board({"H8", "H10"}, {"H9", "H6"});  // all on column H
  for (SymmetryMode m : {SymmetryMode::POSITION, SymmetryMode::POINTWISE_D4}) {
    EXPECT_TRUE(proposals_equivalent(b, P("G8"), P("I8"), m));
    EXPECT_TRUE(proposals_equivalent(b, P("A1"), P("O1"), m));
  }
  EXPECT_FALSE(proposals_equivalent(b, P("G8"), P("J7"), SymmetryMode::POSITION));
  EXPECT_FALSE(proposals_equivalent(b, P("G8"), P("G7"), SymmetryMode::POSITION));
  EXPECT_EQ(canonical_proposal(b, P("I8"), SymmetryMode::POSITION),
            canonical_proposal(b, P("G8"), SymmetryMode::POSITION));
}

TEST(Symmetry, PositionVersusPointwise) {
  // Asymmetric position: A1 and O15 are rot180 images of each other as points (E3) but the
  // resulting positions differ (E1).
  const Board b = test::make_board({"H8", "G10"}, {"I9", "K5"});
  EXPECT_FALSE(proposals_equivalent(b, P("A1"), P("O15"), SymmetryMode::POSITION));
  EXPECT_TRUE(proposals_equivalent(b, P("A1"), P("O15"), SymmetryMode::POINTWISE_D4));
}

TEST(Symmetry, AdmissibilityRules) {
  const Board b = test::make_board({"H8", "H10"}, {"H9", "H6"});
  Taraguchi10Params params;
  const BlackRuleOptions opt;
  std::string why;
  const std::vector<Point> offered = {P("G8"), P("A1")};
  EXPECT_FALSE(proposal_admissible(b, offered, P("I8"), params, opt, &why));
  EXPECT_NE(why.find("symmetric"), std::string::npos);
  EXPECT_FALSE(proposal_admissible(b, offered, P("H9"), params, opt, &why));
  EXPECT_FALSE(proposal_admissible(b, offered, P("G8"), params, opt, &why));
  EXPECT_TRUE(proposal_admissible(b, offered, P("J7"), params, opt, &why)) << why;

  params.proposal_region = 9;
  EXPECT_FALSE(proposal_admissible(b, {}, P("A1"), params, opt, &why));
  EXPECT_TRUE(proposal_admissible(b, {}, P("D4"), params, opt, &why)) << why;

  // admissible_proposals never offers a point equivalent to an existing proposal.
  params = Taraguchi10Params{};
  const auto next = admissible_proposals(b, offered, params, opt);
  for (Point q : next) {
    for (Point o : offered) EXPECT_FALSE(proposals_equivalent(b, q, o, SymmetryMode::POSITION));
  }
  // 221 empty points; G8/I8 and A1/O1 are excluded once G8 and A1 are offered.
  EXPECT_EQ(admissible_proposals(b, {}, params, opt).size(), 221u);
  EXPECT_EQ(next.size(), 217u);
}

TEST(Symmetry, ValidateFullSet) {
  const Board b = test::make_board({"H8", "G10"}, {"I9", "K5"});
  Taraguchi10Params params;
  std::vector<Point> ten;
  for (const char* s : {"A1", "O15", "C3", "M13", "E5", "J12", "B14", "N2", "F11", "L4"}) ten.push_back(P(s));
  std::string why;
  EXPECT_TRUE(validate_proposals(b, ten, params, {}, &why)) << why;
  ten.pop_back();
  EXPECT_FALSE(validate_proposals(b, ten, params, {}, &why));
  ten.push_back(P("A1"));
  EXPECT_FALSE(validate_proposals(b, ten, params, {}, &why));
  params.proposal_symmetry = SymmetryMode::POINTWISE_D4;
  ten.back() = P("L4");
  EXPECT_FALSE(validate_proposals(b, ten, params, {}, &why)) << "A1/O15 are pointwise symmetric";
}

// RULES.md §10.2 theorem: for every reachable Taraguchi-10 4-move position, E1 (position
// equivalence) equals E2 (stabilizer orbit). Exhaustive in Release; every 7th position in Debug.
TEST(Symmetry, PositionEquivalenceEqualsStabilizerOrbitForAllFourMovePositions) {
#ifdef NDEBUG
  constexpr int kStride = 1;
#else
  constexpr int kStride = 7;
#endif
  long long positions = 0, checked = 0;
  for (int w2 = 0; w2 < kNumPoints; ++w2) {
    if (w2 == kCenterPoint || !in_central(static_cast<Point>(w2), 3)) continue;
    for (int b3 = 0; b3 < kNumPoints; ++b3) {
      if (b3 == kCenterPoint || b3 == w2 || !in_central(static_cast<Point>(b3), 5)) continue;
      for (int w4 = 0; w4 < kNumPoints; ++w4) {
        if (w4 == kCenterPoint || w4 == w2 || w4 == b3 || !in_central(static_cast<Point>(w4), 7)) continue;
        if (positions++ % kStride != 0) continue;
        Board b;
        b.place(kCenterPoint, Color::BLACK);
        b.place(static_cast<Point>(b3), Color::BLACK);
        b.place(static_cast<Point>(w2), Color::WHITE);
        b.place(static_cast<Point>(w4), Color::WHITE);
        const uint8_t stab = stabilizer_mask(b);
        for (int q = 0; q < kNumPoints; ++q) {
          if (!b.empty_at(static_cast<Point>(q))) continue;
          Point e2 = static_cast<Point>(q);
          for (int t = 0; t < kNumTransforms; ++t) {
            if (stab & (1u << t)) e2 = std::min(e2, transform_point(t, static_cast<Point>(q)));
          }
          const Point e1 = canonical_proposal(b, static_cast<Point>(q), SymmetryMode::POSITION);
          ASSERT_EQ(e1, e2) << "q=" << to_notation(static_cast<Point>(q)) << "\n" << b.to_string();
          ++checked;
        }
      }
    }
  }
  EXPECT_EQ(positions, 8LL * 23 * 46);
  EXPECT_GT(checked, 0);
}

}  // namespace
}  // namespace dz
