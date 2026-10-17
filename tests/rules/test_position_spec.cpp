// Malformed state rejection (RULES.md §8.3) and position input parsing.
#include <gtest/gtest.h>

#include "common/test_util.h"
#include "game/position_spec.h"

namespace dz {
namespace {

using nlohmann::json;

std::optional<GameState> load_json(const json& j, ProfilePtr profile, std::string* error) {
  auto spec = parse_position_spec(j, error);
  if (!spec) return std::nullopt;
  return load_position(*spec, std::move(profile), error);
}

ProfilePtr tara() { return make_profile(RuleProfile::taraguchi10()); }
ProfilePtr omok() { return make_profile(RuleProfile::omokdojang_dan()); }
ProfilePtr rif() { return make_profile(RuleProfile::renju_rif()); }

void expect_rejected(const json& j, ProfilePtr profile, const std::string& needle) {
  std::string error;
  auto g = load_json(j, std::move(profile), &error);
  EXPECT_FALSE(g) << j.dump();
  EXPECT_NE(error.find(needle), std::string::npos) << "error was: " << error;
}

TEST(PositionSpec, RejectsMalformed) {
  expect_rejected(json{{"black", {"H8"}}, {"white", {"H8"}}}, rif(), "duplicate");
  expect_rejected(json{{"black", {"H8", "H8"}}}, rif(), "duplicate");
  expect_rejected(json{{"black", {"H8"}}, {"phase", "SWAP_AFTER_1"}}, rif(), "TARAGUCHI_10");
  expect_rejected(json{{"black", {"H8", "H9"}}, {"phase", "SWAP_AFTER_1"}}, tara(), "requires");
  expect_rejected(json{{"black", {"G7"}}, {"phase", "WHITE_MOVE_2"}}, tara(), "regions");
  expect_rejected(json{{"black", {"H8"}}, {"white", {"F8"}}, {"phase", "BLACK_MOVE_3"}}, tara(), "regions");
  expect_rejected(json{{"black", {"H8", "E8"}}, {"white", {"H9"}}, {"phase", "WHITE_MOVE_4"}}, tara(), "regions");
  expect_rejected(json{{"black", {"H8", "H10"}}, {"white", {"H9", "D8"}}, {"phase", "CHOICE_AFTER_4"}}, tara(),
                  "regions");
  expect_rejected(json{{"black", {"H8", "H10", "C8"}}, {"white", {"H9", "H6"}}, {"phase", "SWAP_AFTER_DIRECT_5"}},
                  tara(), "regions");
  expect_rejected(json{{"black", {"D8", "E8", "F8", "G8", "H8"}}, {"white", {"A1", "C1", "E1", "G1", "I1"}}}, rif(),
                  "five");
  expect_rejected(json{{"black", {"A1"}}, {"white", {"H8", "H9", "H10", "H11", "H12"}}, {"side", "BLACK"}}, rif(),
                  "five");
  expect_rejected(json{{"black", {"H8"}}, {"assignment", {{"black", "P0"}, {"white", "P0"}}}}, rif(), "assignment");
  expect_rejected(json{{"black", {"H8", "A1"}}, {"white", {"H9"}}, {"side", "BLACK"}}, omok(), "inconsistent");
  expect_rejected(json{{"black", {"H8", "A1"}}, {"white", json::array()}}, rif(), "infer");
  expect_rejected(json{{"black", {"H8"}}, {"white", {"H9"}}, {"offered", {"A1"}}}, rif(), "offered");
  expect_rejected(json{{"moves", {"H8", "A1"}}}, tara(), "illegal");
  expect_rejected(json{{"moves", {"H8", "keep", "F8"}}}, tara(), "illegal");
  expect_rejected(json{{"moves", {"H8"}}, {"black", {"A1"}}}, tara(), "combined");
  expect_rejected(json{{"black", {"H8"}}, {"white", json::array()}, {"last_move", "A1"}}, rif(), "last_move");
}

TEST(PositionSpec, RejectsBadOfferedSets) {
  // Column-H position is mirror symmetric: G8 and J8 are equivalent proposals.
  const json base = {{"black", {"H8", "H10"}}, {"white", {"H9", "H6"}}, {"phase", "OFFER_TEN_5TH_MOVES"}};
  json sym = base;
  sym["offered"] = {"G8", "I8"};
  expect_rejected(sym, tara(), "symmetric");
  json occupied = base;
  occupied["offered"] = {"H9"};
  expect_rejected(occupied, tara(), "occupied");
  json too_few = base;
  too_few["phase"] = "SELECT_ONE_OF_TEN";
  too_few["offered"] = {"G8"};
  expect_rejected(too_few, tara(), "size");
  json ok = base;
  ok["offered"] = {"G8", "G7", "G6"};
  std::string error;
  auto g = load_json(ok, tara(), &error);
  ASSERT_TRUE(g) << error;
  EXPECT_EQ(g->offered().size(), 3u);
  EXPECT_EQ(g->phase(), OpeningPhase::OFFER_TEN_5TH_MOVES);
}

TEST(PositionSpec, ParsesLinesAndDiagram) {
  std::string error;
  auto spec = parse_position_spec(json{{"lines", {{{"from", "D8"}, {"dir", "H"}, {"cells", "X.O"}}}}}, &error);
  ASSERT_TRUE(spec) << error;
  EXPECT_EQ(spec->black, std::vector<Point>{*parse_point("D8")});
  EXPECT_EQ(spec->white, std::vector<Point>{*parse_point("F8")});

  std::vector<std::string> rows(15, "...............");
  rows[7] = ".......X.......";   // rank 8 -> H8
  rows[14] = "O..............";  // rank 1 -> A1
  spec = parse_position_spec(json{{"diagram", rows}}, &error);
  ASSERT_TRUE(spec) << error;
  EXPECT_EQ(spec->black, std::vector<Point>{kCenterPoint});
  EXPECT_EQ(spec->white, std::vector<Point>{*parse_point("A1")});

  EXPECT_FALSE(parse_position_spec(json{{"lines", {{{"from", "N8"}, {"dir", "H"}, {"cells", "XXX"}}}}}, &error));
  EXPECT_FALSE(parse_position_spec(json{{"blak", {"H8"}}}, &error));
  EXPECT_NE(error.find("unknown key"), std::string::npos);
}

TEST(PositionSpec, AcceptsValidOpeningSetups) {
  std::string error;
  auto g = load_json(json{{"black", {"H8", "G10"}}, {"white", {"I9", "K5"}}, {"phase", "CHOICE_AFTER_4"},
                          {"assignment", {{"black", "P1"}, {"white", "P0"}}}},
                     tara(), &error);
  ASSERT_TRUE(g) << error;
  EXPECT_EQ(g->actor(), PlayerId::P1);  // black holder decides (A-T1)
  auto g6 = load_json(json{{"black", {"H8", "G10", "A1"}}, {"white", {"I9", "K5"}}, {"phase", "WHITE_MOVE_6"}},
                      tara(), &error);
  ASSERT_TRUE(g6) << error << " (selected fifth may be anywhere)";
  auto moves = load_json(json{{"moves", {"H8", "swap", "I9", "keep", "G10", "keep", "K5", "offer10", "A1",
                                          "B1", "C1", "D1", "E1", "F1", "G1", "B2", "C2", "D2", "select:D2",
                                          "H9"}}},
                         tara(), &error);
  ASSERT_TRUE(moves) << error;
  EXPECT_EQ(moves->phase(), OpeningPhase::NORMAL_RENJU);
  EXPECT_EQ(moves->chosen_fifth(), *parse_point("D2"));
}

}  // namespace
}  // namespace dz
