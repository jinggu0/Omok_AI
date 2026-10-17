// GameState rules in NORMAL_RENJU: win/loss/draw, forbidden handling, pass, undo, hashing, D4.
#include <gtest/gtest.h>

#include <random>

#include "common/test_util.h"
#include "game/game_state.h"
#include "game/position_spec.h"

namespace dz {
namespace {

using test::P;

GameState setup(ProfilePtr profile, std::initializer_list<const char*> black,
                std::initializer_list<const char*> white, std::optional<Color> side = std::nullopt) {
  PositionSpec spec;
  for (const char* s : black) spec.black.push_back(P(s));
  for (const char* s : white) spec.white.push_back(P(s));
  spec.side_to_move = side;
  std::string error;
  auto g = load_position(spec, std::move(profile), &error);
  EXPECT_TRUE(g) << error;
  return g ? *g : GameState(make_profile(RuleProfile::renju_rif()));
}

ProfilePtr rif() { return make_profile(RuleProfile::renju_rif()); }
ProfilePtr rif_loss() {
  RuleProfile p = RuleProfile::renju_rif();
  p.illegal_move_behavior = IllegalMoveBehavior::LOSS;
  return make_profile(p);
}

// Scattered filler stones on rank 1 / rank 15 that form no pattern near the action.
const std::initializer_list<const char*> kFillB = {"A1", "C1", "E1", "G1", "I1", "K1"};
const std::initializer_list<const char*> kFillW = {"A15", "C15", "E15", "G15", "I15", "K15"};

TEST(GameRules, FreshNormalGame) {
  GameState g(rif());
  EXPECT_EQ(g.phase(), OpeningPhase::NORMAL_RENJU);
  EXPECT_EQ(g.side_to_move(), Color::BLACK);
  EXPECT_EQ(g.actor(), PlayerId::P0);
  EXPECT_EQ(g.legal_actions().size(), static_cast<size_t>(kNumPoints + 1));  // + pass
  GameState od(make_profile(RuleProfile::omokdojang_dan()));
  EXPECT_EQ(od.phase(), OpeningPhase::BLACK_MOVE_1);
}

TEST(GameRules, BlackExactFiveWins) {
  GameState g = setup(rif(), {"D8", "E8", "F8", "G8", "A1", "C1"}, {"D9", "E9", "F9", "G9", "A15", "C15"});
  ASSERT_EQ(g.side_to_move(), Color::BLACK);
  ASSERT_TRUE(g.apply(Action::place(P("H8"))));
  EXPECT_EQ(g.outcome().result, GameResult::BLACK_WIN);
  EXPECT_EQ(g.outcome().reason, EndReason::FIVE);
  EXPECT_EQ(*g.winner(), PlayerId::P0);
  EXPECT_FALSE(g.apply(Action::place(P("H9")))) << "terminal state accepts no action";
  EXPECT_TRUE(g.legal_actions().empty());
  g.undo();
  EXPECT_FALSE(g.is_terminal());
}

TEST(GameRules, WhiteOverlineWins) {
  GameState g = setup(rif(), {"A1", "C1", "E1", "G1", "I1", "K1"}, {"D8", "E8", "F8", "G8", "I8"});
  ASSERT_EQ(g.side_to_move(), Color::WHITE);
  ASSERT_TRUE(g.apply(Action::place(P("H8"))));
  EXPECT_EQ(g.outcome().result, GameResult::WHITE_WIN);
  EXPECT_EQ(*g.winner(), PlayerId::P1);
}

TEST(GameRules, BlackOverlineRejectedUnderReject) {
  GameState g = setup(rif(), {"D8", "E8", "F8", "G8", "I8"}, {"A15", "C15", "E15", "G15", "I15"});
  const uint64_t h = g.hash();
  EXPECT_FALSE(g.is_legal(Action::place(P("H8"))));
  EXPECT_FALSE(g.apply(Action::place(P("H8"))));
  EXPECT_EQ(g.hash(), h);
  EXPECT_FALSE(g.can_undo());
  for (const Action& a : g.legal_actions()) EXPECT_NE(a, Action::place(P("H8")));
}

TEST(GameRules, BlackForbiddenLosesUnderLoss) {
  GameState g = setup(rif_loss(), {"F8", "G8", "H6", "H7"}, {"A15", "C15", "E15", "G15"});
  EXPECT_TRUE(g.is_legal(Action::place(P("H8"))));
  for (const Action& a : g.legal_actions()) EXPECT_NE(a, Action::place(P("H8"))) << "engines never pick it";
  ASSERT_TRUE(g.apply(Action::place(P("H8"))));
  EXPECT_EQ(g.outcome().result, GameResult::WHITE_WIN);
  EXPECT_EQ(g.outcome().reason, EndReason::FORBIDDEN_MOVE);
  g.undo();
  EXPECT_FALSE(g.is_terminal());
  EXPECT_TRUE(g.board().empty_at(P("H8")));
}

TEST(GameRules, DoubleThreeRejectedFourThreeAllowed) {
  GameState g = setup(rif(), {"F8", "G8", "H6", "H7"}, {"A15", "C15", "E15", "G15"});
  EXPECT_FALSE(g.apply(Action::place(P("H8"))));
  EXPECT_EQ(g.forbidden(P("H8")).verdict, MoveVerdict::FORBIDDEN_DOUBLE_THREE);
  GameState g2 = setup(rif(), {"F8", "G8", "H8", "I6", "I7"}, {"A15", "C15", "E15", "G15", "I15"});
  EXPECT_TRUE(g2.apply(Action::place(P("I8"))));
  EXPECT_FALSE(g2.is_terminal());
}

TEST(GameRules, FiveWithDoubleFourWins) {
  GameState g = setup(rif(), {"D8", "E8", "F8", "G8", "H5", "H6", "H7", "E5", "F6", "G7"},
                      {"A15", "C15", "E15", "G15", "I15", "K15", "M15", "O15", "B13", "D13"});
  ASSERT_TRUE(g.apply(Action::place(P("H8"))));
  EXPECT_EQ(g.outcome().result, GameResult::BLACK_WIN);
}

TEST(GameRules, PassRules) {
  GameState g(rif());
  ASSERT_TRUE(g.apply(Action::pass()));
  EXPECT_EQ(g.side_to_move(), Color::WHITE);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  ASSERT_TRUE(g.apply(Action::pass()));
  EXPECT_EQ(g.outcome().result, GameResult::DRAW);
  EXPECT_EQ(g.outcome().reason, EndReason::DOUBLE_PASS);
  g.undo();
  ASSERT_TRUE(g.apply(Action::place(P("H8"))));
  EXPECT_EQ(g.consecutive_passes(), 0);

  RuleProfile nopass = RuleProfile::renju_rif();
  nopass.pass_allowed = false;
  GameState h(make_profile(nopass));
  EXPECT_FALSE(h.apply(Action::pass()));
  for (const Action& a : h.legal_actions()) EXPECT_NE(a.type, ActionType::PASS);
}

TEST(GameRules, BoardFullIsDraw) {
  // Tiling with runs of at most 2 in every direction: black iff (col + 2*row) % 4 < 2.
  PositionSpec spec;
  Point last = kNoPoint;
  for (int p = 0; p < kNumPoints; ++p) {
    const bool black = (col_of(static_cast<Point>(p)) + 2 * row_of(static_cast<Point>(p))) % 4 < 2;
    if (black && last == kNoPoint && p > 100) { last = static_cast<Point>(p); continue; }
    (black ? spec.black : spec.white).push_back(static_cast<Point>(p));
  }
  ASSERT_EQ(spec.black.size(), spec.white.size());
  std::string error;
  RuleProfile nopass = RuleProfile::renju_rif();
  nopass.pass_allowed = false;
  auto g = load_position(spec, make_profile(nopass), &error);
  ASSERT_TRUE(g) << error;
  ASSERT_FALSE(g->is_terminal());
  ASSERT_EQ(g->legal_actions().size(), 1u);
  ASSERT_TRUE(g->apply(Action::place(last)));
  EXPECT_EQ(g->outcome().result, GameResult::DRAW);
  EXPECT_EQ(g->outcome().reason, EndReason::BOARD_FULL);
}

// Random playouts: every apply is undone exactly; hash equals a from-scratch replay; D4 copies
// have transformed legal actions and identical outcomes.
void random_playouts(ProfilePtr profile, uint64_t seed, int games) {
  std::mt19937_64 rng(seed);
  for (int gi = 0; gi < games; ++gi) {
    GameState g(profile, (rng() & 1) ? SideAssignment{PlayerId::P0, PlayerId::P1}
                                     : SideAssignment{PlayerId::P1, PlayerId::P0});
    std::vector<uint64_t> hashes{g.hash()};
    std::vector<OpeningPhase> phases{g.phase()};
    std::vector<SideAssignment> assigns{g.assignment()};
    int steps = 0;
    while (!g.is_terminal() && steps < 120) {
      auto actions = g.legal_actions();
      ASSERT_FALSE(actions.empty());
      // Prefer moves near existing stones so games develop tactics.
      const Action a = actions[rng() % actions.size()];
      ASSERT_TRUE(g.is_legal(a));
      ASSERT_TRUE(g.apply(a)) << to_string(a);
      hashes.push_back(g.hash());
      phases.push_back(g.phase());
      assigns.push_back(g.assignment());
      ++steps;
      if (steps % 17 == 0) {
        GameState replay(profile, g.initial_setup().assignment);
        for (const Action& h : g.action_history()) ASSERT_TRUE(replay.apply(h));
        ASSERT_EQ(replay.hash(), g.hash());
        ASSERT_EQ(replay.board(), g.board());
        const int t = static_cast<int>(rng() % kNumTransforms);
        const GameState tg = g.transformed(t);
        ASSERT_EQ(tg.outcome().result, g.outcome().result);
        ASSERT_EQ(tg.phase(), g.phase());
        ASSERT_EQ(tg.canonical_hash(), g.canonical_hash());
        auto la = g.legal_actions();
        auto lt = tg.legal_actions();
        ASSERT_EQ(la.size(), lt.size());
        for (Action x : la) {
          if (x.has_point()) x.point = transform_point(t, x.point);
          ASSERT_NE(std::find(lt.begin(), lt.end(), x), lt.end()) << to_string(x);
        }
      }
    }
    for (int i = static_cast<int>(hashes.size()) - 1; i > 0; --i) {
      g.undo();
      ASSERT_EQ(g.hash(), hashes[i - 1]);
      ASSERT_EQ(g.phase(), phases[i - 1]);
      ASSERT_EQ(g.assignment(), assigns[i - 1]);
    }
    EXPECT_FALSE(g.can_undo());
    EXPECT_EQ(g.board().stone_count(), 0);
  }
}

TEST(GameRules, RandomPlayoutsRenju) { random_playouts(rif(), 21, 40); }
TEST(GameRules, RandomPlayoutsTaraguchi) {
  random_playouts(make_profile(RuleProfile::taraguchi10()), 22, 25);
}
TEST(GameRules, RandomPlayoutsOmokdojang) {
  random_playouts(make_profile(RuleProfile::omokdojang_dan()), 23, 25);
}

}  // namespace
}  // namespace dz
