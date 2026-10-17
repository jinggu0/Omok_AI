// Taraguchi-10 state machine (RULES.md §8, §9).
#include <gtest/gtest.h>

#include <functional>

#include "common/test_util.h"
#include "game/game_state.h"

namespace dz {
namespace {

using test::P;

ProfilePtr tara() { return make_profile(RuleProfile::taraguchi10()); }

size_t count_place(const GameState& g) {
  size_t n = 0;
  for (const Action& a : g.legal_actions()) n += a.type == ActionType::PLACE ? 1 : 0;
  return n;
}

TEST(Opening, DirectFifthPathPhasesActorsAndRegions) {
  GameState g(tara(), {PlayerId::P0, PlayerId::P1});
  EXPECT_EQ(g.phase(), OpeningPhase::BLACK_MOVE_1);
  EXPECT_EQ(g.actor(), PlayerId::P0);
  ASSERT_EQ(g.legal_actions(), std::vector<Action>{Action::place(kCenterPoint)});
  EXPECT_FALSE(g.apply(Action::place(P("G7"))));
  EXPECT_FALSE(g.apply(Action::pass()));
  ASSERT_TRUE(g.apply(Action::place(P("H8"))));

  EXPECT_EQ(g.phase(), OpeningPhase::SWAP_AFTER_1);
  EXPECT_EQ(g.actor_role(), Role::WHITE_HOLDER);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  EXPECT_EQ(g.legal_actions(), (std::vector<Action>{Action::swap(), Action::keep()}));
  EXPECT_FALSE(g.apply(Action::place(P("H9"))));
  ASSERT_TRUE(g.apply(Action::swap()));
  EXPECT_EQ(g.assignment().black_holder, PlayerId::P1);
  EXPECT_EQ(g.assignment().white_holder, PlayerId::P0);

  EXPECT_EQ(g.phase(), OpeningPhase::WHITE_MOVE_2);
  EXPECT_EQ(g.actor(), PlayerId::P0) << "after the swap P0 holds white and plays move 2";
  EXPECT_EQ(count_place(g), 8u);
  EXPECT_FALSE(g.apply(Action::place(P("F8"))));
  ASSERT_TRUE(g.apply(Action::place(P("I9"))));

  EXPECT_EQ(g.phase(), OpeningPhase::SWAP_AFTER_2);
  EXPECT_EQ(g.actor_role(), Role::BLACK_HOLDER);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  ASSERT_TRUE(g.apply(Action::keep()));

  EXPECT_EQ(g.phase(), OpeningPhase::BLACK_MOVE_3);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  EXPECT_EQ(count_place(g), 23u);
  EXPECT_FALSE(g.apply(Action::place(P("E8"))));
  ASSERT_TRUE(g.apply(Action::place(P("G10"))));

  EXPECT_EQ(g.phase(), OpeningPhase::SWAP_AFTER_3);
  EXPECT_EQ(g.actor(), PlayerId::P0);
  ASSERT_TRUE(g.apply(Action::keep()));

  EXPECT_EQ(g.phase(), OpeningPhase::WHITE_MOVE_4);
  EXPECT_EQ(count_place(g), 46u);
  EXPECT_FALSE(g.apply(Action::place(P("D8"))));
  ASSERT_TRUE(g.apply(Action::place(P("K5"))));

  EXPECT_EQ(g.phase(), OpeningPhase::CHOICE_AFTER_4);
  EXPECT_EQ(g.actor_role(), Role::BLACK_HOLDER);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  EXPECT_EQ(g.legal_actions(), (std::vector<Action>{Action::direct_fifth(), Action::ten_proposals()}));
  ASSERT_TRUE(g.apply(Action::direct_fifth()));

  EXPECT_EQ(g.phase(), OpeningPhase::SWAP_BEFORE_DIRECT_5);
  EXPECT_EQ(g.actor_role(), Role::BLACK_HOLDER);  // A-T2 default
  EXPECT_EQ(g.actor(), PlayerId::P1);
  ASSERT_TRUE(g.apply(Action::swap()));
  EXPECT_EQ(g.assignment().black_holder, PlayerId::P0);

  EXPECT_EQ(g.phase(), OpeningPhase::BLACK_DIRECT_MOVE_5);
  EXPECT_EQ(g.actor(), PlayerId::P0);
  EXPECT_EQ(count_place(g), 77u);
  EXPECT_FALSE(g.apply(Action::place(P("C8"))));
  ASSERT_TRUE(g.apply(Action::place(P("L12"))));

  EXPECT_EQ(g.phase(), OpeningPhase::SWAP_AFTER_DIRECT_5);
  EXPECT_EQ(g.actor_role(), Role::WHITE_HOLDER);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  ASSERT_TRUE(g.apply(Action::keep()));

  EXPECT_EQ(g.phase(), OpeningPhase::WHITE_MOVE_6);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  EXPECT_EQ(count_place(g), 220u);
  ASSERT_TRUE(g.apply(Action::place(P("A15"))));

  EXPECT_EQ(g.phase(), OpeningPhase::NORMAL_RENJU);
  EXPECT_EQ(g.side_to_move(), Color::BLACK);
  EXPECT_EQ(g.actor(), PlayerId::P0);
  EXPECT_EQ(g.board().stone_count(Color::BLACK), 3);
  EXPECT_EQ(g.board().stone_count(Color::WHITE), 3);
  EXPECT_TRUE(g.apply(Action::pass())) << "TARAGUCHI10 profile allows passing in NORMAL_RENJU";
}

TEST(Opening, TenProposalPath) {
  GameState g(tara());
  for (const char* m : {"H8"}) ASSERT_TRUE(g.apply(Action::place(P(m))));
  ASSERT_TRUE(g.apply(Action::keep()));
  ASSERT_TRUE(g.apply(Action::place(P("I9"))));
  ASSERT_TRUE(g.apply(Action::keep()));
  ASSERT_TRUE(g.apply(Action::place(P("G10"))));
  ASSERT_TRUE(g.apply(Action::keep()));
  ASSERT_TRUE(g.apply(Action::place(P("K5"))));
  ASSERT_TRUE(g.apply(Action::ten_proposals()));
  EXPECT_EQ(g.phase(), OpeningPhase::OFFER_TEN_5TH_MOVES);
  EXPECT_EQ(g.actor(), PlayerId::P0);
  EXPECT_FALSE(g.apply(Action::swap())) << "colors are fixed once ten proposals are chosen";
  EXPECT_FALSE(g.apply(Action::place(P("A1")))) << "proposals are PROPOSE_FIFTH actions";
  const char* props[] = {"A1", "O15", "C3", "M13", "E5", "J12", "B14", "N2", "F11", "L4"};
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(g.apply(Action::propose(P("H8")))) << "occupied";
    ASSERT_TRUE(g.apply(Action::propose(P(props[i])))) << props[i];
    if (i < 9) {
      EXPECT_EQ(g.phase(), OpeningPhase::OFFER_TEN_5TH_MOVES);
      EXPECT_FALSE(g.apply(Action::propose(P(props[i])))) << "duplicate";
    }
  }
  EXPECT_EQ(g.phase(), OpeningPhase::SELECT_ONE_OF_TEN);
  EXPECT_EQ(g.actor_role(), Role::WHITE_HOLDER);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  EXPECT_EQ(g.legal_actions().size(), 10u);
  EXPECT_FALSE(g.apply(Action::select(P("B1"))));
  EXPECT_EQ(g.board().stone_count(), 4) << "proposals are not stones";
  ASSERT_TRUE(g.apply(Action::select(P("E5"))));
  EXPECT_EQ(g.chosen_fifth(), P("E5"));
  EXPECT_EQ(g.board().at(P("E5")), BLACK_STONE);
  EXPECT_EQ(g.board().stone_count(), 5);
  EXPECT_EQ(g.phase(), OpeningPhase::WHITE_MOVE_6);
  EXPECT_EQ(g.actor(), PlayerId::P1);
  EXPECT_EQ(g.last_move(), P("E5"));
  EXPECT_FALSE(g.apply(Action::swap()));
  ASSERT_TRUE(g.apply(Action::place(P("O1"))));
  EXPECT_EQ(g.phase(), OpeningPhase::NORMAL_RENJU);

  // Undo back to the very beginning restores every field.
  const GameState fresh(tara());
  while (g.can_undo()) g.undo();
  EXPECT_EQ(g.hash(), fresh.hash());
  EXPECT_EQ(g.phase(), OpeningPhase::BLACK_MOVE_1);
  EXPECT_EQ(g.offered().size(), 0u);
  EXPECT_EQ(g.chosen_fifth(), kNoPoint);
  EXPECT_EQ(g.board().stone_count(), 0);
}

TEST(Opening, SwapBeforeDirectFifthActorIsConfigurable) {
  RuleProfile p = RuleProfile::taraguchi10();
  p.taraguchi.swap_before_direct5_actor = Role::WHITE_HOLDER;
  p.taraguchi.choice_actor = Role::WHITE_HOLDER;
  GameState g(make_profile(p));
  for (const Action& a : {Action::place(P("H8")), Action::keep(), Action::place(P("I9")), Action::keep(),
                          Action::place(P("G10")), Action::keep(), Action::place(P("K5"))}) {
    ASSERT_TRUE(g.apply(a));
  }
  EXPECT_EQ(g.actor_role(), Role::WHITE_HOLDER);
  ASSERT_TRUE(g.apply(Action::direct_fifth()));
  EXPECT_EQ(g.actor_role(), Role::WHITE_HOLDER);
  EXPECT_EQ(g.actor(), PlayerId::P1);
}

// Enumerates every combination of opening decisions (fixed stones) and checks the actor of each
// phase against RULES.md §8.3 and §9.3, and that the final colors follow from the swaps.
TEST(Opening, AllDecisionPathsFollowTheTable) {
  int paths = 0;
  std::function<void(GameState&)> walk = [&](GameState& g) {
    if (g.phase() == OpeningPhase::NORMAL_RENJU) {
      ++paths;
      // The number of swaps decides who holds black.
      int swaps = 0;
      for (const Action& a : g.action_history()) swaps += a.type == ActionType::SWAP ? 1 : 0;
      EXPECT_EQ(g.assignment().black_holder, swaps % 2 == 0 ? PlayerId::P0 : PlayerId::P1);
      return;
    }
    const OpeningPhase ph = g.phase();
    Role expected_role;
    switch (ph) {
      case OpeningPhase::SWAP_AFTER_1: case OpeningPhase::SWAP_AFTER_3: case OpeningPhase::SWAP_AFTER_DIRECT_5:
      case OpeningPhase::WHITE_MOVE_2: case OpeningPhase::WHITE_MOVE_4: case OpeningPhase::WHITE_MOVE_6:
      case OpeningPhase::SELECT_ONE_OF_TEN:
        expected_role = Role::WHITE_HOLDER; break;
      default:
        expected_role = Role::BLACK_HOLDER;
    }
    EXPECT_EQ(g.actor_role(), expected_role) << phase_name(ph);
    std::vector<Action> choices;
    switch (ph) {
      case OpeningPhase::BLACK_MOVE_1: choices = {Action::place(P("H8"))}; break;
      case OpeningPhase::WHITE_MOVE_2: choices = {Action::place(P("H9"))}; break;
      case OpeningPhase::BLACK_MOVE_3: choices = {Action::place(P("J10"))}; break;
      case OpeningPhase::WHITE_MOVE_4: choices = {Action::place(P("E6"))}; break;
      case OpeningPhase::BLACK_DIRECT_MOVE_5: choices = {Action::place(P("D4"))}; break;
      case OpeningPhase::WHITE_MOVE_6: choices = {Action::place(P("O1"))}; break;
      case OpeningPhase::OFFER_TEN_5TH_MOVES: {
        auto legal = g.legal_actions();
        choices = {legal.front()};
        break;
      }
      case OpeningPhase::SELECT_ONE_OF_TEN: choices = {g.legal_actions().back()}; break;
      default: choices = g.legal_actions();
    }
    for (const Action& a : choices) {
      const uint64_t before = g.hash();
      const SideAssignment sa = g.assignment();
      ASSERT_TRUE(g.apply(a)) << phase_name(ph) << " " << to_string(a);
      walk(g);
      g.undo();
      ASSERT_EQ(g.hash(), before);
      ASSERT_EQ(g.assignment(), sa);
      ASSERT_EQ(g.phase(), ph);
    }
  };
  GameState g(tara());
  walk(g);
  // 2^3 swap choices x (direct: 2 x 2 swaps + ten: 1) = 8 x 5 = 40 paths.
  EXPECT_EQ(paths, 40);
}

TEST(Opening, WinnerIsReportedAsPlayer) {
  // After an odd number of swaps P1 holds black; a black five must report P1 as winner.
  GameState g(tara());
  for (const Action& a : {Action::place(P("H8")), Action::swap(), Action::place(P("H9")), Action::keep(),
                          Action::place(P("I8")), Action::keep(), Action::place(P("H10")),
                          Action::direct_fifth(), Action::keep(), Action::place(P("J8")), Action::keep(),
                          Action::place(P("H11")), Action::place(P("K8")), Action::place(P("H12"))}) {
    ASSERT_TRUE(g.apply(a)) << to_string(a) << " in " << phase_name(g.phase());
  }
  ASSERT_EQ(g.assignment().black_holder, PlayerId::P1);
  ASSERT_TRUE(g.apply(Action::place(P("G8"))));  // G8 H8 I8 J8 K8
  EXPECT_EQ(g.outcome().result, GameResult::BLACK_WIN);
  EXPECT_EQ(*g.winner(), PlayerId::P1);
}

}  // namespace
}  // namespace dz
