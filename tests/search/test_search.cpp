// Search verification (ARCHITECTURE.md §3.10): tactics shortcuts, MCTS-solver, actor-aware
// backup across swaps and consecutive same-player opening decisions, Gumbel root, batched path
// equivalence and tree invariants.
#include <gtest/gtest.h>

#include <cmath>
#include <chrono>
#include <random>
#include <thread>

#include "common/test_util.h"
#include "game/position_spec.h"
#include "search/bots.h"
#include "search/search.h"
#include "tactics/engine.h"
#include "tools/random_positions.h"

namespace dz::search {
namespace {

using test::P;

GameState setup(std::initializer_list<const char*> black, std::initializer_list<const char*> white, Color side,
                const RuleProfile& profile = RuleProfile::renju_rif()) {
  PositionSpec spec;
  for (const char* s : black) spec.black.push_back(P(s));
  for (const char* s : white) spec.white.push_back(P(s));
  spec.side_to_move = side;
  std::string error;
  auto g = load_position(spec, make_profile(profile), &error);
  EXPECT_TRUE(g) << error;
  return *g;
}

// W/D/L logits that encode a value v for the actor (P(win) - P(loss) = v).
Evaluation value_eval(float v) {
  Evaluation e;
  const float pw = std::max(1e-6f, (1 + v) / 2), pl = std::max(1e-6f, (1 - v) / 2);
  e.wdl = {std::log(pw), std::log(1e-6f), std::log(pl)};
  return e;
}

std::string dump(const SearchResult& r) {
  std::string s = "action " + to_string(r.action) + " value " + std::to_string(r.value) + " sims " +
                  std::to_string(r.simulations) + " proven " + std::to_string(r.proven) + "\n";
  for (const auto& c : r.children) {
    s += "  " + to_string(c.action) + " n=" + std::to_string(c.visits) + " q=" + std::to_string(c.q) +
         " p=" + std::to_string(c.prior) + " proven=" + std::to_string(c.proven) + "\n";
    if (s.size() > 3000) break;
  }
  return s;
}

SearchConfig cfg(int sims, RootMode root = RootMode::PUCT) {
  SearchConfig c;
  c.simulations = sims;
  c.root = root;
  c.root_tactics_nodes = 0;
  c.max_inflight = 1;
  return c;
}

TEST(Search, ImmediateWinShortCircuit) {
  GameState g = setup({"F8", "G8", "H8", "I8"}, {"B15", "E15", "H15", "K15"}, Color::BLACK);
  UniformEvaluator u;
  const SearchResult r = Tree(g, cfg(100)).run(u);
  EXPECT_TRUE(r.action == Action::place(P("E8")) || r.action == Action::place(P("J8")));
  EXPECT_EQ(r.proven, 1);
  EXPECT_EQ(r.simulations, 0);
}

TEST(Search, ForcedBlockIsTheOnlyChild) {
  GameState g = setup({"E8", "A1", "A4", "A7"}, {"F8", "G8", "H8", "I8"}, Color::BLACK);
  UniformEvaluator u;
  const SearchResult r = Tree(g, cfg(100)).run(u);
  EXPECT_EQ(r.action, Action::place(P("J8")));
  ASSERT_EQ(r.children.size(), 1u);
}

TEST(Search, NoDefenseIsProvenLoss) {
  GameState g = setup({"A1", "A4", "A7", "A10"}, {"F8", "G8", "H8", "I8"}, Color::BLACK);
  UniformEvaluator u;
  const SearchResult r = Tree(g, cfg(100)).run(u);
  EXPECT_EQ(r.proven, -1);
  EXPECT_TRUE(r.action == Action::place(P("E8")) || r.action == Action::place(P("J8"))) << to_string(r.action);
}

TEST(Search, SolverProvesStraightFourWithoutRootTactics) {
  // Open three: E8/I8 makes a straight four. Uniform priors, no root tactics: the MCTS-solver
  // must prove the win through the leaf checks (white facing two five points = proven loss).
  GameState g = setup({"F8", "G8", "H8"}, {"B15", "E15", "H15"}, Color::BLACK);
  UniformEvaluator u;
  SearchConfig c = cfg(600);
  c.fpu_reduction = 0.0f;  // uniform priors carry no information: explore breadth-first
  Tree t(g, c);
  const SearchResult r = t.run(u);
  EXPECT_EQ(r.proven, 1);
  EXPECT_EQ(r.proof_source, "tree");
  EXPECT_TRUE(r.action == Action::place(P("E8")) || r.action == Action::place(P("I8"))) << dump(r);
  std::string why;
  EXPECT_TRUE(t.check_invariants(&why)) << why;
}

// Actor-aware backup. The evaluators below are consistent with the game tree: the winner is fixed
// by the first decision taken below the root and never changes afterwards, so the exact value of
// each root action is known. Sign errors in backup (e.g. negamax flipping when the same player
// acts twice, or confusing colour with player) make these tests fail.
Evaluation fixed_winner(const GameState& s, std::optional<PlayerId> winner, float v) {
  if (!winner) return value_eval(0.0f);
  return value_eval(s.actor() == *winner ? v : -v);
}

TEST(Search, SwapDecisionUsesPlayerNotColor) {
  // Root: SWAP_AFTER_1, the white holder P1 decides. Winner = P1 if it swaps, else P0.
  GameState g(make_profile(RuleProfile::taraguchi10()));
  ASSERT_TRUE(g.apply(Action::place(kCenterPoint)));
  ASSERT_EQ(g.actor(), PlayerId::P1);
  auto winner_if = [](bool swap_wins_for_p1) {
    return [swap_wins_for_p1](const GameState& s) {
      const auto h = s.action_history();
      if (h.size() < 2) return fixed_winner(s, std::nullopt, 0.9f);
      const bool swapped = h[1].type == ActionType::SWAP;
      const PlayerId w = (swapped == swap_wins_for_p1) ? PlayerId::P1 : PlayerId::P0;
      return fixed_winner(s, w, 0.9f);
    };
  };
  for (RootMode mode : {RootMode::PUCT, RootMode::GUMBEL}) {
    LambdaEvaluator swap_good(winner_if(true));
    SearchResult r = Tree(g, cfg(64, mode)).run(swap_good);
    EXPECT_EQ(r.action, Action::swap()) << "mode " << static_cast<int>(mode) << "\n" << dump(r);
    EXPECT_GT(r.value, 0.5f) << dump(r);
    LambdaEvaluator keep_good(winner_if(false));
    r = Tree(g, cfg(64, mode)).run(keep_good);
    EXPECT_EQ(r.action, Action::keep()) << "mode " << static_cast<int>(mode) << "\n" << dump(r);
    EXPECT_GT(r.value, 0.5f) << dump(r);
  }
}

// Consecutive decisions by the same player: at CHOICE_AFTER_4 the black holder (P0) chooses and
// both branches continue with P0 acting again (SWAP_BEFORE_DIRECT_5 / OFFER_TEN_5TH_MOVES).
// Ten-proposal line: P0 wins with value 0.9; direct line: P0 wins with value 0.6. Negamax-style
// sign flipping between same-player nodes would prefer the direct line.
TEST(Search, ConsecutiveSamePlayerDecisions) {
  GameState g(make_profile(RuleProfile::taraguchi10()));
  for (const Action& a : {Action::place(P("H8")), Action::keep(), Action::place(P("I9")), Action::keep(),
                          Action::place(P("G10")), Action::keep(), Action::place(P("K5"))}) {
    ASSERT_TRUE(g.apply(a));
  }
  ASSERT_EQ(g.phase(), OpeningPhase::CHOICE_AFTER_4);
  ASSERT_EQ(g.actor(), PlayerId::P0);
  const size_t root_len = g.action_history().size();
  LambdaEvaluator eval([root_len](const GameState& s) {
    const auto h = s.action_history();
    if (h.size() <= root_len) return fixed_winner(s, std::nullopt, 0.0f);
    const bool ten = h[root_len].type == ActionType::CHOOSE_TEN_PROPOSALS;
    return fixed_winner(s, PlayerId::P0, ten ? 0.9f : 0.6f);
  });
  for (RootMode mode : {RootMode::PUCT, RootMode::GUMBEL}) {
    const SearchResult r = Tree(g, cfg(200, mode)).run(eval);
    EXPECT_EQ(r.action, Action::ten_proposals()) << "mode " << static_cast<int>(mode) << "\n" << dump(r);
    // Child values are exact: ten line 0.9, direct line 0.6 (for P0, the root actor).
    for (const auto& c : r.children) {
      if (c.visits == 0) continue;
      EXPECT_NEAR(c.q, c.action == Action::ten_proposals() ? 0.9f : 0.6f, 1e-3f) << dump(r);
    }
  }
}

TEST(Search, GumbelPolicyTargetIsADistribution) {
  std::mt19937_64 rng(3);
  GameState g = tools::random_midgame(rng, make_profile(RuleProfile::renju_rif()), 10, 20);
  UniformEvaluator u;
  SearchConfig c = cfg(32, RootMode::GUMBEL);
  const SearchResult r = Tree(g, c).run(u);
  float s = 0;
  for (float p : r.policy_target) s += p;
  if (!r.policy_target.empty()) EXPECT_NEAR(s, 1.0f, 1e-4f);
  EXPECT_TRUE(g.is_legal(r.action));
  EXPECT_LE(r.simulations, 32);
}

// The batched API with one leaf in flight must reproduce the synchronous search exactly.
TEST(Search, BatchedPathMatchesSynchronous) {
  std::mt19937_64 rng(5);
  const features::FeatureConfig fc;
  std::vector<uint8_t> buf(static_cast<size_t>(fc.num_planes()) * kNumPoints);
  for (int trial = 0; trial < 5; ++trial) {
    GameState g = tools::random_midgame(rng, make_profile(RuleProfile::renju_rif()), 8, 40);
    if (g.is_terminal()) continue;
    for (RootMode mode : {RootMode::PUCT, RootMode::GUMBEL}) {
      UniformEvaluator u;
      const SearchResult a = Tree(g, cfg(80, mode)).run(u);
      Tree t(g, cfg(80, mode));
      while (!t.done()) {
        const int ticket = t.select(buf.data());
        if (ticket >= 0) t.complete(ticket, Evaluation{});
        else if (ticket == -2) break;
      }
      const SearchResult b = t.result();
      EXPECT_EQ(a.action, b.action);
      ASSERT_EQ(a.children.size(), b.children.size());
      for (size_t i = 0; i < a.children.size(); ++i) EXPECT_EQ(a.children[i].visits, b.children[i].visits);
    }
  }
}

TEST(Search, VirtualLossBatchingKeepsInvariants) {
  std::mt19937_64 rng(7);
  const features::FeatureConfig fc;
  std::vector<std::vector<uint8_t>> bufs(8, std::vector<uint8_t>(static_cast<size_t>(fc.num_planes()) * kNumPoints));
  for (int trial = 0; trial < 6; ++trial) {
    GameState g = tools::random_midgame(rng, make_profile(trial % 2 ? RuleProfile::renju_rif() : RuleProfile::omokdojang_dan()),
                                        5, 40);
    if (g.is_terminal()) continue;
    SearchConfig c = cfg(150, trial % 3 == 0 ? RootMode::GUMBEL : RootMode::PUCT);
    c.max_inflight = 8;
    Tree t(g, c);
    int guard = 0;
    while (!t.done() && guard++ < 10000) {
      std::vector<int> tickets;
      for (int k = 0; k < 8; ++k) {
        const int ticket = t.select(bufs[k].data());
        if (ticket >= 0) tickets.push_back(ticket);
        if (ticket == -2) break;
      }
      for (int ticket : tickets) t.complete(ticket, Evaluation{});
    }
    EXPECT_EQ(t.inflight(), 0);
    std::string why;
    EXPECT_TRUE(t.check_invariants(&why)) << why;
    EXPECT_TRUE(g.is_legal(t.result().action));
  }
}

TEST(Search, AnyPhaseProducesALegalAction) {
  std::mt19937_64 rng(9);
  UniformEvaluator u;
  for (int i = 0; i < 40; ++i) {
    GameState g(make_profile(RuleProfile::omokdojang_dan()));
    const int steps = static_cast<int>(rng() % 30);
    for (int k = 0; k < steps && !g.is_terminal(); ++k) {
      auto acts = g.legal_actions();
      g.apply(acts[rng() % acts.size()]);
    }
    if (g.is_terminal()) continue;
    Tree t(g, cfg(40, i % 2 ? RootMode::GUMBEL : RootMode::PUCT));
    const SearchResult r = t.run(u);
    EXPECT_TRUE(g.is_legal(r.action)) << phase_name(g.phase()) << " " << to_string(r.action);
    std::string why;
    EXPECT_TRUE(t.check_invariants(&why)) << why;
  }
}

// Regression: Gumbel candidate set made only of proven-losing moves stalled self-play.
// Every search must finish its budget (or prove the root) without stalling, in tactical midgames.
TEST(Search, NeverStallsInTacticalPositions) {
  std::mt19937_64 rng(13);
  UniformEvaluator u;
  int stalled = 0, searched = 0;
  for (int i = 0; i < 300; ++i) {
    GameState g = tools::random_midgame(rng, make_profile(i % 2 ? RuleProfile::renju_rif() : RuleProfile::omokdojang_dan()),
                                        10, 70);
    if (g.is_terminal()) continue;
    for (RootMode mode : {RootMode::PUCT, RootMode::GUMBEL}) {
      SearchConfig c = cfg(64, mode);
      c.gumbel_m = 4;  // small candidate sets make the all-losing case likely
      Tree t(g, c);
      const SearchResult r = t.run(u);
      ++searched;
      if (r.stalled || !t.done()) {
        ++stalled;
        ADD_FAILURE() << "stalled in mode " << static_cast<int>(mode) << "\n" << g.to_string() << dump(r);
      }
      ASSERT_TRUE(g.is_legal(r.action));
    }
  }
  EXPECT_EQ(stalled, 0);
  EXPECT_GT(searched, 400);
}

// Time-limited search (external tournaments are time-controlled): stops near max_ms with
// `simulations` only a cap, always completes at least one simulation, returns a legal action.
TEST(Search, TimeLimitStopsSearch) {
  std::mt19937_64 rng(5);
  GameState g = tools::random_midgame(rng, make_profile(RuleProfile::renju_rif()), 10, 20);
  LambdaEvaluator slow([](const GameState&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return Evaluation{};
  });
  SearchConfig c = cfg(1000000);
  c.max_ms = 60;
  const SearchResult r = Tree(g, c).run(slow);
  EXPECT_TRUE(g.is_legal(r.action));
  EXPECT_GE(r.simulations, 2);
  EXPECT_LT(r.simulations, 1000000);
  EXPECT_GE(r.elapsed_ms, 60.0);
  EXPECT_LT(r.elapsed_ms, 60.0 + 250.0) << "one evaluation past the limit at most (plus scheduler jitter)";

  c.max_ms = 0.001;  // below one evaluation: still one simulation, never an empty result
  const SearchResult r1 = Tree(g, c).run(slow);
  EXPECT_EQ(r1.simulations, 1);
  EXPECT_TRUE(g.is_legal(r1.action));
}

TEST(Search, TimeLimitCapsRootTactics) {
  // Find a position whose root tactics take long with the node budget alone, then check that the
  // time limit caps them (tactics get root_tactics_time_share of max_ms).
  std::mt19937_64 rng(9);
  const uint64_t n = 400000;
  for (int k = 0; k < 40; ++k) {
    GameState g = tools::random_midgame(rng, make_profile(RuleProfile::renju_rif()), 20, 34);
    if (g.is_terminal()) continue;
    const auto t0 = std::chrono::steady_clock::now();
    tactics::analyze_tactics(g, {n, n, n / 2, 40, 0});
    const double slow_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (slow_ms < 400) continue;
    UniformEvaluator u;
    SearchConfig c = cfg(1000000);
    c.root_tactics_nodes = n;
    c.max_ms = 100;
    const SearchResult r = Tree(g, c).run(u);
    EXPECT_TRUE(g.is_legal(r.action));
    EXPECT_LT(r.elapsed_ms, 100.0 + 250.0) << "unbounded root tactics took " << slow_ms << " ms";
    return;
  }
  GTEST_SKIP() << "no position with slow root tactics found";
}

TEST(Bots, HeuristicTakesWinsAndBlocks) {
  std::mt19937_64 rng(1);
  GameState win = setup({"F8", "G8", "H8", "I8"}, {"B15", "E15", "H15", "K15"}, Color::BLACK);
  const Action a = heuristic_action(win, rng);
  EXPECT_TRUE(a == Action::place(P("E8")) || a == Action::place(P("J8")));
  GameState block = setup({"E8", "A1", "A4", "A7"}, {"F8", "G8", "H8", "I8"}, Color::BLACK);
  EXPECT_EQ(heuristic_action(block, rng), Action::place(P("J8")));
  for (int i = 0; i < 20; ++i) {
    GameState g = tools::random_midgame(rng, make_profile(RuleProfile::omokdojang_dan()), 0, 30);
    if (g.is_terminal()) continue;
    EXPECT_TRUE(g.is_legal(heuristic_action(g, rng)));
    EXPECT_TRUE(g.is_legal(random_action(g, rng)));
  }
}

}  // namespace
}  // namespace dz::search
