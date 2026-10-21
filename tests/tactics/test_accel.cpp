// Solver acceleration (PROGRESS Phase 8 E3b): the optimized df-pn / analyze_tactics must agree with the
// frozen ee1d011 solver (dz::tactics::ref) bit for bit under node budgets, every acceleration switch
// must be result-neutral, no proof may appear that the independent verifier refutes, and the
// solver must leave its input untouched and be safe to run on many threads at once.
#include <gtest/gtest.h>

#include <random>
#include <thread>

#include "board/transform.h"
#include "common/test_util.h"
#include "game/position_spec.h"
#include "tactics/engine.h"
#include "tactics/reference/ref.h"
#include "tactics/verify.h"
#include "tools/random_positions.h"

namespace dz::tactics {
namespace {

#ifdef NDEBUG
constexpr int kScale = 2;
#else
constexpr int kScale = 1;
#endif

// Random positions in normal Renju play (RIF and Omokdojang dan), dense enough for tactics.
std::vector<GameState> random_positions(uint64_t seed, int n) {
  std::mt19937_64 rng(seed);
  std::vector<GameState> out;
  for (int i = 0; static_cast<int>(out.size()) < n && i < n * 20; ++i) {
    auto profile = make_profile(i % 3 == 2 ? RuleProfile::omokdojang_dan() : RuleProfile::renju_rif());
    GameState g = tools::random_midgame(rng, profile, 18, 70);
    if (g.phase() == OpeningPhase::NORMAL_RENJU && !g.is_terminal()) out.push_back(g);
  }
  return out;
}

void expect_same(const DfpnResult& a, const DfpnResult& b, const std::string& what) {
  EXPECT_EQ(a.result, b.result) << what;
  EXPECT_EQ(a.nodes, b.nodes) << what;
  EXPECT_EQ(a.best, b.best) << what;
  EXPECT_EQ(a.pv, b.pv) << what;
  EXPECT_EQ(a.budget_exhausted, b.budget_exhausted) << what;
  EXPECT_EQ(a.proof_moves, b.proof_moves) << what;
}

DfpnLimits limits_with(uint64_t nodes, bool cache, bool lazy, bool fast) {
  DfpnLimits l{nodes, 40, false, 0};
  l.gen_cache = cache;
  l.lazy_legality = lazy;
  l.fast_refute = fast;
  return l;
}

// 2/8: optimized vs reference, and every acceleration switch on/off: identical results and trees.
TEST(SolverAccel, DfpnMatchesReferenceAndSwitchesAreNeutral) {
  int compared = 0;
  for (const GameState& g : random_positions(101, 30 * kScale)) {
    auto pos = TPos::from_state(g);
    ASSERT_TRUE(pos);
    for (DfpnMode mode : {DfpnMode::PROVE_WIN, DfpnMode::PROVE_LOSS}) {
      const DfpnResult r = ref::solve_dfpn(*pos, mode, {3000, 40, false, 0});
      for (int m = 0; m < 8; ++m) {
        const DfpnResult o = solve_dfpn(*pos, mode, limits_with(3000, m & 1, m & 2, m & 4));
        expect_same(r, o, "switches " + std::to_string(m) + "\n" + g.to_string());
      }
      ++compared;
    }
  }
  EXPECT_GT(compared, 20);
}

// Same with non-default black options (fast_refute must fall back to the exact path).
TEST(SolverAccel, NonDefaultBlackOptionsMatchReference) {
  for (const GameState& g : random_positions(102, 12 * kScale)) {
    auto p0 = TPos::from_state(g);
    ASSERT_TRUE(p0);
    Rules rules = p0->rules();
    rules.black.exact_five_precedence = false;
    rules.black.five_beats_overline_other_dir = false;
    const TPos pos(p0->board(), p0->to_move(), rules);
    expect_same(ref::solve_dfpn(pos, DfpnMode::PROVE_WIN, {2000, 40, false, 0}),
                solve_dfpn(pos, DfpnMode::PROVE_WIN, limits_with(2000, true, true, true)), g.to_string());
  }
}

// Dense black clusters (overlines, 4-4, 3-3, recursive false-forbidden) with both colours attacking.
TEST(SolverAccel, DenseForbiddenBoardsMatchReference) {
  std::mt19937_64 rng(103);
  const Rules rif{RuleProfile::renju_rif().black, false};
  int compared = 0;
  for (int i = 0; i < 40 * kScale; ++i) {
    const Board b = tools::random_dense_board(rng);
    for (Color side : {Color::BLACK, Color::WHITE}) {
      const TPos pos(b, side, rif);
      std::vector<Point> f;
      five_points(b, Color::BLACK, rif, f);
      if (!f.empty()) continue;  // keep positions where neither side has already won
      five_points(b, Color::WHITE, rif, f);
      if (!f.empty()) continue;
      expect_same(ref::solve_dfpn(pos, DfpnMode::PROVE_WIN, {1500, 40, false, 0}),
                  solve_dfpn(pos, DfpnMode::PROVE_WIN, limits_with(1500, true, true, true)), b.to_string());
      ++compared;
    }
  }
  EXPECT_GT(compared, 10);
}

// analyze_tactics without a clock (the time split is irrelevant then) equals the reference.
TEST(SolverAccel, AnalyzeMatchesReferenceWithoutClock) {
  for (const GameState& g : random_positions(104, 25 * kScale)) {
    const TacticalLimits lim{20000, 3000, 2000, 40, 0};
    const TacticalDecision a = ref::analyze_tactics(g, lim), b = analyze_tactics(g, lim);
    EXPECT_EQ(a.result, b.result) << g.to_string();
    EXPECT_EQ(a.reason, b.reason) << g.to_string();
    EXPECT_EQ(a.move, b.move) << g.to_string();
    EXPECT_EQ(a.nodes, b.nodes) << g.to_string();
    EXPECT_EQ(a.pv, b.pv) << g.to_string();
  }
}

// 10: no false proof under any node budget, with or without the time carry-over.
TEST(SolverAccel, BudgetLimitedClaimsAreSound) {
  int claims = 0;
  for (const GameState& g : random_positions(105, 20 * kScale)) {
    for (uint64_t budget : {30ull, 300ull, 3000ull}) {
      for (bool carry : {false, true}) {
        TacticalLimits lim{budget, budget, budget, 30, carry ? 5.0 : 0.0};
        lim.carry_unused_time = carry;
        const TacticalDecision d = analyze_tactics(g, lim);
        if (d.result == ProofResult::UNKNOWN) continue;
        ++claims;
        ProofMoveFn oracle = [&](uint64_t key) -> std::optional<Point> {
          auto it = d.proof_moves.find(key);
          if (it == d.proof_moves.end()) return std::nullopt;
          return it->second;
        };
        const VerifyStatus v = d.result == ProofResult::WIN ? verify_forced_win(g, oracle, {60, 200000})
                                                            : verify_forced_loss(g, oracle, {60, 200000});
        ASSERT_NE(v, VerifyStatus::REFUTED) << d.to_string() << "\n" << g.to_string();
      }
    }
  }
  EXPECT_GT(claims, 0);
}

// 3: D4 symmetry. Move order follows point numbering, so trees (and budget-limited outcomes) can
// differ between images; a proof in one image must never be contradicted in another, and with a
// generous budget proofs must be found in every image.
TEST(SolverAccel, D4ImagesNeverContradict) {
  int proven = 0;
  for (const GameState& g : random_positions(106, 4 * kScale)) {
    auto pos = TPos::from_state(g);
    ASSERT_TRUE(pos);
    const DfpnResult base = solve_dfpn(*pos, DfpnMode::PROVE_WIN, limits_with(5000, true, true, true));
    for (int t = 1; t < 8; ++t) {
      const TPos img(pos->board().transformed(t), pos->to_move(), pos->rules());
      const DfpnResult r = solve_dfpn(img, DfpnMode::PROVE_WIN, limits_with(5000, true, true, true));
      const DfpnResult l = solve_dfpn(img, DfpnMode::PROVE_LOSS, limits_with(5000, true, true, true));
      EXPECT_FALSE(r.result == ProofResult::WIN && l.result == ProofResult::LOSS) << g.to_string();
      if (base.result == ProofResult::WIN && !base.budget_exhausted) {
        EXPECT_NE(l.result, ProofResult::LOSS) << "t=" << t << "\n" << g.to_string();
      }
      if (base.result == ProofResult::WIN && r.result == ProofResult::WIN) {
        ++proven;
        EXPECT_EQ(r.best == kNoPoint, base.best == kNoPoint);
      }
    }
  }
  SUCCEED() << proven;
}

// 1/11: the solver works on its own copy; play/undo (incl. fast_refute's place/remove) restore the
// board, all eight Zobrist hashes and the side to move exactly.
TEST(SolverAccel, InputPositionUnchangedAndPlayUndoRestores) {
  std::mt19937_64 rng(107);
  for (const GameState& g : random_positions(108, 10 * kScale)) {
    auto pos = TPos::from_state(g);
    ASSERT_TRUE(pos);
    const TPos before = *pos;
    solve_dfpn(*pos, DfpnMode::PROVE_WIN, limits_with(2000, true, true, true));
    EXPECT_EQ(pos->board().cells(), before.board().cells());
    EXPECT_EQ(pos->key(), before.key());
    // Stress: random legal play/undo sequences restore everything.
    TPos p = *pos;
    std::vector<Point> played;
    for (int step = 0; step < 400; ++step) {
      if (!played.empty() && (rng() % 3 == 0 || played.size() > 30)) {
        p.undo(played.back());
        played.pop_back();
        continue;
      }
      const Point m = static_cast<Point>(rng() % kNumPoints);
      if (!p.board().empty_at(m)) continue;
      p.play(m);
      played.push_back(m);
    }
    while (!played.empty()) {
      p.undo(played.back());
      played.pop_back();
    }
    EXPECT_EQ(p.board().cells(), before.board().cells());
    for (int t = 0; t < 8; ++t) EXPECT_EQ(p.board().hash(t), before.board().hash(t));
    EXPECT_EQ(p.to_move(), before.to_move());
  }
}

// 9: many solver calls in parallel (as the root defense filter does) give the sequential results.
TEST(SolverAccel, ParallelCallsMatchSequential) {
  const std::vector<GameState> ps = random_positions(109, 16 * kScale);
  const TacticalLimits lim{20000, 2000, 0, 40, 0};
  std::vector<TacticalDecision> seq, par(ps.size());
  for (const auto& g : ps) seq.push_back(analyze_tactics(g, lim));
  std::vector<std::thread> pool;
  for (int t = 0; t < 8; ++t) {
    pool.emplace_back([&, t] {
      for (size_t i = t; i < ps.size(); i += 8) par[i] = analyze_tactics(ps[i], lim);
    });
  }
  for (auto& th : pool) th.join();
  for (size_t i = 0; i < ps.size(); ++i) {
    EXPECT_EQ(seq[i].result, par[i].result);
    EXPECT_EQ(seq[i].nodes, par[i].nodes);
    EXPECT_EQ(seq[i].move, par[i].move);
  }
}

// UNKNOWN is never LOSS: a budget-exhausted search reports UNKNOWN in both modes.
TEST(SolverAccel, ExhaustedBudgetIsUnknown) {
  for (const GameState& g : random_positions(110, 10 * kScale)) {
    auto pos = TPos::from_state(g);
    ASSERT_TRUE(pos);
    for (DfpnMode mode : {DfpnMode::PROVE_WIN, DfpnMode::PROVE_LOSS}) {
      const DfpnResult r = solve_dfpn(*pos, mode, limits_with(3, true, true, true));
      if (r.budget_exhausted) EXPECT_EQ(r.result, ProofResult::UNKNOWN);
    }
  }
}

}  // namespace
}  // namespace dz::tactics
