// Proof-aware MCTS (PROGRESS Phase 8 E3d): solver proofs attached to tree nodes while searching.
// Proofs must be sound (independent verifier), a proven root win must be played, trees must stay
// consistent, and destroying a tree with attempts still running must be safe.
#include <gtest/gtest.h>

#include <random>
#include <thread>

#include "search/proof_pool.h"
#include "search/search.h"
#include "tactics/dfpn.h"
#include "tactics/engine.h"
#include "tactics/vcf.h"
#include "tactics/verify.h"
#include "tools/random_positions.h"

namespace dz::search {
namespace {

SearchConfig proof_cfg(int sims, int threads = 4) {
  SearchConfig c;
  c.simulations = sims;
  c.root = RootMode::PUCT;
  c.root_tactics_nodes = 0;
  c.max_inflight = 1;
  c.proof_threads = threads;
  c.proof_vcf_nodes = 5000;
  c.proof_vct_nodes = 1000;
  c.proof_max_vct_nodes = 16000;
  c.proof_min_visits = 2;
  c.proof_max_depth = 2;
  c.proof_scan_every = 1;
  return c;
}

std::vector<GameState> midgames(uint64_t seed, int n) {
  std::mt19937_64 rng(seed);
  std::vector<GameState> out;
  auto profile = make_profile(RuleProfile::renju_rif());
  for (int i = 0; static_cast<int>(out.size()) < n && i < n * 20; ++i) {
    GameState g = tools::random_midgame(rng, profile, 20, 60);
    if (g.phase() == OpeningPhase::NORMAL_RENJU && !g.is_terminal()) out.push_back(g);
  }
  return out;
}

// Every root child the tree marks as lost for the root actor must not be contradicted: an
// independent df-pn proof of the opponent's win (when found within its budget) must survive the
// GameState verifier, and the opponent must never be provably lost there.
TEST(ProofMcts, ProvenRootChildrenAreSound) {
  ProofCache::instance().clear();
  int proven_children = 0, verified = 0;
  for (const GameState& g : midgames(301, 12)) {
    UniformEvaluator u;
    Tree t(g, proof_cfg(600));
    const SearchResult r = t.run(u);
    std::string why;
    ASSERT_TRUE(t.check_invariants(&why)) << why;
    for (const ChildStat& c : r.children) {
      if (c.proven == 2) continue;
      ++proven_children;
      GameState child = g;
      ASSERT_TRUE(child.apply(c.action));
      if (child.is_terminal()) continue;
      auto pos = tactics::TPos::from_state(child);
      ASSERT_TRUE(pos);
      // c.proven is for the root actor; the opponent is to move in `child`.
      const tactics::DfpnMode mode = c.proven < 0 ? tactics::DfpnMode::PROVE_WIN : tactics::DfpnMode::PROVE_LOSS;
      const tactics::DfpnResult other = tactics::solve_dfpn(
          *pos, c.proven < 0 ? tactics::DfpnMode::PROVE_LOSS : tactics::DfpnMode::PROVE_WIN, {20000, 40, false, 0});
      EXPECT_NE(other.result, c.proven < 0 ? tactics::ProofResult::LOSS : tactics::ProofResult::WIN)
          << "contradicted proof\n" << child.to_string();
      const tactics::DfpnResult d = tactics::solve_dfpn(*pos, mode, {60000, 40, false, 0});
      if (d.result == tactics::ProofResult::UNKNOWN) continue;
      tactics::ProofMoveFn oracle = [&](uint64_t key) -> std::optional<Point> {
        auto it = d.proof_moves.find(key);
        if (it == d.proof_moves.end()) return std::nullopt;
        return it->second;
      };
      const tactics::VerifyStatus v = d.result == tactics::ProofResult::WIN
                                          ? tactics::verify_forced_win(child, oracle, {60, 400000})
                                          : tactics::verify_forced_loss(child, oracle, {60, 400000});
      ASSERT_NE(v, tactics::VerifyStatus::REFUTED) << child.to_string();
      verified += v == tactics::VerifyStatus::VERIFIED;
    }
  }
  std::cout << "[proof-mcts] proven root children=" << proven_children << " verified=" << verified << "\n";
}

// A VCF win found only by the proof workers (root tactics off) is played.
TEST(ProofMcts, ProvenRootWinIsPlayed) {
  ProofCache::instance().clear();
  std::mt19937_64 rng(302);
  auto profile = make_profile(RuleProfile::renju_rif());
  int checked = 0;
  for (int i = 0; i < 400 && checked < 4; ++i) {
    GameState g = tools::random_midgame(rng, profile, 20, 60);
    auto pos = tactics::TPos::from_state(g);
    if (!pos) continue;
    std::vector<Point> five;
    tactics::five_points(pos->board(), pos->to_move(), pos->rules(), five);
    if (!five.empty()) continue;  // an immediate win is handled without proofs
    if (tactics::solve_vcf(*pos, {20000, 40}).status != tactics::VcfResult::Status::FOUND) continue;
    ++checked;
    UniformEvaluator u;
    SearchConfig c = proof_cfg(4000);
    c.proof_min_visits = 1;
    c.proof_vcf_nodes = 20000;
    const SearchResult r = Tree(g, c).run(u);
    EXPECT_EQ(r.proven, 1) << g.to_string();
    GameState after = g;
    ASSERT_TRUE(after.apply(r.action));
    if (after.is_terminal()) continue;
    auto p2 = tactics::TPos::from_state(after);
    ASSERT_TRUE(p2);
    // The opponent (to move) must be lost after the chosen move.
    EXPECT_EQ(tactics::solve_dfpn(*p2, tactics::DfpnMode::PROVE_LOSS, {100000, 40, false, 0}).result,
              tactics::ProofResult::LOSS)
        << g.to_string() << " played " << to_string(r.action);
  }
  EXPECT_GT(checked, 0);
}

// Many trees at once (as in batched self-play), proofs on: invariants hold and nothing crashes;
// trees destroyed with attempts still queued or running are safe.
TEST(ProofMcts, ParallelTreesAndEarlyDestruction) {
  const std::vector<GameState> ps = midgames(303, 16);
  std::vector<std::thread> pool;
  std::atomic<int> bad{0};
  for (int t = 0; t < 4; ++t) {
    pool.emplace_back([&, t] {
      for (size_t i = t; i < ps.size(); i += 4) {
        UniformEvaluator u;
        Tree tree(ps[i], proof_cfg(300, 6));
        tree.run(u);
        std::string why;
        if (!tree.check_invariants(&why)) ++bad;
      }
    });
  }
  for (auto& th : pool) th.join();
  EXPECT_EQ(bad.load(), 0);
  for (int k = 0; k < 20; ++k) {  // destroy immediately after queueing large attempts
    SearchConfig c = proof_cfg(3, 8);
    c.proof_vct_nodes = 200000;
    c.proof_max_vct_nodes = 200000;
    c.proof_min_visits = 1;
    UniformEvaluator u;
    Tree tree(ps[k % ps.size()], c);
    tree.run(u);
  }
  SUCCEED();
}

// Proofs off: identical to the plain search (same seed, synchronous evaluator).
TEST(ProofMcts, OffIsUnchanged) {
  for (const GameState& g : midgames(304, 6)) {
    UniformEvaluator u1, u2;
    SearchConfig a = proof_cfg(200, 0), b = proof_cfg(200, 0);
    b.proof_vct_nodes = 999;  // irrelevant when off
    const SearchResult r1 = Tree(g, a).run(u1), r2 = Tree(g, b).run(u2);
    EXPECT_EQ(to_string(r1.action), to_string(r2.action));
    EXPECT_EQ(r1.simulations, r2.simulations);
    EXPECT_EQ(r1.proof_jobs, 0);
  }
}

// The cache never turns UNKNOWN into a result.
TEST(ProofMcts, CacheKeepsUnknownSeparate) {
  ProofCache& c = ProofCache::instance();
  c.clear();
  c.put_unknown(42, 1000);
  EXPECT_EQ(c.get(42).exact, 0);
  EXPECT_EQ(c.get(42).tried_vct, 1000u);
  c.put_exact(42, +1, 7);
  EXPECT_EQ(c.get(42).exact, 1);
  EXPECT_EQ(c.get(42).move, 7);
  c.clear();
}

}  // namespace
}  // namespace dz::search
