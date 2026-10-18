// Tactical solvers: fixed benchmark suites, PV replay, and soundness fuzzing against the
// independent GameState verifier.
#include <gtest/gtest.h>

#include <random>

#include "common/test_util.h"
#include "game/position_spec.h"
#include "tactics/benchmark.h"
#include "tactics/engine.h"
#include "tactics/verify.h"
#include "tools/random_positions.h"

namespace dz::tactics {
namespace {

using nlohmann::json;

struct Problem {
  std::string file;
  json data;
};

std::vector<Problem> load_problems() {
  std::vector<Problem> out;
#ifdef NDEBUG
  const std::vector<const char*> dirs = {"tactical", "vcf", "vct"};
#else
  const std::vector<const char*> dirs = {"tactical"};  // generated suites are Release-only (slow verify)
#endif
  for (const char* dir : dirs) {
    for (const auto& path : test::json_files_in(test::source_dir() / "benchmarks" / dir)) {
      const json doc = test::read_json(path);
      for (const auto& p : doc.at("problems")) out.push_back({path.filename().string(), p});
    }
  }
  return out;
}

class TacticalBenchmark : public ::testing::TestWithParam<Problem> {};

TEST_P(TacticalBenchmark, Solves) {
  const ProblemOutcome o = run_problem(GetParam().data);
  EXPECT_FALSE(o.false_claim) << o.id << ": " << o.message;
  EXPECT_TRUE(o.pass) << o.id << ": " << o.message << " (claimed " << o.claimed << ", verification "
                      << o.verification << ")";
}

INSTANTIATE_TEST_SUITE_P(Suites, TacticalBenchmark, ::testing::ValuesIn(load_problems()),
                         [](const ::testing::TestParamInfo<Problem>& info) {
                           std::string s;
                           for (char ch : info.param.data.at("id").get<std::string>()) {
                             s += std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_';
                           }
                           return s;
                         });

TEST(TacticalSolvers, VcfPvReplaysToAWin) {
  std::mt19937_64 rng(41);
  auto profile = make_profile(RuleProfile::renju_rif());
  int found = 0;
  for (int i = 0; i < 200 && found < 25; ++i) {
    GameState g = tools::random_midgame(rng, profile, 20, 60);
    auto pos = TPos::from_state(g);
    if (!pos) continue;
    const VcfResult r = solve_vcf(*pos, {50000, 40});
    if (r.status != VcfResult::Status::FOUND) continue;
    ++found;
    GameState s = g;
    const Color attacker = g.side_to_move();
    for (Point m : r.pv) {
      if (m == kNoPoint) {  // defender cannot block: let it play anything legal
        auto a = s.legal_actions();
        ASSERT_FALSE(a.empty());
        ASSERT_TRUE(s.apply(a.front()));
        continue;
      }
      ASSERT_TRUE(s.apply(Action::place(m))) << to_notation(m) << "\n" << s.to_string();
    }
    ASSERT_TRUE(s.is_terminal());
    EXPECT_EQ(s.outcome().result, attacker == Color::BLACK ? GameResult::BLACK_WIN : GameResult::WHITE_WIN);
  }
  EXPECT_GT(found, 3);
}

// Every WIN / LOSS claim made on random midgame positions must survive full-width verification.
TEST(TacticalSolvers, ClaimsSurviveIndependentVerification) {
#ifdef NDEBUG
  constexpr int kPositions = 120;
#else
  constexpr int kPositions = 15;
#endif
  std::mt19937_64 rng(42);
  int claims = 0, verified = 0, too_large = 0;
  for (int i = 0; i < kPositions; ++i) {
    auto profile = make_profile(i % 2 ? RuleProfile::renju_rif() : RuleProfile::omokdojang_dan());
    GameState g = tools::random_midgame(rng, profile, 15, 70);
    if (g.phase() != OpeningPhase::NORMAL_RENJU || g.is_terminal()) continue;
    const TacticalDecision d = analyze_tactics(g, {20000, 10000, 5000, 30});
    if (d.result == ProofResult::UNKNOWN) continue;
    ++claims;
    ProofMoveFn oracle = [&](uint64_t key) -> std::optional<Point> {
      auto it = d.proof_moves.find(key);
      if (it == d.proof_moves.end()) return std::nullopt;
      return it->second;
    };
    const VerifyStatus v = d.result == ProofResult::WIN ? verify_forced_win(g, oracle, {60, 400000})
                                                        : verify_forced_loss(g, oracle, {60, 400000});
    ASSERT_NE(v, VerifyStatus::REFUTED) << d.to_string() << "\n" << g.to_string();
    verified += v == VerifyStatus::VERIFIED;
    too_large += v == VerifyStatus::TOO_LARGE;
  }
  std::cout << "[tactics-soundness] claims=" << claims << " verified=" << verified << " too_large=" << too_large
            << "\n";
  EXPECT_GT(claims, 0);
  EXPECT_GE(verified * 10, claims * 8) << "most claims should be small enough to verify";
}

// The verifier must reject wrong claims, not only accept right ones.
GameState setup(std::initializer_list<const char*> black, std::initializer_list<const char*> white, Color side) {
  PositionSpec spec;
  for (const char* s : black) spec.black.push_back(test::P(s));
  for (const char* s : white) spec.white.push_back(test::P(s));
  spec.side_to_move = side;
  std::string error;
  auto g = load_position(spec, make_profile(RuleProfile::renju_rif()), &error);
  EXPECT_TRUE(g) << error;
  return *g;
}

TEST(TacticalVerifier, RejectsBogusClaims) {
  const ProofMoveFn none = [](uint64_t) -> std::optional<Point> { return std::nullopt; };
  // Black open two: no forced win. A "proof" playing H8 (open three) must be refuted.
  GameState two = setup({"F8", "G8"}, {"B15", "E15"}, Color::BLACK);
  const uint64_t k = position_key(two.board(), Color::BLACK);
  const ProofMoveFn h8 = [k](uint64_t key) -> std::optional<Point> {
    if (key == k) return test::P("H8");
    return std::nullopt;
  };
  EXPECT_EQ(verify_forced_win(two, h8), VerifyStatus::REFUTED);
  EXPECT_EQ(verify_forced_win(two, none), VerifyStatus::REFUTED);
  // Side to move is not lost: a LOSS claim must be refuted.
  EXPECT_EQ(verify_forced_loss(two, none), VerifyStatus::REFUTED);
  // Proof move that is forbidden for black (3-3 at H8) is refuted.
  GameState f = setup({"F8", "G8", "H6", "H7"}, {"B15", "E15", "H15", "K15"}, Color::BLACK);
  const uint64_t kf = position_key(f.board(), Color::BLACK);
  const ProofMoveFn bad = [kf](uint64_t key) -> std::optional<Point> {
    if (key == kf) return test::P("H8");
    return std::nullopt;
  };
  EXPECT_EQ(verify_forced_win(f, bad), VerifyStatus::REFUTED);
  // Sanity: an open four is verified without any proof entries.
  GameState four = setup({"F8", "G8", "H8"}, {"B15", "E15"}, Color::BLACK);
  EXPECT_EQ(verify_forced_win(four, none), VerifyStatus::VERIFIED);
}

TEST(TacticalSolvers, NotInNormalRenjuIsUnknown) {
  GameState g(make_profile(RuleProfile::taraguchi10()));
  const TacticalDecision d = analyze_tactics(g);
  EXPECT_EQ(d.result, ProofResult::UNKNOWN);
  EXPECT_EQ(d.reason, TacticReason::NONE);
}

}  // namespace
}  // namespace dz::tactics
