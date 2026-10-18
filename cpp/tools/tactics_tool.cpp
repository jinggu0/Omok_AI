// dojozero_tactics: tactical benchmark generation, soundness fuzzing and performance.
//
//   dojozero_tactics generate --kind vcf|vct|loss --count 40 --seed 1 --out benchmarks/vcf/generated.json
//   dojozero_tactics fuzz --positions 5000 --threads 8 --seed 1
//   dojozero_tactics perf --positions 300 --seed 7
//
// generate: random NORMAL_RENJU midgames; keeps positions whose claim the independent verifier
// VERIFIES (never solver-only results). fuzz: every WIN/LOSS claim of analyze_tactics must survive
// verification (exit 1 on any REFUTED).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "tactics/engine.h"
#include "tactics/verify.h"
#include "tools/random_positions.h"

using namespace dz;
using namespace dz::tactics;
using nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

std::map<std::string, std::string> parse_flags(int argc, char** argv) {
  std::map<std::string, std::string> f;
  for (int i = 2; i + 1 < argc; i += 2) f[std::string(argv[i]).substr(2)] = argv[i + 1];
  return f;
}

ProfilePtr profile_for(int i) {
  return make_profile(i % 2 ? RuleProfile::renju_rif() : RuleProfile::omokdojang_dan());
}

ProofMoveFn oracle(const std::unordered_map<uint64_t, Point>& proof) {
  return [&proof](uint64_t key) -> std::optional<Point> {
    auto it = proof.find(key);
    if (it == proof.end()) return std::nullopt;
    return it->second;
  };
}

json position_json(const GameState& g) {
  std::vector<std::string> black, white;
  for (int p = 0; p < kNumPoints; ++p) {
    const Cell c = g.board().at(static_cast<Point>(p));
    if (c == BLACK_STONE) black.push_back(to_notation(static_cast<Point>(p)));
    if (c == WHITE_STONE) white.push_back(to_notation(static_cast<Point>(p)));
  }
  return {{"black", black}, {"white", white}, {"side", color_name(g.side_to_move())}};
}

int cmd_generate(const std::map<std::string, std::string>& f) {
  const std::string kind = f.at("kind");
  const int count = std::atoi(f.at("count").c_str());
  const uint64_t seed = std::strtoull(f.at("seed").c_str(), nullptr, 10);
  const int min_pv = f.count("min-pv") ? std::atoi(f.at("min-pv").c_str()) : 5;
  std::mt19937_64 rng(seed);
  json problems = json::array();
  std::set<uint64_t> seen;
  long long tried = 0, rejected_unverified = 0;
  for (int i = 0; static_cast<int>(problems.size()) < count && i < 200000; ++i) {
    auto profile = profile_for(i);
    GameState g = tools::random_midgame(rng, profile, 16, 80);
    auto pos = TPos::from_state(g);
    if (!pos || !seen.insert(g.canonical_hash()).second) continue;
    std::vector<Point> five;
    five_points(pos->board(), pos->to_move(), pos->rules(), five);
    if (!five.empty()) continue;  // trivial
    five_points(pos->board(), opposite(pos->to_move()), pos->rules(), five);
    if (!five.empty()) continue;  // forced block positions are trivial
    ++tried;
    std::unordered_map<uint64_t, Point> proof;
    std::string query, expected;
    size_t pv_len = 0;
    VerifyStatus v = VerifyStatus::REFUTED;
    if (kind == "vcf") {
      const VcfResult r = solve_vcf(*pos, {200000, 40});
      if (r.status != VcfResult::Status::FOUND || r.pv.size() < static_cast<size_t>(min_pv)) continue;
      proof = proof_from_pv(g, r.pv);
      pv_len = r.pv.size();
      v = verify_forced_win(g, oracle(proof), {80, 5000000});
      query = "vcf";
      expected = "FOUND";
    } else if (kind == "vct") {
      if (solve_vcf(*pos, {200000, 40}).status != VcfResult::Status::NONE) continue;  // genuinely needs threes
      const DfpnResult r = solve_dfpn(*pos, DfpnMode::PROVE_WIN, {100000, 40, false});
      if (r.result != ProofResult::WIN || r.pv.size() < static_cast<size_t>(min_pv)) continue;
      proof = r.proof_moves;
      pv_len = r.pv.size();
      v = verify_forced_win(g, oracle(proof), {80, 5000000});
      query = "vct";
      expected = "WIN";
    } else if (kind == "loss") {
      const DfpnResult r = solve_dfpn(*pos, DfpnMode::PROVE_LOSS, {100000, 40, false});
      if (r.result != ProofResult::LOSS || r.pv.size() < static_cast<size_t>(min_pv)) continue;
      proof = r.proof_moves;
      pv_len = r.pv.size();
      v = verify_forced_loss(g, oracle(proof), {80, 5000000});
      query = "loss";
      expected = "LOSS";
    } else {
      std::fprintf(stderr, "unknown kind %s\n", kind.c_str());
      return 2;
    }
    if (v != VerifyStatus::VERIFIED) {
      ++rejected_unverified;
      if (v == VerifyStatus::REFUTED) {
        std::printf("REFUTED claim (solver bug):\n%s\n", g.to_string().c_str());
        return 1;
      }
      continue;
    }
    json p;
    p["id"] = "gen-" + kind + "-" + std::to_string(seed) + "-" + std::to_string(problems.size());
    p["category"] = kind == "vcf" ? "VCF" : kind == "vct" ? "VCT" : "proven-loss";
    p["source"] = {{"kind", "generated+verified"},
                   {"generator", "dojozero_tactics generate --kind " + kind + " --seed " + std::to_string(seed)},
                   {"note", "claim verified full-width by tactics/verify (GameState only); pv length " +
                                std::to_string(pv_len)}};
    p["profile"] = g.profile().id;
    p["position"] = position_json(g);
    p["query"] = query;
    p["expected"] = {{"result", expected}};
    problems.push_back(p);
    std::printf("kept %zu/%d (pv %zu)\n", problems.size(), count, pv_len);
    std::fflush(stdout);
  }
  json doc;
  doc["schema_version"] = 1;
  doc["suite"] = "generated-" + kind;
  doc["description"] = "Machine-generated " + kind +
                       " problems from random midgames; every expected result was verified by the independent "
                       "full-width GameState verifier when generated. tried=" + std::to_string(tried) +
                       " unverifiable_skipped=" + std::to_string(rejected_unverified);
  doc["problems"] = problems;
  std::ofstream(f.at("out")) << doc.dump(1) << "\n";
  std::printf("wrote %zu problems to %s\n", problems.size(), f.at("out").c_str());
  return 0;
}

int cmd_fuzz(const std::map<std::string, std::string>& f) {
  const int positions = std::atoi(f.at("positions").c_str());
  const int threads = f.count("threads") ? std::atoi(f.at("threads").c_str()) : 8;
  const uint64_t seed = std::strtoull(f.at("seed").c_str(), nullptr, 10);
  std::atomic<int> next{0};
  std::atomic<long long> claims{0}, wins{0}, losses{0}, verified{0}, too_large{0}, refuted{0};
  std::mutex mu;
  const auto t0 = Clock::now();
  auto worker = [&](int tid) {
    std::mt19937_64 rng(seed * 7919 + tid);
    while (next.fetch_add(1) < positions) {
      GameState g = tools::random_midgame(rng, profile_for(static_cast<int>(rng() % 2)), 10, 90);
      if (g.phase() != OpeningPhase::NORMAL_RENJU || g.is_terminal()) continue;
      const TacticalDecision d = analyze_tactics(g, {20000, 10000, 5000, 30});
      if (d.result == ProofResult::UNKNOWN) continue;
      claims++;
      (d.result == ProofResult::WIN ? wins : losses)++;
      const VerifyStatus v = d.result == ProofResult::WIN ? verify_forced_win(g, oracle(d.proof_moves), {60, 2000000})
                                                          : verify_forced_loss(g, oracle(d.proof_moves), {60, 2000000});
      if (v == VerifyStatus::VERIFIED) verified++;
      if (v == VerifyStatus::TOO_LARGE) too_large++;
      if (v == VerifyStatus::REFUTED) {
        refuted++;
        std::lock_guard<std::mutex> lock(mu);
        std::printf("REFUTED %s\n%s\n", d.to_string().c_str(), g.to_string().c_str());
      }
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();
  std::printf("positions=%d claims=%lld (win %lld, loss %lld) verified=%lld too_large=%lld refuted=%lld time=%.0fs\n",
              positions, claims.load(), wins.load(), losses.load(), verified.load(), too_large.load(),
              refuted.load(), std::chrono::duration<double>(Clock::now() - t0).count());
  return refuted.load() == 0 ? 0 : 1;
}

int cmd_perf(const std::map<std::string, std::string>& f) {
  const int n = std::atoi(f.at("positions").c_str());
  std::mt19937_64 rng(std::strtoull(f.at("seed").c_str(), nullptr, 10));
  std::vector<GameState> states;
  while (static_cast<int>(states.size()) < n) {
    GameState g = tools::random_midgame(rng, profile_for(static_cast<int>(states.size())), 20, 70);
    if (TPos::from_state(g)) states.push_back(g);
  }
  uint64_t vcf_nodes = 0, df_nodes = 0;
  double vcf_s = 0, df_s = 0;
  std::vector<double> decide_ms, capped_ms;
  int vcf_found = 0, vct_won = 0;
  for (const GameState& g : states) {
    const TPos pos = *TPos::from_state(g);
    auto t0 = Clock::now();
    const VcfResult v = solve_vcf(pos, {100000, 40});
    vcf_s += std::chrono::duration<double>(Clock::now() - t0).count();
    vcf_nodes += v.nodes;
    vcf_found += v.status == VcfResult::Status::FOUND;
    t0 = Clock::now();
    const DfpnResult d = solve_dfpn(pos, DfpnMode::PROVE_WIN, {20000, 40, false});
    df_s += std::chrono::duration<double>(Clock::now() - t0).count();
    df_nodes += d.nodes;
    vct_won += d.result == ProofResult::WIN;
    decide_ms.push_back(analyze_tactics(g).elapsed_ms);
    capped_ms.push_back(analyze_tactics(g, {100000, 50000, 20000, 40, 500}).elapsed_ms);
  }
  std::sort(decide_ms.begin(), decide_ms.end());
  std::sort(capped_ms.begin(), capped_ms.end());
  std::printf("positions=%d\n", n);
  std::printf("VCF: %.0f nodes/s (found in %d), df-pn VCT: %.0f nodes/s (proved %d)\n", vcf_nodes / vcf_s, vcf_found,
              df_nodes / df_s, vct_won);
  std::printf("analyze_tactics latency (default limits): p50=%.1fms p90=%.1fms p99=%.1fms max=%.1fms\n",
              decide_ms[decide_ms.size() / 2], decide_ms[decide_ms.size() * 9 / 10],
              decide_ms[decide_ms.size() * 99 / 100], decide_ms.back());
  std::printf("analyze_tactics latency (max_ms=500): p50=%.1fms p90=%.1fms p99=%.1fms max=%.1fms\n",
              capped_ms[capped_ms.size() / 2], capped_ms[capped_ms.size() * 9 / 10],
              capped_ms[capped_ms.size() * 99 / 100], capped_ms.back());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: dojozero_tactics generate|fuzz|perf --flag value ...\n");
    return 2;
  }
  const std::string cmd = argv[1];
  const auto flags = parse_flags(argc, argv);
  try {
    if (cmd == "generate") return cmd_generate(flags);
    if (cmd == "fuzz") return cmd_fuzz(flags);
    if (cmd == "perf") return cmd_perf(flags);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "unknown command\n");
  return 2;
}
