// dojozero_datagen: Stage A tactical dataset (TRAINING.md §2.1).
//
//   dojozero_datagen --out data/stage_a --samples 300000 --seed 1 --threads 8
//                    [--vcf-nodes 20000 --vct-nodes 8000 --loss-nodes 4000 --git <commit>]
//
// Positions: random NORMAL_RENJU midgames (RENJU_RIF and OMOKDOJANG_DAN alternating), deduplicated
// by canonical D4 hash. Labels come only from exact solvers (analyze_tactics):
//   policy  : IMMEDIATE_WIN -> every five point; FORCED_BLOCK -> the block; VCF/VCT -> proven move
//   value   : +1 proven WIN, -1 proven LOSS, otherwise masked (UNKNOWN is never labelled a loss)
// Features and auxiliary targets are NOT stored: they are recomputed by the C++ encoder at load
// time so a rule or feature change can never leave stale targets behind.
//
// Output (NumPy .npy, C order): boards int8[N,225] (0 empty, 1 black, 2 white), sides uint8[N]
// (0 black to move), profiles uint8[N] (0 RENJU_RIF, 1 OMOKDOJANG_DAN), policy uint8[N,226]
// (target set), has_policy uint8[N], value int8[N], has_value uint8[N], reason uint8[N]
// (TacticReason), plies uint16[N]; manifest.json.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "features/features.h"
#include "tactics/engine.h"
#include "tools/random_positions.h"

using namespace dz;
using namespace dz::tactics;
namespace fs = std::filesystem;

namespace {

struct Sample {
  std::array<int8_t, kNumPoints> board{};
  uint8_t side = 0, profile = 0;
  std::array<uint8_t, features::kNumPolicyActions> policy{};
  uint8_t has_policy = 0;
  int8_t value = 0;
  uint8_t has_value = 0, reason = 0;
  uint16_t plies = 0;
};

// Minimal .npy (format 1.0) writer.
void write_npy(const fs::path& path, const char* descr, const std::vector<size_t>& shape, const void* data,
               size_t bytes) {
  std::string shape_s = "(";
  for (size_t i = 0; i < shape.size(); ++i) shape_s += std::to_string(shape[i]) + (shape.size() == 1 ? "," : i + 1 < shape.size() ? ", " : "");
  shape_s += ")";
  std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape_s + ", }";
  const size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header += '\n';
  std::ofstream out(path, std::ios::binary);
  out.write("\x93NUMPY\x01\x00", 8);
  const uint16_t len = static_cast<uint16_t>(header.size());
  out.write(reinterpret_cast<const char*>(&len), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!out) throw std::runtime_error("write failed: " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> f;
  for (int i = 1; i + 1 < argc; i += 2) f[std::string(argv[i]).substr(2)] = argv[i + 1];
  if (!f.count("out") || !f.count("samples")) {
    std::fprintf(stderr, "usage: dojozero_datagen --out DIR --samples N [--seed S --threads T ...]\n");
    return 2;
  }
  const fs::path out_dir = f.at("out");
  const size_t target = std::strtoull(f.at("samples").c_str(), nullptr, 10);
  const uint64_t seed = f.count("seed") ? std::strtoull(f.at("seed").c_str(), nullptr, 10) : 1;
  const int threads = f.count("threads") ? std::atoi(f.at("threads").c_str()) : 8;
  const TacticalLimits limits{f.count("vcf-nodes") ? std::strtoull(f.at("vcf-nodes").c_str(), nullptr, 10) : 20000,
                              f.count("vct-nodes") ? std::strtoull(f.at("vct-nodes").c_str(), nullptr, 10) : 8000,
                              f.count("loss-nodes") ? std::strtoull(f.at("loss-nodes").c_str(), nullptr, 10) : 4000,
                              40, 0};
  fs::create_directories(out_dir);

  const ProfilePtr profiles[2] = {make_profile(RuleProfile::renju_rif()), make_profile(RuleProfile::omokdojang_dan())};
  std::vector<Sample> samples;
  samples.reserve(target);
  std::unordered_set<uint64_t> seen;
  std::mutex mu;
  std::atomic<long long> games{0}, duplicates{0};
  const auto t0 = std::chrono::steady_clock::now();

  auto worker = [&](int tid) {
    std::mt19937_64 rng(seed * 1000003ULL + static_cast<uint64_t>(tid));
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mu);
        if (samples.size() >= target) return;
      }
      const uint8_t prof = static_cast<uint8_t>(rng() % 2);
      GameState g = tools::random_midgame(rng, profiles[prof], 6, 90);
      games++;
      if (g.phase() != OpeningPhase::NORMAL_RENJU || g.is_terminal()) continue;
      const uint64_t key = g.canonical_hash();
      {
        std::lock_guard<std::mutex> lock(mu);
        if (!seen.insert(key).second) { duplicates++; continue; }
      }
      const TacticalDecision d = analyze_tactics(g, limits);
      Sample s;
      for (int p = 0; p < kNumPoints; ++p) {
        const Cell c = g.board().at(static_cast<Point>(p));
        s.board[p] = c == BLACK_STONE ? 1 : c == WHITE_STONE ? 2 : 0;
      }
      s.side = g.side_to_move() == Color::BLACK ? 0 : 1;
      s.profile = prof;
      s.plies = static_cast<uint16_t>(g.board().stone_count());
      s.reason = static_cast<uint8_t>(d.reason);
      if (d.result == ProofResult::WIN) { s.value = 1; s.has_value = 1; }
      if (d.result == ProofResult::LOSS) { s.value = -1; s.has_value = 1; }
      if (d.reason == TacticReason::IMMEDIATE_WIN) {
        std::vector<Point> five;
        five_points(g.board(), g.side_to_move(), Rules{g.profile().black, g.profile().pass_allowed}, five);
        for (Point p : five) s.policy[p] = 1;
        s.has_policy = 1;
      } else if ((d.reason == TacticReason::VCF || d.reason == TacticReason::VCT ||
                  d.reason == TacticReason::FORCED_BLOCK) && d.move) {
        s.policy[*d.move] = 1;
        s.has_policy = 1;
      }
      std::lock_guard<std::mutex> lock(mu);
      if (samples.size() < target) samples.push_back(s);
      if (samples.size() % 20000 == 0) {
        std::printf("%zu samples (%lld games)\n", samples.size(), games.load());
        std::fflush(stdout);
      }
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();
  // Deterministic order regardless of thread scheduling: sort by board content.
  std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) { return a.board < b.board || (a.board == b.board && a.side < b.side); });
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const size_t n = samples.size();
  std::vector<int8_t> boards(n * kNumPoints), value(n);
  std::vector<uint8_t> sides(n), profs(n), policy(n * features::kNumPolicyActions), has_policy(n), has_value(n), reason(n);
  std::vector<uint16_t> plies(n);
  std::map<std::string, long long> reason_counts;
  long long wins = 0, losses = 0, with_policy = 0;
  for (size_t i = 0; i < n; ++i) {
    const Sample& s = samples[i];
    std::copy(s.board.begin(), s.board.end(), boards.begin() + static_cast<long long>(i * kNumPoints));
    std::copy(s.policy.begin(), s.policy.end(), policy.begin() + static_cast<long long>(i * features::kNumPolicyActions));
    sides[i] = s.side; profs[i] = s.profile; has_policy[i] = s.has_policy; value[i] = s.value;
    has_value[i] = s.has_value; reason[i] = s.reason; plies[i] = s.plies;
    reason_counts[tactic_reason_name(static_cast<TacticReason>(s.reason))]++;
    wins += s.has_value && s.value > 0;
    losses += s.has_value && s.value < 0;
    with_policy += s.has_policy;
  }
  write_npy(out_dir / "boards.npy", "|i1", {n, static_cast<size_t>(kNumPoints)}, boards.data(), boards.size());
  write_npy(out_dir / "sides.npy", "|u1", {n}, sides.data(), n);
  write_npy(out_dir / "profiles.npy", "|u1", {n}, profs.data(), n);
  write_npy(out_dir / "policy.npy", "|u1", {n, static_cast<size_t>(features::kNumPolicyActions)}, policy.data(), policy.size());
  write_npy(out_dir / "has_policy.npy", "|u1", {n}, has_policy.data(), n);
  write_npy(out_dir / "value.npy", "|i1", {n}, value.data(), n);
  write_npy(out_dir / "has_value.npy", "|u1", {n}, has_value.data(), n);
  write_npy(out_dir / "reason.npy", "|u1", {n}, reason.data(), n);
  write_npy(out_dir / "plies.npy", "<u2", {n}, plies.data(), n * 2);

  nlohmann::json m;
  m["format_version"] = 1;
  m["generator"] = "dojozero_datagen";
  m["samples"] = n;
  m["seed"] = seed;
  m["threads"] = threads;
  m["games"] = games.load();
  m["duplicates_skipped"] = duplicates.load();
  m["limits"] = {{"vcf_nodes", limits.vcf_nodes}, {"vct_nodes", limits.vct_nodes}, {"loss_nodes", limits.loss_nodes}};
  m["profiles"] = {{"0", "RENJU_RIF"}, {"1", "OMOKDOJANG_DAN"}};
  m["profile_hashes"] = {{"RENJU_RIF", profiles[0]->hash()}, {"OMOKDOJANG_DAN", profiles[1]->hash()}};
  m["label_counts"] = {{"value_win", wins}, {"value_loss", losses}, {"with_policy", with_policy}, {"reasons", reason_counts}};
  m["git_commit"] = f.count("git") ? f.at("git") : "unknown";
  m["seconds"] = secs;
  m["source"] = {{"kind", "self-generated (exact solver labels)"}, {"license", "project-owned"}};
  std::ofstream(out_dir / "manifest.json") << m.dump(2) << "\n";
  std::printf("wrote %zu samples to %s in %.0fs (win %lld, loss %lld, policy %lld)\n", n, out_dir.string().c_str(),
              secs, wins, losses, with_policy);
  return 0;
}
