// Large-scale differential fuzzing: fast checker vs reference oracle (ARCHITECTURE.md §7).
// Also measures recursion depth of 3-3 judgement and dumps deep false-forbidden candidates for
// human review.
//
//   dojozero_fuzz --comparisons 10000000 --seed 1 --threads 8 --candidates out.jsonl
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "rules/forbidden.h"
#include "rules/reference_rules.h"
#include "tools/random_positions.h"

using namespace dz;

namespace {

struct Shared {
  std::atomic<long long> comparisons{0};
  std::atomic<long long> boards{0};
  std::atomic<long long> mismatches{0};
  std::atomic<long long> apparent33{0};
  std::atomic<long long> false33{0};
  std::array<std::atomic<long long>, 16> depth_hist{};  // max recursion depth at apparent 3-3 points
  std::mutex out_mu;
  std::vector<nlohmann::json> candidates;
  std::array<int, 16> per_depth_saved{};
};

nlohmann::json describe(const Board& b, Point q, const ForbiddenInfo& info, int depth) {
  nlohmann::json j;
  std::vector<std::string> black, white;
  for (int p = 0; p < kNumPoints; ++p) {
    if (b.at(static_cast<Point>(p)) == BLACK_STONE) black.push_back(to_notation(static_cast<Point>(p)));
    if (b.at(static_cast<Point>(p)) == WHITE_STONE) white.push_back(to_notation(static_cast<Point>(p)));
  }
  j["position"] = {{"black", black}, {"white", white}};
  j["query"] = to_notation(q);
  j["verdict"] = verdict_name(info.verdict);
  j["fours"] = info.four_count();
  j["apparent_threes"] = info.apparent_three_count();
  j["real_threes"] = info.real_three_count();
  j["max_depth"] = depth;
  j["explain"] = info.to_string();
  return j;
}

void worker(uint64_t seed, long long target, Shared& sh) {
  std::mt19937_64 rng(seed);
  while (sh.comparisons.load() < target) {
    const Board b = tools::random_dense_board(rng);
    sh.boards.fetch_add(1);
    for (int p = 0; p < kNumPoints; ++p) {
      const Point q = static_cast<Point>(p);
      if (!b.empty_at(q) || !tools::near_black(b, q)) continue;
      const MoveVerdict fast = black_verdict(b, q);
      const MoveVerdict ref = ref::black_verdict(b, q);
      const ForbiddenInfo info = explain_black_move(b, q);
      sh.comparisons.fetch_add(1);
      if (fast != ref || info.verdict != fast) {
        if (sh.mismatches.fetch_add(1) < 5) {
          std::lock_guard<std::mutex> lock(sh.out_mu);
          std::printf("MISMATCH seed=%llu point=%s fast=%s ref=%s\n%s%s\n",
                      static_cast<unsigned long long>(seed), to_notation(q).c_str(), verdict_name(fast),
                      verdict_name(ref), b.to_string().c_str(), info.to_string().c_str());
        }
        continue;
      }
      if (info.apparent_three_count() >= 2 && !info.five_dirs.size() && !info.overline_dirs.size() &&
          info.four_count() < 2) {
        sh.apparent33.fetch_add(1);
        int depth = 0;
        for (const auto& s : info.recursion) depth = std::max(depth, s.depth);
        depth = std::min(depth, 15);
        sh.depth_hist[depth].fetch_add(1);
        const bool is_false = info.verdict != MoveVerdict::FORBIDDEN_DOUBLE_THREE;
        if (is_false) sh.false33.fetch_add(1);
        // Keep false 3-3 examples whose outcome needed recursion depth >= 2.
        if (is_false && depth >= 2) {
          std::lock_guard<std::mutex> lock(sh.out_mu);
          if (sh.per_depth_saved[depth] < 40) {
            ++sh.per_depth_saved[depth];
            sh.candidates.push_back(describe(b, q, info, depth));
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  long long target = 1000000;
  uint64_t seed = 1;
  int threads = static_cast<int>(std::thread::hardware_concurrency());
  std::string candidates_path;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--comparisons") target = std::atoll(v.c_str());
    else if (k == "--seed") seed = std::strtoull(v.c_str(), nullptr, 10);
    else if (k == "--threads") threads = std::atoi(v.c_str());
    else if (k == "--candidates") candidates_path = v;
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  if (threads < 1) threads = 1;
  Shared sh;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker, seed * 1000003ULL + t, target, std::ref(sh));
  for (auto& th : pool) th.join();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("seed=%llu threads=%d boards=%lld comparisons=%lld mismatches=%lld time=%.1fs\n",
              static_cast<unsigned long long>(seed), threads, sh.boards.load(), sh.comparisons.load(),
              sh.mismatches.load(), secs);
  std::printf("apparent 3-3 points=%lld, of which false (legal)=%lld\n", sh.apparent33.load(), sh.false33.load());
  std::printf("max recursion depth histogram at apparent 3-3 points:\n");
  for (int d = 0; d < 16; ++d) {
    if (sh.depth_hist[d].load()) std::printf("  depth %2d: %lld\n", d, sh.depth_hist[d].load());
  }
  if (!candidates_path.empty()) {
    std::ofstream out(candidates_path);
    for (const auto& c : sh.candidates) out << c.dump() << "\n";
    std::printf("wrote %zu candidates to %s\n", sh.candidates.size(), candidates_path.c_str());
  }
  return sh.mismatches.load() == 0 ? 0 : 1;
}
