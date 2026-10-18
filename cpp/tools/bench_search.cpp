// MCTS tree cost without a network (EVALUATION.md §8.2 "MCTS nodes/s, uniform evaluator").
//   dojozero_bench_search [--positions 100 --sims 800 --seed 3]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include "search/search.h"
#include "tools/random_positions.h"

using namespace dz;
using namespace dz::search;

int main(int argc, char** argv) {
  std::map<std::string, std::string> f;
  for (int i = 1; i + 1 < argc; i += 2) f[std::string(argv[i]).substr(2)] = argv[i + 1];
  const int n = f.count("positions") ? std::atoi(f["positions"].c_str()) : 100;
  const int sims = f.count("sims") ? std::atoi(f["sims"].c_str()) : 800;
  std::mt19937_64 rng(f.count("seed") ? std::strtoull(f["seed"].c_str(), nullptr, 10) : 3);
  std::vector<GameState> states;
  while (static_cast<int>(states.size()) < n) {
    GameState g = tools::random_midgame(rng, make_profile(RuleProfile::renju_rif()), 10, 50);
    if (!g.is_terminal()) states.push_back(g);
  }
  for (int variant = 0; variant < 4; ++variant) {
    SearchConfig c;
    c.simulations = sims;
    c.root_tactics_nodes = 0;
    c.max_inflight = 1;
    c.leaf_tactics = variant % 2 == 0;
    c.root = variant < 2 ? RootMode::PUCT : RootMode::GUMBEL;
    UniformEvaluator u;
    long long total = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& g : states) total += Tree(g, c).run(u).simulations;
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s leaf_tactics=%d: %.0f simulations/s (%lld sims, %d positions)\n",
                c.root == RootMode::PUCT ? "PUCT  " : "GUMBEL", c.leaf_tactics ? 1 : 0, total / s, total, n);
  }
  return 0;
}
