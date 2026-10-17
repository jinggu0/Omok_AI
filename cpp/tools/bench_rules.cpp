// Rule-engine micro benchmarks (EVALUATION.md §8). Single-threaded.
//   dojozero_bench [--positions 300] [--seed 7]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "game/game_state.h"
#include "rules/forbidden.h"
#include "tools/random_positions.h"

using namespace dz;
using Clock = std::chrono::steady_clock;

namespace {

volatile long long g_sink = 0;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Midgame positions from random play biased to moves near existing stones.
std::vector<GameState> make_positions(int n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  auto profile = make_profile(RuleProfile::renju_rif());
  std::vector<GameState> out;
  while (static_cast<int>(out.size()) < n) {
    GameState g(profile);
    const int plies = 20 + static_cast<int>(rng() % 50);
    std::vector<Action> legal;
    for (int i = 0; i < plies && !g.is_terminal(); ++i) {
      g.legal_actions(legal);
      std::vector<Action> near;
      for (const Action& a : legal) {
        if (a.type == ActionType::PLACE) {
          // prefer points within distance 2 of any stone
          bool close = g.board().stone_count() == 0;
          for (int dr = -2; dr <= 2 && !close; ++dr) {
            for (int dc = -2; dc <= 2 && !close; ++dc) {
              const int r = row_of(a.point) + dr, c = col_of(a.point) + dc;
              close = on_board(r, c) && !g.board().empty_at(make_point(r, c));
            }
          }
          if (close) near.push_back(a);
        }
      }
      const auto& pool = near.empty() ? legal : near;
      g.apply(pool[rng() % pool.size()]);
    }
    if (!g.is_terminal()) out.push_back(g);
  }
  return out;
}

struct Pct { double p50, p99, max; };
Pct percentiles(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {v[v.size() / 2], v[v.size() * 99 / 100], v.back()};
}

}  // namespace

int main(int argc, char** argv) {
  int n_positions = 300;
  uint64_t seed = 7;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--positions") n_positions = std::atoi(argv[i + 1]);
    else if (k == "--seed") seed = std::strtoull(argv[i + 1], nullptr, 10);
  }
  auto positions = make_positions(n_positions, seed);
  long long stones = 0;
  for (const auto& g : positions) stones += g.board().stone_count();
  std::printf("positions=%d avg_stones=%.1f\n", n_positions, static_cast<double>(stones) / n_positions);

  // 1. apply + undo of every legal action.
  {
    std::vector<std::vector<Action>> legal(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) legal[i] = positions[i].legal_actions();
    long long ops = 0;
    const auto t0 = Clock::now();
    for (int rep = 0; rep < 20; ++rep) {
      for (size_t i = 0; i < positions.size(); ++i) {
        GameState& g = positions[i];
        for (const Action& a : legal[i]) {
          g.apply(a);  // includes legality check and black verdict
          g.undo();
          ++ops;
        }
      }
    }
    const double s = seconds_since(t0);
    std::printf("apply+undo (GameState, incl. legality): %.2f M/s\n", ops / s / 1e6);
  }
  // 2. raw Board place/remove.
  {
    Board b;
    long long ops = 0;
    const auto t0 = Clock::now();
    for (int rep = 0; rep < 200000; ++rep) {
      for (int p = 0; p < kNumPoints; p += 7) {
        b.place(static_cast<Point>(p), (p & 1) ? Color::WHITE : Color::BLACK);
        b.remove(static_cast<Point>(p));
        ++ops;
      }
    }
    const double s = seconds_since(t0);
    g_sink += static_cast<long long>(b.hash());
    std::printf("Board place+remove: %.2f M/s\n", ops / s / 1e6);
  }
  // 3. black_verdict on every empty point of midgame positions (throughput + latency).
  auto verdict_bench = [&](const char* label, const std::vector<Board>& boards) {
    long long calls = 0, forbidden = 0;
    const auto t0 = Clock::now();
    for (int rep = 0; rep < 5; ++rep) {
      for (const Board& b : boards) {
        for (int p = 0; p < kNumPoints; ++p) {
          if (!b.empty_at(static_cast<Point>(p))) continue;
          forbidden += is_forbidden(black_verdict(b, static_cast<Point>(p))) ? 1 : 0;
          ++calls;
        }
      }
    }
    const double s = seconds_since(t0);
    std::vector<double> lat;
    for (const Board& b : boards) {
      for (int p = 0; p < kNumPoints; ++p) {
        if (!b.empty_at(static_cast<Point>(p))) continue;
        const auto t1 = Clock::now();
        g_sink += static_cast<int>(black_verdict(b, static_cast<Point>(p)));
        lat.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t1).count());
      }
    }
    const Pct pc = percentiles(lat);
    std::printf("black_verdict [%s]: %.2f M/s (forbidden %.2f%%), latency p50=%.0fns p99=%.0fns max=%.0fns\n",
                label, calls / s / 1e6, 100.0 * forbidden / calls, pc.p50, pc.p99, pc.max);
  };
  {
    std::vector<Board> boards;
    for (const auto& g : positions) boards.push_back(g.board());
    verdict_bench("midgame", boards);
    std::mt19937_64 rng(seed + 1);
    std::vector<Board> dense;
    for (int i = 0; i < n_positions; ++i) dense.push_back(tools::random_dense_board(rng));
    verdict_bench("dense-black fuzz boards", dense);
  }
  // 4. legal_actions (black to move positions exercise the forbidden checker).
  {
    long long calls = 0;
    std::vector<Action> out;
    const auto t0 = Clock::now();
    for (int rep = 0; rep < 20; ++rep) {
      for (const auto& g : positions) {
        g.legal_actions(out);
        g_sink += static_cast<long long>(out.size());
        ++calls;
      }
    }
    const double s = seconds_since(t0);
    std::printf("legal_actions (NORMAL_RENJU): %.0f k/s\n", calls / s / 1e3);
  }
  // 5. explain (debug path).
  {
    long long calls = 0;
    const auto t0 = Clock::now();
    for (const auto& g : positions) {
      for (int p = 0; p < kNumPoints; ++p) {
        if (!g.board().empty_at(static_cast<Point>(p))) continue;
        g_sink += explain_black_move(g.board(), static_cast<Point>(p)).four_count();
        ++calls;
      }
    }
    std::printf("explain_black_move: %.0f k/s\n", calls / seconds_since(t0) / 1e3);
  }
  return 0;
}
