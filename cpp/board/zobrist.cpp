#include "board/zobrist.h"

#include <array>

namespace dz::zobrist {
namespace {

uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

struct Keys {
  std::array<std::array<uint64_t, kNumPoints>, 2> stone{};
  std::array<uint64_t, kNumPoints> offered{};
  std::array<uint64_t, kMaxPhases> phase{};
  uint64_t white_to_move = 0;

  Keys() {
    uint64_t s = kSeed;
    for (auto& per_color : stone) {
      for (auto& k : per_color) k = splitmix64(s);
    }
    for (auto& k : offered) k = splitmix64(s);
    for (auto& k : phase) k = splitmix64(s);
    white_to_move = splitmix64(s);
  }
};

const Keys& keys() {
  static const Keys k;
  return k;
}

}  // namespace

uint64_t stone(int color, Point p) { return keys().stone[color][p]; }
uint64_t offered(Point p) { return keys().offered[p]; }
uint64_t phase(int phase_id) { return keys().phase[phase_id]; }
uint64_t side_to_move_white() { return keys().white_to_move; }

}  // namespace dz::zobrist
