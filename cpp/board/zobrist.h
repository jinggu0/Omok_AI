// Zobrist keys, generated deterministically from a fixed seed (reproducibility).
#pragma once

#include <cstdint>

#include "board/coord.h"

namespace dz::zobrist {

inline constexpr uint64_t kSeed = 0x446F6A6F5A65726FULL;  // "DojoZero"
inline constexpr int kMaxPhases = 32;

uint64_t stone(int color, Point p);   // color: 0 = black, 1 = white
uint64_t offered(Point p);            // proposed fifth move (opening)
uint64_t phase(int phase_id);
uint64_t side_to_move_white();

}  // namespace dz::zobrist
