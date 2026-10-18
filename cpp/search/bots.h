// Baseline opponents (EVALUATION.md §6): random-legal and a 1-ply heuristic bot.
#pragma once

#include <random>

#include "game/game_state.h"

namespace dz::search {

// Uniformly random engine-legal action (never a forbidden black move).
Action random_action(const GameState& g, std::mt19937_64& rng);

// 1-ply heuristic: win if possible, block an opponent five point, otherwise maximise
// own fours/threes plus blocking the opponent's fours/threes near existing stones (random
// tie-break). Opening decisions are random; opening placements prefer the centre.
Action heuristic_action(const GameState& g, std::mt19937_64& rng);

}  // namespace dz::search
