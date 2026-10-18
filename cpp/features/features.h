// NN input planes and auxiliary targets (ARCHITECTURE.md §3.8, TRAINING.md §4).
// Single implementation used by training (via bindings) and inference. All planes are binary.
//
// Layout: planes[c][row][col], row 0 = rank 1, col 0 = column A (RULES.md §1.2), index = row*15+col.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "game/game_state.h"

namespace dz::features {

struct FeatureConfig {
  int history_planes = 0;         // previous K placements before the last move, one plane each
  bool tactical_planes = false;   // ablation E: rule-derived threat planes for both sides
  bool forbidden_plane = true;    // black forbidden points as an input (rule information)
  int version = 1;                // bump when plane semantics change

  int num_planes() const;
  std::vector<std::string> plane_names() const;
  // FNV-1a of the plane name list + version; stored in checkpoints and model manifests.
  uint64_t hash() const;
};

// Policy action indices: 0..224 = points, 225 = pass.
inline constexpr int kNumPolicyActions = kNumPoints + 1;
inline constexpr int kPassIndex = kNumPoints;

// out must hold num_planes()*225 bytes; planes are 0/1.
void encode(const GameState& g, const FeatureConfig& cfg, uint8_t* out);

// Legal point/pass actions of the current phase in policy-index space (PLACE, PROPOSE_FIFTH,
// SELECT_PROPOSED_FIFTH points; pass). Decision phases (swap/branch) have no point actions.
void legal_policy_mask(const GameState& g, uint8_t* out /* 226 */);

// Auxiliary targets for the side to move (NORMAL_RENJU semantics; zero planes otherwise):
//   0 immediate_win   : points where the actor wins at once
//   1 forced_block    : opponent five points (the actor must block)
//   2 four_threat     : actor moves creating a four
//   3 three_threat    : actor moves creating a three (black: real three)
//   4 forbidden       : points forbidden for black on the current board
inline constexpr int kNumAuxPlanes = 5;
void aux_targets(const GameState& g, uint8_t* out /* 5*225 */);

}  // namespace dz::features
