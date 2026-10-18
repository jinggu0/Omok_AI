// Tactical decision layer (PRD FR-08): immediate win -> forced defense -> proven tactical
// result. Neural MCTS (Phase 5) consults this first and falls back when the result is UNKNOWN.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "game/game_state.h"
#include "tactics/dfpn.h"
#include "tactics/vcf.h"

namespace dz::tactics {

enum class TacticReason : uint8_t {
  IMMEDIATE_WIN,   // a five is available
  NO_DEFENSE,      // opponent has two five points, or its only five point is forbidden for us
  FORCED_BLOCK,    // opponent has one five point: the block is the only non-losing move
  VCF,             // proven win by continuous fours
  VCT,             // proven win by threat-space proof-number search
  PROVEN_LOSS,     // every move loses by force (df-pn, opponent attacking)
  NONE,
};
const char* tactic_reason_name(TacticReason r);

struct TacticalLimits {
  uint64_t vcf_nodes = 100000;
  uint64_t vct_nodes = 50000;
  uint64_t loss_nodes = 20000;
  int max_depth = 40;
  double max_ms = 0;  // total wall-clock limit (split 40/40/20 over VCF/VCT/loss); 0 = node budgets only
};

struct TacticalDecision {
  ProofResult result = ProofResult::UNKNOWN;  // for the side to move
  TacticReason reason = TacticReason::NONE;
  std::optional<Point> move;  // proven / forced move when known
  std::vector<Point> pv;
  // Attacker move of every proven attacker node (position key -> move), for verification.
  std::unordered_map<uint64_t, Point> proof_moves;
  uint64_t nodes = 0;
  double elapsed_ms = 0;
  std::string to_string() const;
};

// Requires NORMAL_RENJU (otherwise returns UNKNOWN / NONE).
TacticalDecision analyze_tactics(const GameState& g, const TacticalLimits& limits = {});

}  // namespace dz::tactics
