// ProofNumberSolver: depth-first proof-number search (df-pn) over a threat-restricted game tree
// (ARCHITECTURE.md §3.7). Used as the VCT / threat-space solver.
//
// Soundness argument (why WIN/LOSS claims are proofs):
//  * OR nodes (attacker to move) only need one winning move, so restricting them to threats
//    (fours, threes, forced blocks) can only lose completeness, never soundness.
//  * AND nodes (defender to move) consider EVERY legal defender move, including pass when the
//    profile allows it. A move is skipped only when it is checked, on the actual board after that
//    move, to lose by force: the attacker then has a legal move creating two five points while the
//    defender has no five point of its own. Facing an attacker four, the only non-losing replies
//    are a defender five or the block, so only the block is searched (an illegal black block means
//    the defender loses: forbidden induction).
//  * Depth limits and budget exhaustion only ever produce "not proven" (UNKNOWN), never a proof.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tactics/threats.h"

namespace dz::tactics {

enum class DfpnMode : uint8_t {
  PROVE_WIN,   // side to move is the attacker
  PROVE_LOSS,  // the opponent of the side to move is the attacker
};

struct DfpnLimits {
  uint64_t max_nodes = 200000;
  int max_depth = 40;       // plies from the root
  bool fours_only = false;  // restrict attacker to fours (proof-number VCF)
  double max_ms = 0;        // wall-clock limit, 0 = none (keep 0 for reproducible results)
};

struct DfpnResult {
  ProofResult result = ProofResult::UNKNOWN;
  Point best = kNoPoint;     // proven first move (PROVE_WIN), kNoPoint otherwise
  std::vector<Point> pv;     // one proven line; kNoPoint inside = defender pass
  uint64_t nodes = 0;
  bool budget_exhausted = false;
  // Winning move of every proven attacker node (position key -> move), for independent verification.
  std::unordered_map<uint64_t, Point> proof_moves;
};

DfpnResult solve_dfpn(const TPos& pos, DfpnMode mode, const DfpnLimits& limits = {});

}  // namespace dz::tactics
