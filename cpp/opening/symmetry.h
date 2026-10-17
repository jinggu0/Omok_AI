// Ten-fifth-proposal symmetry and validation (RULES.md §10). Single implementation for C++ and
// (later) Python bindings.
#pragma once

#include <span>
#include <string>
#include <vector>

#include "board/board.h"
#include "rules/rule_profile.h"

namespace dz {

// Bitmask of transforms t with t(board) == board (colors preserved). Bit 0 is always set.
uint8_t stabilizer_mask(const Board& board);

// E1 (POSITION): board + black q1 and board + black q2 are the same position up to D4.
// E3 (POINTWISE_D4): some D4 transform maps q1 to q2.
bool proposals_equivalent(const Board& board, Point q1, Point q2, SymmetryMode mode);

// E2: q1, q2 in the same orbit of the stabilizer of the current position (used to test E1 == E2).
bool proposals_stabilizer_equivalent(const Board& board, Point q1, Point q2);

// Smallest point index in the equivalence class of empty point q.
Point canonical_proposal(const Board& board, Point q, SymmetryMode mode);

// Is p an admissible next proposal given already-offered points (partial validation,
// RULES.md §10.1 items 1-3 and 5, plus feasibility of filling the remaining slots)?
bool proposal_admissible(const Board& board, std::span<const Point> offered, Point p,
                         const Taraguchi10Params& params, const BlackRuleOptions& black_opt,
                         std::string* why = nullptr);

// All admissible next proposals (same rules as proposal_admissible, computed in one pass).
std::vector<Point> admissible_proposals(const Board& board, std::span<const Point> offered,
                                        const Taraguchi10Params& params,
                                        const BlackRuleOptions& black_opt);

// Full validation of a complete proposal set.
bool validate_proposals(const Board& board, std::span<const Point> proposals,
                        const Taraguchi10Params& params, const BlackRuleOptions& black_opt,
                        std::string* why = nullptr);

}  // namespace dz
