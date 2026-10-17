// Reference oracle for black move verdicts (ARCHITECTURE.md §3.4).
// Implements RULES.md §3-§4 as literally as possible, with a deliberately different formulation
// from the fast checker (5-cell windows on a plain 2D copy, no shared helpers, no caching).
// Test/fuzz use only: never call from engine hot paths.
#pragma once

#include "board/board.h"
#include "rules/forbidden.h"

namespace dz::ref {

MoveVerdict black_verdict(const Board& board, Point p, const BlackRuleOptions& opt = {});

// Counts used by tests (computed on board + black stone at p).
int count_black_fours(const Board& board, Point p);
int count_black_real_threes(const Board& board, Point p, const BlackRuleOptions& opt = {});
int count_black_apparent_threes(const Board& board, Point p);

}  // namespace dz::ref
