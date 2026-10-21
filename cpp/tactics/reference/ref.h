// Frozen reference tactical solver (exact copy of the ee1d011 implementation) used as the
// differential oracle and same-binary baseline for the optimized solver (PROGRESS Phase 8 E3b).
// Shares the public types (TPos, Rules, results, limits) with dz::tactics.
#pragma once

#include "tactics/engine.h"
#include "tactics/verify.h"

namespace dz::tactics::ref {

bool is_legal_move(const Board& b, Point p, Color c, const Rules& r);
bool wins_with(const Board& b, Point p, Color c, const Rules& r);
void five_points(const Board& b, Color c, const Rules& r, std::vector<Point>& out);
void five_points_near(const Board& b, Point center, Color c, const Rules& r, std::vector<Point>& out);
void four_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out);
void double_threat_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out);
void threat_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out);

VcfResult solve_vcf(const TPos& pos, const VcfLimits& limits = {});
DfpnResult solve_dfpn(const TPos& pos, DfpnMode mode, const DfpnLimits& limits = {});
TacticalDecision analyze_tactics(const GameState& g, const TacticalLimits& limits = {});

}  // namespace dz::tactics::ref
