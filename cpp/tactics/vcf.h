// VCFSolver (victory by continuous fours) for the side to move (ARCHITECTURE.md §3.7).
//
// Attacker moves: legal fours only (or, when the defender has a five point, a four that also
// blocks it). Defender replies are forced: any move other than the single completion point loses
// to the attacker's five, unless the defender can make a five. A black defender whose block point
// is forbidden cannot block (forbidden induction).
#pragma once

#include <cstdint>
#include <vector>

#include "tactics/threats.h"

namespace dz::tactics {

struct VcfLimits {
  uint64_t max_nodes = 200000;
  int max_depth = 40;  // attacker fours
  double max_ms = 0;   // wall-clock limit, 0 = none (keep 0 for reproducible results)
};

struct VcfResult {
  enum class Status : uint8_t { FOUND, NONE, BUDGET };
  Status status = Status::NONE;
  // Principal variation: attacker four, defender reply, ..., attacker five.
  // A defender entry of kNoPoint means "no saving reply" (block point forbidden for black).
  std::vector<Point> pv;
  uint64_t nodes = 0;
  bool depth_limited = false;  // NONE may be due to the depth limit (then it is not exhaustive)
};
const char* vcf_status_name(VcfResult::Status s);

VcfResult solve_vcf(const TPos& pos, const VcfLimits& limits = {});

}  // namespace dz::tactics
