#include "tactics/engine.h"

#include "tactics/verify.h"

#include <chrono>
#include <sstream>

namespace dz::tactics {

const char* tactic_reason_name(TacticReason r) {
  switch (r) {
    case TacticReason::IMMEDIATE_WIN: return "IMMEDIATE_WIN";
    case TacticReason::NO_DEFENSE: return "NO_DEFENSE";
    case TacticReason::FORCED_BLOCK: return "FORCED_BLOCK";
    case TacticReason::VCF: return "VCF";
    case TacticReason::VCT: return "VCT";
    case TacticReason::PROVEN_LOSS: return "PROVEN_LOSS";
    case TacticReason::NONE: return "NONE";
  }
  return "?";
}

std::string TacticalDecision::to_string() const {
  std::ostringstream os;
  os << proof_name(result) << " (" << tactic_reason_name(reason) << ")";
  if (move) os << " move=" << to_notation(*move);
  if (!pv.empty()) {
    os << " pv=";
    for (size_t i = 0; i < pv.size(); ++i) os << (i ? " " : "") << (pv[i] == kNoPoint ? "*" : to_notation(pv[i]));
  }
  os << " nodes=" << nodes << " time=" << elapsed_ms << "ms";
  return os.str();
}

TacticalDecision analyze_tactics(const GameState& g, const TacticalLimits& limits) {
  const auto t0 = std::chrono::steady_clock::now();
  TacticalDecision d;
  auto finish = [&]() {
    d.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return d;
  };
  auto pos = TPos::from_state(g);
  if (!pos) return finish();
  const Color me = pos->to_move(), opp = opposite(me);
  const Rules& rules = pos->rules();
  std::vector<Point> pts;

  // 1. Exact immediate win.
  five_points(pos->board(), me, rules, pts);
  if (!pts.empty()) {
    d.result = ProofResult::WIN;
    d.reason = TacticReason::IMMEDIATE_WIN;
    d.move = pts.front();
    d.pv = {pts.front()};
    return finish();
  }
  // 2. Forced defense.
  std::vector<Point> threats;
  five_points(pos->board(), opp, rules, threats);
  if (threats.size() >= 2 || (threats.size() == 1 && !is_legal_move(pos->board(), threats[0], me, rules))) {
    d.result = ProofResult::LOSS;
    d.reason = TacticReason::NO_DEFENSE;
    return finish();
  }
  // 3. Proven tactical win: VCF, then threat-space df-pn.
  const VcfResult vcf = solve_vcf(*pos, {limits.vcf_nodes, limits.max_depth, limits.max_ms * 0.4});
  d.nodes += vcf.nodes;
  if (vcf.status == VcfResult::Status::FOUND) {
    d.result = ProofResult::WIN;
    d.reason = TacticReason::VCF;
    d.move = vcf.pv.front();
    d.pv = vcf.pv;
    d.proof_moves = proof_from_pv(g, vcf.pv);
    return finish();
  }
  const DfpnResult vct = solve_dfpn(*pos, DfpnMode::PROVE_WIN, {limits.vct_nodes, limits.max_depth, false, limits.max_ms * 0.4});
  d.nodes += vct.nodes;
  if (vct.result == ProofResult::WIN) {
    d.result = ProofResult::WIN;
    d.reason = TacticReason::VCT;
    d.move = vct.best;
    d.pv = vct.pv;
    d.proof_moves = vct.proof_moves;
    return finish();
  }
  if (threats.size() == 1) {
    d.reason = TacticReason::FORCED_BLOCK;
    d.move = threats.front();
  }
  // 4. Proven loss (the opponent attacks, every reply of ours is searched).
  const DfpnResult loss = solve_dfpn(*pos, DfpnMode::PROVE_LOSS, {limits.loss_nodes, limits.max_depth, false, limits.max_ms * 0.2});
  d.nodes += loss.nodes;
  if (loss.result == ProofResult::LOSS) {
    d.result = ProofResult::LOSS;
    d.reason = TacticReason::PROVEN_LOSS;
    d.pv = loss.pv;
    d.proof_moves = loss.proof_moves;
  }
  return finish();
}

}  // namespace dz::tactics
