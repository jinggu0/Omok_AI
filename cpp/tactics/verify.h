// Independent full-width verification of a claimed forced win (EVALUATION.md §3).
//
// Uses ONLY GameState (apply / legal_actions / outcome) for rule decisions, not the solvers'
// threat code: a win counts only if GameState reports it after apply(), and at every defender node
// every legal defender action (including pass) from GameState::legal_actions() is tried.
// At attacker nodes the move comes from the proof (position key -> move); if the proof has no
// entry, an immediate win or a double threat (win in 2) is searched. Candidate attacker moves may
// be *suggested* by hints (e.g. the threat detector) to keep this fast; hints only order/propose
// moves, every suggested move is still judged by GameState, and the double-threat search falls back
// to all legal moves when no hint works. A false hint can therefore only cause REFUTED/TOO_LARGE,
// never a false VERIFIED.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

#include "game/game_state.h"

namespace dz::tactics {

uint64_t position_key(const Board& b, Color to_move);  // same key as TPos::key()

enum class VerifyStatus : uint8_t { VERIFIED, REFUTED, TOO_LARGE };
const char* verify_status_name(VerifyStatus s);

struct VerifyLimits {
  int max_depth = 80;
  uint64_t max_nodes = 3000000;  // GameState::apply calls
};

using ProofMoveFn = std::function<std::optional<Point>(uint64_t key)>;
using HintFn = std::function<std::vector<Point>(const GameState&)>;

struct VerifyHints {
  HintFn immediate_wins;  // candidate winning moves of the side to move
  HintFn double_threats;  // candidate double-threat moves of the side to move
};
// Hints from the tactical threat detector (five points / double-threat moves).
VerifyHints threat_detector_hints();

// The side to move in `g` (NORMAL_RENJU) is the attacker.
VerifyStatus verify_forced_win(const GameState& g, const ProofMoveFn& proof_move, const VerifyLimits& limits = {},
                               const VerifyHints& hints = threat_detector_hints());
// Same, but the side to move is the defender (verifies a LOSS claim).
VerifyStatus verify_forced_loss(const GameState& g, const ProofMoveFn& proof_move, const VerifyLimits& limits = {},
                                const VerifyHints& hints = threat_detector_hints());

// Proof map for a VCF principal variation (attacker moves keyed by the position before them).
std::unordered_map<uint64_t, Point> proof_from_pv(const GameState& g, const std::vector<Point>& pv);

}  // namespace dz::tactics
