// Action variant (RULES.md §8.3, PRD §10): opening decisions are NOT encoded as flat 225 moves.
#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "board/coord.h"
#include "opening/opening_phase.h"

namespace dz {

enum class ActionType : uint8_t {
  PLACE,
  PASS,
  SWAP,
  KEEP_COLOR,
  CHOOSE_DIRECT_FIFTH,
  CHOOSE_TEN_PROPOSALS,
  PROPOSE_FIFTH,
  SELECT_PROPOSED_FIFTH,
};

struct Action {
  ActionType type = ActionType::PLACE;
  Point point = kNoPoint;  // PLACE / PROPOSE_FIFTH / SELECT_PROPOSED_FIFTH only

  static Action place(Point p) { return {ActionType::PLACE, p}; }
  static Action pass() { return {ActionType::PASS, kNoPoint}; }
  static Action swap() { return {ActionType::SWAP, kNoPoint}; }
  static Action keep() { return {ActionType::KEEP_COLOR, kNoPoint}; }
  static Action direct_fifth() { return {ActionType::CHOOSE_DIRECT_FIFTH, kNoPoint}; }
  static Action ten_proposals() { return {ActionType::CHOOSE_TEN_PROPOSALS, kNoPoint}; }
  static Action propose(Point p) { return {ActionType::PROPOSE_FIFTH, p}; }
  static Action select(Point p) { return {ActionType::SELECT_PROPOSED_FIFTH, p}; }

  bool has_point() const {
    return type == ActionType::PLACE || type == ActionType::PROPOSE_FIFTH ||
           type == ActionType::SELECT_PROPOSED_FIFTH;
  }
  bool operator==(const Action&) const = default;
};

// "H8", "pass", "swap", "keep", "direct5", "offer10", "propose:H8", "select:H8".
std::string to_string(const Action& a);
std::optional<Action> parse_action(std::string_view s);
// Like parse_action, but a bare point means PLACE / PROPOSE_FIFTH / SELECT_PROPOSED_FIFTH
// depending on the phase.
std::optional<Action> parse_action_in_phase(std::string_view s, OpeningPhase phase);

}  // namespace dz
