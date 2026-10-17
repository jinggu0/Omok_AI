#include "game/action.h"

namespace dz {

std::string to_string(const Action& a) {
  switch (a.type) {
    case ActionType::PLACE: return to_notation(a.point);
    case ActionType::PASS: return "pass";
    case ActionType::SWAP: return "swap";
    case ActionType::KEEP_COLOR: return "keep";
    case ActionType::CHOOSE_DIRECT_FIFTH: return "direct5";
    case ActionType::CHOOSE_TEN_PROPOSALS: return "offer10";
    case ActionType::PROPOSE_FIFTH: return "propose:" + to_notation(a.point);
    case ActionType::SELECT_PROPOSED_FIFTH: return "select:" + to_notation(a.point);
  }
  return "?";
}

std::optional<Action> parse_action(std::string_view s) {
  if (s == "pass") return Action::pass();
  if (s == "swap") return Action::swap();
  if (s == "keep") return Action::keep();
  if (s == "direct5") return Action::direct_fifth();
  if (s == "offer10") return Action::ten_proposals();
  auto with_point = [&](std::string_view prefix, ActionType type) -> std::optional<Action> {
    if (s.substr(0, prefix.size()) != prefix) return std::nullopt;
    auto p = parse_point(s.substr(prefix.size()));
    if (!p) return std::nullopt;
    return Action{type, *p};
  };
  if (auto a = with_point("propose:", ActionType::PROPOSE_FIFTH)) return a;
  if (auto a = with_point("select:", ActionType::SELECT_PROPOSED_FIFTH)) return a;
  if (auto p = parse_point(s)) return Action::place(*p);
  return std::nullopt;
}

std::optional<Action> parse_action_in_phase(std::string_view s, OpeningPhase phase) {
  auto a = parse_action(s);
  if (!a || a->type != ActionType::PLACE) return a;
  if (phase == OpeningPhase::OFFER_TEN_5TH_MOVES) return Action::propose(a->point);
  if (phase == OpeningPhase::SELECT_ONE_OF_TEN) return Action::select(a->point);
  return a;
}

}  // namespace dz
