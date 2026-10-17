// Taraguchi-10 phases (RULES.md §8.3). The transition table lives in game_state.cpp; this header
// holds the per-phase static facts (actor role, stone color, region).
#pragma once

#include <optional>
#include <string_view>

#include "rules/rule_profile.h"

namespace dz {

enum class OpeningPhase : uint8_t {
  BLACK_MOVE_1,
  SWAP_AFTER_1,
  WHITE_MOVE_2,
  SWAP_AFTER_2,
  BLACK_MOVE_3,
  SWAP_AFTER_3,
  WHITE_MOVE_4,
  CHOICE_AFTER_4,
  SWAP_BEFORE_DIRECT_5,
  BLACK_DIRECT_MOVE_5,
  SWAP_AFTER_DIRECT_5,
  OFFER_TEN_5TH_MOVES,
  SELECT_ONE_OF_TEN,
  WHITE_MOVE_6,
  NORMAL_RENJU,
};
inline constexpr int kNumOpeningPhases = 15;

const char* phase_name(OpeningPhase ph);
std::optional<OpeningPhase> parse_phase(std::string_view s);

bool is_swap_phase(OpeningPhase ph);
// Phases whose action is PLACE (stone placement), including NORMAL_RENJU.
bool is_placement_phase(OpeningPhase ph);
// Color of the next stone that will be placed from this phase (for decision phases too).
Color next_stone_color(OpeningPhase ph);
// Central(N) restriction for opening placement phases; 0 = unrestricted.
int placement_region(OpeningPhase ph, const Taraguchi10Params& t);
// Actor role of an opening phase. NORMAL_RENJU: the holder of the color to move (caller decides).
Role phase_role(OpeningPhase ph, const Taraguchi10Params& t);
// Number of black / white stones on the board in this phase (opening phases only).
void phase_stone_counts(OpeningPhase ph, int& black, int& white);

}  // namespace dz
