#include "opening/opening_phase.h"

namespace dz {

namespace {
constexpr const char* kPhaseNames[kNumOpeningPhases] = {
    "BLACK_MOVE_1",         "SWAP_AFTER_1",        "WHITE_MOVE_2",        "SWAP_AFTER_2",
    "BLACK_MOVE_3",         "SWAP_AFTER_3",        "WHITE_MOVE_4",        "CHOICE_AFTER_4",
    "SWAP_BEFORE_DIRECT_5", "BLACK_DIRECT_MOVE_5", "SWAP_AFTER_DIRECT_5", "OFFER_TEN_5TH_MOVES",
    "SELECT_ONE_OF_TEN",    "WHITE_MOVE_6",        "NORMAL_RENJU"};
}  // namespace

const char* phase_name(OpeningPhase ph) { return kPhaseNames[static_cast<int>(ph)]; }

std::optional<OpeningPhase> parse_phase(std::string_view s) {
  for (int i = 0; i < kNumOpeningPhases; ++i) {
    if (s == kPhaseNames[i]) return static_cast<OpeningPhase>(i);
  }
  return std::nullopt;
}

bool is_swap_phase(OpeningPhase ph) {
  switch (ph) {
    case OpeningPhase::SWAP_AFTER_1:
    case OpeningPhase::SWAP_AFTER_2:
    case OpeningPhase::SWAP_AFTER_3:
    case OpeningPhase::SWAP_BEFORE_DIRECT_5:
    case OpeningPhase::SWAP_AFTER_DIRECT_5:
      return true;
    default:
      return false;
  }
}

bool is_placement_phase(OpeningPhase ph) {
  switch (ph) {
    case OpeningPhase::BLACK_MOVE_1:
    case OpeningPhase::WHITE_MOVE_2:
    case OpeningPhase::BLACK_MOVE_3:
    case OpeningPhase::WHITE_MOVE_4:
    case OpeningPhase::BLACK_DIRECT_MOVE_5:
    case OpeningPhase::WHITE_MOVE_6:
    case OpeningPhase::NORMAL_RENJU:
      return true;
    default:
      return false;
  }
}

Color next_stone_color(OpeningPhase ph) {
  switch (ph) {
    case OpeningPhase::SWAP_AFTER_1:
    case OpeningPhase::WHITE_MOVE_2:
    case OpeningPhase::SWAP_AFTER_3:
    case OpeningPhase::WHITE_MOVE_4:
    case OpeningPhase::SWAP_AFTER_DIRECT_5:
    case OpeningPhase::WHITE_MOVE_6:
      return Color::WHITE;
    default:
      return Color::BLACK;  // NORMAL_RENJU is handled by GameState (side to move)
  }
}

int placement_region(OpeningPhase ph, const Taraguchi10Params& t) {
  switch (ph) {
    case OpeningPhase::BLACK_MOVE_1: return t.region[0];
    case OpeningPhase::WHITE_MOVE_2: return t.region[1];
    case OpeningPhase::BLACK_MOVE_3: return t.region[2];
    case OpeningPhase::WHITE_MOVE_4: return t.region[3];
    case OpeningPhase::BLACK_DIRECT_MOVE_5: return t.region[4];
    default: return 0;
  }
}

Role phase_role(OpeningPhase ph, const Taraguchi10Params& t) {
  switch (ph) {
    // After move k by color C, the holder of the other color decides the swap (RULES.md §9.3).
    case OpeningPhase::SWAP_AFTER_1: return Role::WHITE_HOLDER;
    case OpeningPhase::SWAP_AFTER_2: return Role::BLACK_HOLDER;
    case OpeningPhase::SWAP_AFTER_3: return Role::WHITE_HOLDER;
    case OpeningPhase::CHOICE_AFTER_4: return t.choice_actor;
    case OpeningPhase::SWAP_BEFORE_DIRECT_5: return t.swap_before_direct5_actor;
    case OpeningPhase::SWAP_AFTER_DIRECT_5: return Role::WHITE_HOLDER;
    case OpeningPhase::OFFER_TEN_5TH_MOVES: return Role::BLACK_HOLDER;
    case OpeningPhase::SELECT_ONE_OF_TEN: return Role::WHITE_HOLDER;
    default:
      return next_stone_color(ph) == Color::BLACK ? Role::BLACK_HOLDER : Role::WHITE_HOLDER;
  }
}

void phase_stone_counts(OpeningPhase ph, int& black, int& white) {
  switch (ph) {
    case OpeningPhase::BLACK_MOVE_1: black = 0; white = 0; return;
    case OpeningPhase::SWAP_AFTER_1:
    case OpeningPhase::WHITE_MOVE_2: black = 1; white = 0; return;
    case OpeningPhase::SWAP_AFTER_2:
    case OpeningPhase::BLACK_MOVE_3: black = 1; white = 1; return;
    case OpeningPhase::SWAP_AFTER_3:
    case OpeningPhase::WHITE_MOVE_4: black = 2; white = 1; return;
    case OpeningPhase::CHOICE_AFTER_4:
    case OpeningPhase::SWAP_BEFORE_DIRECT_5:
    case OpeningPhase::BLACK_DIRECT_MOVE_5:
    case OpeningPhase::OFFER_TEN_5TH_MOVES:
    case OpeningPhase::SELECT_ONE_OF_TEN: black = 2; white = 2; return;
    case OpeningPhase::SWAP_AFTER_DIRECT_5:
    case OpeningPhase::WHITE_MOVE_6: black = 3; white = 2; return;
    case OpeningPhase::NORMAL_RENJU: black = -1; white = -1; return;
  }
}

}  // namespace dz
