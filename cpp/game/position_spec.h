// Position input (tests, compatibility cases, CLI) and malformed-state rejection (RULES.md §8.3).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "game/game_state.h"

namespace dz {

struct PositionSpec {
  // Either replay these actions from a fresh game (strongest validation) ...
  std::optional<std::vector<std::string>> moves;
  // ... or set the position up directly.
  std::vector<Point> black, white;
  std::optional<OpeningPhase> phase;   // default NORMAL_RENJU
  std::optional<Color> side_to_move;   // NORMAL_RENJU only; inferred from stone counts if absent
  SideAssignment assignment;
  std::vector<Point> offered;
  Point last_move = kNoPoint;
};

// JSON keys: "moves" | "black", "white", "lines" [{"from","dir","cells"}], "diagram" (15 rows,
// rank 15 first, 'X' black, 'O' white, '.' empty), "phase", "side", "assignment"
// {"black": "P0", "white": "P1"}, "offered", "last_move".
std::optional<PositionSpec> parse_position_spec(const nlohmann::json& j, std::string* error);

// Validates and builds the state. Rejected (nullopt + error): duplicate/overlapping stones,
// stone counts or region placement inconsistent with the phase, invalid offered set, invalid
// assignment, illegal replayed move, or a position that already contains a five/overline.
std::optional<GameState> load_position(const PositionSpec& spec, ProfilePtr profile, std::string* error);

}  // namespace dz
