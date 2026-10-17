// RuleProfile (ARCHITECTURE.md §3.4). Every behaviour that may differ from the Omokdojang app is a
// field here (ASSUMPTIONS.md Compatibility Matrix), so evidence can change defaults without code.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "rules/forbidden.h"

namespace dz {

enum class PlayerId : uint8_t { P0 = 0, P1 = 1 };
constexpr PlayerId other_player(PlayerId p) { return p == PlayerId::P0 ? PlayerId::P1 : PlayerId::P0; }
const char* player_name(PlayerId p);

// Which player currently holds which color (RULES.md §8.2). Swap exchanges the two.
struct SideAssignment {
  PlayerId black_holder = PlayerId::P0;
  PlayerId white_holder = PlayerId::P1;

  PlayerId holder(Color c) const { return c == Color::BLACK ? black_holder : white_holder; }
  Color color_of(PlayerId p) const { return p == black_holder ? Color::BLACK : Color::WHITE; }
  SideAssignment swapped() const { return {white_holder, black_holder}; }
  bool valid() const { return black_holder != white_holder; }
  bool operator==(const SideAssignment&) const = default;
};

// Actor role of a decision phase; mapped to a PlayerId through SideAssignment.
enum class Role : uint8_t { BLACK_HOLDER, WHITE_HOLDER };
const char* role_name(Role r);

enum class OpeningRule : uint8_t { NONE, FIXED_POSITION, TARAGUCHI_10 };
enum class IllegalMoveBehavior : uint8_t { REJECT, LOSS };
// Ten-proposal symmetry (RULES.md §10.2): POSITION = E1 (default), POINTWISE_D4 = E3.
enum class SymmetryMode : uint8_t { POSITION, POINTWISE_D4 };

struct Taraguchi10Params {
  // Central(N) region for moves 1..5 (direct fifth). Move 6 is unrestricted.
  std::array<int, 5> region = {1, 3, 5, 7, 9};
  int n_proposals = 10;
  Role choice_actor = Role::BLACK_HOLDER;               // ASSUMPTIONS A-T1 / C-07
  Role swap_before_direct5_actor = Role::BLACK_HOLDER;  // ASSUMPTIONS A-T2 / C-08
  SymmetryMode proposal_symmetry = SymmetryMode::POSITION;  // A-T3 / C-10
  int proposal_region = 0;  // 0 = anywhere on the board (A-T5), else Central(N)
};

inline constexpr int kMaxProposals = 16;

struct RuleProfile {
  std::string id = "RENJU_RIF";
  OpeningRule opening_rule = OpeningRule::NONE;
  bool pass_allowed = true;
  IllegalMoveBehavior illegal_move_behavior = IllegalMoveBehavior::REJECT;
  BlackRuleOptions black;  // exact_five_precedence, five_beats_overline_other_dir
  Taraguchi10Params taraguchi;

  // Fixed Renju constants; listed so a profile file states them explicitly (PRD FR-03).
  static constexpr int kBoardSize = 15;
  static constexpr const char* kBlackWin = "EXACT_FIVE";
  static constexpr const char* kWhiteWin = "FIVE_OR_MORE";

  static RuleProfile renju_rif();
  static RuleProfile taraguchi10();
  static RuleProfile omokdojang_dan();

  nlohmann::json to_json() const;
  // Rejects unknown keys and unsupported values (no silent defaults for misspelled fields).
  static std::optional<RuleProfile> from_json(const nlohmann::json& j, std::string* error);
  // FNV-1a 64 of the canonical JSON dump; recorded with games/checkpoints.
  uint64_t hash() const;
};

using ProfilePtr = std::shared_ptr<const RuleProfile>;
inline ProfilePtr make_profile(RuleProfile p) { return std::make_shared<const RuleProfile>(std::move(p)); }

}  // namespace dz
