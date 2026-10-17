// GameState (ARCHITECTURE.md §3.6): board + opening state machine + side assignment + history.
// apply/undo are incremental (no board copies); undo restores phase, assignment, offered set.
#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "board/board.h"
#include "game/action.h"
#include "opening/opening_phase.h"
#include "rules/forbidden.h"
#include "rules/rule_profile.h"

namespace dz {

enum class GameResult : uint8_t { ONGOING, BLACK_WIN, WHITE_WIN, DRAW };
enum class EndReason : uint8_t { NONE, FIVE, FORBIDDEN_MOVE, BOARD_FULL, DOUBLE_PASS, NO_LEGAL_MOVE };
const char* result_name(GameResult r);
const char* end_reason_name(EndReason r);

struct Outcome {
  GameResult result = GameResult::ONGOING;
  EndReason reason = EndReason::NONE;
  Point point = kNoPoint;  // the move that ended the game, if any
  bool operator==(const Outcome&) const = default;
};

// Direct setup (validated by load_position in position_spec.h).
struct Setup {
  std::vector<Point> black, white;
  OpeningPhase phase = OpeningPhase::NORMAL_RENJU;
  Color side_to_move = Color::BLACK;  // used in NORMAL_RENJU
  SideAssignment assignment;
  std::vector<Point> offered;
  Point last_move = kNoPoint;
};

struct PositionSpec;
class GameState;
std::optional<GameState> load_position(const PositionSpec& spec, ProfilePtr profile, std::string* error);

class GameState {
 public:
  // Fresh game: BLACK_MOVE_1 for Taraguchi-10, empty board NORMAL_RENJU otherwise.
  explicit GameState(ProfilePtr profile, SideAssignment initial = {});

  // Returns false (state unchanged) for any illegal action. Under illegal_move_behavior == LOSS a
  // forbidden black PLACE is accepted and ends the game as a white win (RULES.md §7.4).
  bool apply(const Action& a);
  // Precondition: can_undo().
  void undo();
  bool can_undo() const { return !history_.empty(); }

  // Actions an engine may choose. Never contains forbidden black moves.
  std::vector<Action> legal_actions() const;
  void legal_actions(std::vector<Action>& out) const;
  // Whether apply(a) would succeed (includes forbidden black moves under LOSS behaviour).
  bool is_legal(const Action& a) const;

  MoveVerdict black_verdict(Point p) const { return dz::black_verdict(board_, p, profile_->black); }
  ForbiddenInfo forbidden(Point p) const { return explain_black_move(board_, p, profile_->black); }

  const Outcome& outcome() const { return outcome_; }
  bool is_terminal() const { return outcome_.result != GameResult::ONGOING; }
  // Winner as a player (assignment is fixed once the game reaches NORMAL_RENJU).
  std::optional<PlayerId> winner() const;

  OpeningPhase phase() const { return phase_; }
  Role actor_role() const;
  PlayerId actor() const { return assignment_.holder(actor_role() == Role::BLACK_HOLDER ? Color::BLACK : Color::WHITE); }
  // Color of the next stone to be placed (for decision phases: the stone after the decision).
  Color side_to_move() const;
  const SideAssignment& assignment() const { return assignment_; }

  const Board& board() const { return board_; }
  const RuleProfile& profile() const { return *profile_; }
  const ProfilePtr& profile_ptr() const { return profile_; }
  std::span<const Point> offered() const { return {offered_.data(), static_cast<size_t>(n_offered_)}; }
  Point chosen_fifth() const { return chosen_fifth_; }
  Point last_move() const { return last_move_; }
  int consecutive_passes() const { return passes_; }

  std::vector<Action> action_history() const;
  const Setup& initial_setup() const { return initial_; }

  // Position hash: stones, phase, color to move, offered set (in proposal phases).
  uint64_t hash() const { return hash_under(0); }
  uint64_t canonical_hash(int* transform = nullptr) const;
  // D4-transformed copy (initial setup and every action transformed, then replayed).
  GameState transformed(int t) const;

  std::string to_string() const;

 private:
  friend std::optional<GameState> load_position(const PositionSpec&, ProfilePtr, std::string*);
  GameState(ProfilePtr profile, const Setup& setup);  // trusted; load_position validates

  struct UndoRecord {
    Action action;
    OpeningPhase phase;
    Color side;
    SideAssignment assignment;
    Point last_move;
    Point chosen_fifth;
    Point placed;
    uint8_t n_offered;
    uint8_t passes;
    Outcome outcome;
  };

  // verdict_out receives the black verdict of a PLACE (LEGAL otherwise) so apply() computes it once.
  bool is_legal_impl(const Action& a, bool for_engine, MoveVerdict* verdict_out = nullptr) const;
  void place_stone(Point p, Color c, std::optional<MoveVerdict> known_black_verdict = std::nullopt);
  void after_normal_move();
  bool black_has_legal_move() const;
  uint64_t hash_under(int t) const;

  ProfilePtr profile_;
  Board board_;
  OpeningPhase phase_ = OpeningPhase::NORMAL_RENJU;
  Color side_ = Color::BLACK;  // color to move in NORMAL_RENJU
  SideAssignment assignment_;
  std::array<Point, kMaxProposals> offered_{};
  uint8_t n_offered_ = 0;
  Point chosen_fifth_ = kNoPoint;
  Point last_move_ = kNoPoint;
  uint8_t passes_ = 0;
  Outcome outcome_;
  std::vector<UndoRecord> history_;
  Setup initial_;
};

}  // namespace dz
