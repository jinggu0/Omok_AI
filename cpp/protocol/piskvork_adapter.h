// Gomocup / Piskvork protocol codec (ARCHITECTURE.md §3.12, PRD FR-21).
//
// Pure text <-> value conversion plus the engine-side command loop. No process I/O (see
// gomocup_process.h), no search, and no rule logic: legality, forbidden moves and game end are
// always decided by the canonical GameState.
//
// Coordinates: the protocol's "x,y" has its origin at the TOP-left corner (x = column, y = row,
// growing downwards; ASSUMPTIONS A-P2). The project's single coordinate system (RULES.md §1) has
// row 0 at the BOTTOM. The only conversion between the two is gomocup_to_point / point_to_gomocup.
// (Renju rules are D4-invariant, so even a wrong orientation would give equivalent games, but
// opening corpus positions would then be mirrored for the external engine.)
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "board/coord.h"
#include "game/game_state.h"

namespace dz::protocol {

// Gomocup "INFO rule" bit mask: 1 = exactly five, 2 = continuous game, 4 = renju, 8 = caro.
inline constexpr int kRuleExactFive = 1;
inline constexpr int kRuleContinuous = 2;
inline constexpr int kRuleRenju = 4;

std::optional<Point> gomocup_to_point(int x, int y);
void point_to_gomocup(Point p, int& x, int& y);
std::string format_move(Point p);                // "x,y"
std::string format_board_line(Point p, int field);  // "x,y,field" (1 own, 2 opponent)

// ---- manager -> engine commands ----
enum class CommandType { START, RECTSTART, RESTART, BEGIN, TURN, BOARD, INFO, END, ABOUT, TAKEBACK, UNKNOWN };

struct Command {
  CommandType type = CommandType::UNKNOWN;
  int size = 0, width = 0, height = 0;  // START / RECTSTART
  int x = -1, y = -1;                    // TURN / TAKEBACK
  std::string key, value;                // INFO
  std::string raw;
};
// Commands are case-insensitive keywords; arguments separated by spaces/commas.
Command parse_command(const std::string& line);

// ---- engine -> manager replies ----
enum class ReplyKind { MOVE, OK, ERROR, UNKNOWN, MESSAGE, DEBUG, SUGGEST, OTHER };

struct Reply {
  ReplyKind kind = ReplyKind::OTHER;
  int x = -1, y = -1;  // MOVE / SUGGEST
  std::string text;    // the rest of the line (ERROR/UNKNOWN/MESSAGE/DEBUG) or the raw line
};
Reply parse_reply(const std::string& line);

// ---- engine side (pbrain): manager commands -> GameState + a move callback ----
struct ThinkLimits {
  int timeout_turn_ms = 0;   // 0 = not given
  int timeout_match_ms = 0;
  int time_left_ms = -1;     // -1 = not given
  int max_memory_bytes = 0;
};

// Returns the move to play for the side to move of `g` (must be legal; the session checks).
using ThinkFn = std::function<std::optional<Point>(const GameState& g, const ThinkLimits& limits)>;

class EngineSession {
 public:
  // `about` is the ABOUT reply (e.g. name="DojoZero", version="0.8"). Only Renju (rule bit 4)
  // on 15x15 is supported; other rules / sizes answer ERROR.
  EngineSession(ThinkFn think, std::string about);

  // Feed one input line; returns the lines to write back (possibly none). `quit` becomes true on END.
  std::vector<std::string> feed(const std::string& line);
  bool quit() const { return quit_; }
  const GameState* state() const { return state_ ? &*state_ : nullptr; }
  const ThinkLimits& limits() const { return limits_; }
  int rule() const { return rule_; }

 private:
  std::vector<std::string> think_and_reply();
  std::vector<std::string> finish_board();

  ThinkFn think_;
  std::string about_;
  std::optional<GameState> state_;
  ThinkLimits limits_;
  int rule_ = kRuleRenju;  // assumed until INFO rule says otherwise
  bool in_board_ = false;
  std::vector<std::pair<Point, int>> board_lines_;
  bool quit_ = false;
};

}  // namespace dz::protocol
