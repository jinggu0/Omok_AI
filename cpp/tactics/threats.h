// ThreatDetector (ARCHITECTURE.md §3.7): five points, fours, double threats and threes for either
// color, respecting black forbidden moves. Tactical search runs in NORMAL_RENJU only.
#pragma once

#include <optional>
#include <vector>

#include "board/board.h"
#include "game/game_state.h"
#include "rules/forbidden.h"

namespace dz::tactics {

struct Rules {
  BlackRuleOptions black;
  bool pass_allowed = false;
};

// Result of a tactical proof, from the perspective of the side to move at the root.
// UNKNOWN is never a loss: it only means "not proven" (budget, depth or search restriction).
enum class ProofResult : uint8_t { WIN, LOSS, UNKNOWN };
const char* proof_name(ProofResult r);

// Lightweight NORMAL_RENJU position used by the solvers (no opening state, no history).
class TPos {
 public:
  TPos(const Board& board, Color to_move, Rules rules) : board_(board), to_move_(to_move), rules_(rules) {}
  // nullopt unless the game is ongoing and in NORMAL_RENJU.
  static std::optional<TPos> from_state(const GameState& g);

  const Board& board() const { return board_; }
  Board& board() { return board_; }
  Color to_move() const { return to_move_; }
  const Rules& rules() const { return rules_; }

  // No legality check: callers generate legal moves.
  void play(Point p) { board_.place(p, to_move_); to_move_ = opposite(to_move_); }
  void undo(Point p) { to_move_ = opposite(to_move_); board_.remove(p); }
  void pass() { to_move_ = opposite(to_move_); }
  void undo_pass() { to_move_ = opposite(to_move_); }
  uint64_t key() const;

 private:
  Board board_;
  Color to_move_;
  Rules rules_;
};

// Empty point p is a legal move for c (black: not forbidden).
bool is_legal_move(const Board& b, Point p, Color c, const Rules& r);
// Playing empty point p wins immediately for c (black: exact five, honouring the profile options).
bool wins_with(const Board& b, Point p, Color c, const Rules& r);

// All empty points where c wins immediately.
void five_points(const Board& b, Color c, const Rules& r, std::vector<Point>& out);
// Five points of c on the four lines through `center` (within distance 4). Use after placing a
// stone at `center` when c had no five point before: every new five point lies on those lines.
void five_points_near(const Board& b, Point center, Color c, const Rules& r, std::vector<Point>& out);

// Legal moves of c that create at least one five point (fours, including straight fours).
void four_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out);
// Legal moves of c that create at least two five points (straight four / double four): an
// immediate double threat. Precondition: c has no five point.
void double_threat_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out);
// Fours first, then moves creating a three (black: real three; white: straight-four threat).
void threat_moves(Board& b, Color c, const Rules& r, std::vector<Point>& out);

}  // namespace dz::tactics
