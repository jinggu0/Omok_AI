// Fast black move verdict (RULES.md §3-§6). Algorithm: RULES.md §4.
// Recursion is unbounded by design (terminates because each level adds a stone).
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "board/board.h"

namespace dz {

enum class MoveVerdict : uint8_t {
  WIN_FIVE,
  LEGAL,
  FORBIDDEN_OVERLINE,
  FORBIDDEN_DOUBLE_FOUR,
  FORBIDDEN_DOUBLE_THREE,
};
const char* verdict_name(MoveVerdict v);
std::optional<MoveVerdict> parse_verdict(std::string_view s);
constexpr bool is_forbidden(MoveVerdict v) {
  return v == MoveVerdict::FORBIDDEN_OVERLINE || v == MoveVerdict::FORBIDDEN_DOUBLE_FOUR ||
         v == MoveVerdict::FORBIDDEN_DOUBLE_THREE;
}

struct BlackRuleOptions {
  bool exact_five_precedence = true;          // RULES.md §6
  bool five_beats_overline_other_dir = true;  // ASSUMPTIONS A-R1 / C-02
};

// Verdict for black playing at empty point p. Board is not modified.
MoveVerdict black_verdict(const Board& board, Point p, const BlackRuleOptions& opt = {});
// Same, on a padded-cell scratch array that the caller allows to be mutated temporarily
// (it is restored before returning). idx is a padded index (Board::index_of).
MoveVerdict black_verdict_cells(Board::Cells& cells, int idx, const BlackRuleOptions& opt);

// --- Explanation (RULES.md §4.1). Slower; for debugging, CLI and tests. -------------
struct FourInfo {
  int dir = 0;
  std::vector<Point> stones;       // the 4 stones (includes the move)
  std::vector<Point> completions;  // points that turn it into an exact five
};
struct ExtensionInfo {
  Point point = kNoPoint;          // makes a straight four in this direction
  MoveVerdict verdict = MoveVerdict::LEGAL;  // legality of that extension (recursive)
};
struct ThreeInfo {
  int dir = 0;
  std::vector<ExtensionInfo> extensions;  // all straight-four extension points
  bool real = false;                      // at least one extension is LEGAL
};
struct RecursionStep {
  Point point = kNoPoint;  // extension point whose legality was evaluated
  Point parent = kNoPoint; // move whose three it extends
  int depth = 0;           // 1 = extension of the queried move
  MoveVerdict verdict = MoveVerdict::LEGAL;
};
struct ForbiddenInfo {
  Point point = kNoPoint;
  MoveVerdict verdict = MoveVerdict::LEGAL;
  std::vector<int> five_dirs;
  std::vector<int> overline_dirs;
  std::vector<FourInfo> fours;
  std::vector<ThreeInfo> threes;  // directions holding an apparent three
  std::vector<RecursionStep> recursion;
  bool recursion_truncated = false;

  int four_count() const { return static_cast<int>(fours.size()); }
  int apparent_three_count() const { return static_cast<int>(threes.size()); }
  int real_three_count() const;
  std::string to_string() const;
};
ForbiddenInfo explain_black_move(const Board& board, Point p, const BlackRuleOptions& opt = {});

// --- Pattern queries for either color (used by tactics later). ---------------------
// "five" means exact five for black and five-or-more for white.
bool makes_five(const Board& board, Point p, Color c);
bool makes_overline(const Board& board, Point p, Color c);  // run >= 6
int count_fours(const Board& board, Point p, Color c);      // distinct fours through p
// Black: real threes (recursive legality). White: directions with an apparent three.
int count_threes(const Board& board, Point p, Color c, const BlackRuleOptions& opt = {});

}  // namespace dz
