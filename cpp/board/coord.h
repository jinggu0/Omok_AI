// Single coordinate system for the whole project (RULES.md §1.2).
//   row: 0 = bottom (rank 1) .. 14 = top (rank 15)
//   col: 0 = column A (left) .. 14 = column O (right)
//   Point index = row * 15 + col, notation = column letter + rank number ("H8").
// Never use x/y or (col, row) ordering anywhere.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace dz {

inline constexpr int kBoardSize = 15;
inline constexpr int kNumPoints = kBoardSize * kBoardSize;
inline constexpr int kCenterRowCol = 7;

using Point = uint8_t;
inline constexpr Point kNoPoint = 255;

constexpr bool on_board(int row, int col) {
  return row >= 0 && row < kBoardSize && col >= 0 && col < kBoardSize;
}
constexpr Point make_point(int row, int col) { return static_cast<Point>(row * kBoardSize + col); }
constexpr int row_of(Point p) { return p / kBoardSize; }
constexpr int col_of(Point p) { return p % kBoardSize; }
constexpr bool is_valid_point(Point p) { return p < kNumPoints; }

inline constexpr Point kCenterPoint = make_point(kCenterRowCol, kCenterRowCol);  // H8

// "H8" style notation. Columns A..O include the letter I.
std::string to_notation(Point p);
// Case-insensitive; returns nullopt for anything malformed or off-board.
std::optional<Point> parse_point(std::string_view s);

// Directions (RULES.md §1.4).
inline constexpr int kNumDirections = 4;
enum Direction : uint8_t { HORIZONTAL = 0, VERTICAL = 1, DIAG = 2, ANTI_DIAG = 3 };
inline constexpr int kDirDRow[kNumDirections] = {0, 1, 1, 1};
inline constexpr int kDirDCol[kNumDirections] = {1, 0, 1, -1};
const char* direction_name(int dir);
std::optional<int> parse_direction(std::string_view s);  // "H", "V", "D", "A" or full names

// Central(N) region, inclusive (RULES.md §1.3). N must be odd.
constexpr bool in_central(Point p, int n) {
  const int half = (n - 1) / 2;
  const int dr = row_of(p) - kCenterRowCol;
  const int dc = col_of(p) - kCenterRowCol;
  return (dr < 0 ? -dr : dr) <= half && (dc < 0 ? -dc : dc) <= half;
}

}  // namespace dz
