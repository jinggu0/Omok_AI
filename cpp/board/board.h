// Board representation A (ARCHITECTURE.md §3.2): padded cell array.
// A border of WALL cells of width kPad lets line scans run without bounds checks.
// The board also keeps one Zobrist hash per D4 transform (ARCHITECTURE.md §3.3).
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "board/coord.h"
#include "board/transform.h"

namespace dz {

enum class Color : uint8_t { BLACK = 0, WHITE = 1 };
constexpr Color opposite(Color c) { return c == Color::BLACK ? Color::WHITE : Color::BLACK; }
const char* color_name(Color c);

enum Cell : uint8_t { EMPTY = 0, BLACK_STONE = 1, WHITE_STONE = 2, WALL = 3 };
constexpr Cell cell_of(Color c) { return c == Color::BLACK ? BLACK_STONE : WHITE_STONE; }

class Board {
 public:
  static constexpr int kPad = 5;
  static constexpr int kStride = kBoardSize + 2 * kPad;
  static constexpr int kPaddedSize = kStride * kStride;
  // Index step for each Direction in the padded array.
  static constexpr int kStep[kNumDirections] = {1, kStride, kStride + 1, kStride - 1};

  static constexpr int index_of(Point p) { return (row_of(p) + kPad) * kStride + col_of(p) + kPad; }
  static constexpr Point point_of_index(int idx) {
    return make_point(idx / kStride - kPad, idx % kStride - kPad);
  }

  using Cells = std::array<uint8_t, kPaddedSize>;

  Board();

  Cell at(Point p) const { return static_cast<Cell>(cells_[index_of(p)]); }
  bool empty_at(Point p) const { return at(p) == EMPTY; }

  // Precondition: p is empty. Updates counts and all 8 hashes.
  void place(Point p, Color c);
  // Precondition: p holds a stone.
  void remove(Point p);

  int stone_count(Color c) const { return count_[static_cast<int>(c)]; }
  int stone_count() const { return count_[0] + count_[1]; }
  int empty_count() const { return kNumPoints - stone_count(); }

  // Hash of this board as seen through transform t (t = 0 is the plain hash).
  uint64_t hash(int t = 0) const { return hash_[t]; }

  // Raw padded cells; used by the rule checkers which need cheap scratch copies.
  const Cells& cells() const { return cells_; }

  Board transformed(int t) const;

  bool operator==(const Board& o) const { return cells_ == o.cells_; }

  // Text diagram, rank 15 at the top. 'X' = black, 'O' = white, '.' = empty.
  std::string to_string() const;

 private:
  Cells cells_{};
  std::array<int, 2> count_{0, 0};
  std::array<uint64_t, kNumTransforms> hash_{};
};

}  // namespace dz
