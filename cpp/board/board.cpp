#include "board/board.h"

#include <cassert>

#include "board/zobrist.h"

namespace dz {

namespace {

// stone key of (color, point) under each transform, contiguous per point: one lookup per update.
struct StoneKeys {
  std::array<std::array<std::array<uint64_t, kNumTransforms>, kNumPoints>, 2> k{};
  StoneKeys() {
    for (int c = 0; c < 2; ++c) {
      for (int p = 0; p < kNumPoints; ++p) {
        for (int t = 0; t < kNumTransforms; ++t) {
          k[c][p][t] = zobrist::stone(c, transform_point(t, static_cast<Point>(p)));
        }
      }
    }
  }
};

const StoneKeys& stone_keys() {
  static const StoneKeys keys;
  return keys;
}

}  // namespace

const char* color_name(Color c) { return c == Color::BLACK ? "BLACK" : "WHITE"; }

Board::Board() {
  cells_.fill(WALL);
  for (int p = 0; p < kNumPoints; ++p) cells_[index_of(static_cast<Point>(p))] = EMPTY;
}

void Board::place(Point p, Color c) {
  assert(is_valid_point(p) && empty_at(p));
  const int ci = static_cast<int>(c);
  cells_[index_of(p)] = cell_of(c);
  ++count_[ci];
  const auto& keys = stone_keys().k[ci][p];
  for (int t = 0; t < kNumTransforms; ++t) hash_[t] ^= keys[t];
}

void Board::remove(Point p) {
  assert(is_valid_point(p) && !empty_at(p));
  const int ci = at(p) == BLACK_STONE ? 0 : 1;
  cells_[index_of(p)] = EMPTY;
  --count_[ci];
  const auto& keys = stone_keys().k[ci][p];
  for (int t = 0; t < kNumTransforms; ++t) hash_[t] ^= keys[t];
}

Board Board::transformed(int t) const {
  Board out;
  for (int p = 0; p < kNumPoints; ++p) {
    const Cell c = at(static_cast<Point>(p));
    if (c == BLACK_STONE) out.place(transform_point(t, static_cast<Point>(p)), Color::BLACK);
    if (c == WHITE_STONE) out.place(transform_point(t, static_cast<Point>(p)), Color::WHITE);
  }
  return out;
}

std::string Board::to_string() const {
  std::string s;
  for (int row = kBoardSize - 1; row >= 0; --row) {
    const int rank = row + 1;
    if (rank < 10) s += ' ';
    s += std::to_string(rank);
    for (int col = 0; col < kBoardSize; ++col) {
      s += ' ';
      const Cell c = at(make_point(row, col));
      s += c == BLACK_STONE ? 'X' : c == WHITE_STONE ? 'O' : '.';
    }
    s += '\n';
  }
  s += "  ";
  for (int col = 0; col < kBoardSize; ++col) {
    s += ' ';
    s += static_cast<char>('A' + col);
  }
  s += '\n';
  return s;
}

}  // namespace dz
