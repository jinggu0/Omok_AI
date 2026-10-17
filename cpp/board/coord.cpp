#include "board/coord.h"

#include <cctype>

namespace dz {

std::string to_notation(Point p) {
  if (!is_valid_point(p)) return "--";
  std::string s;
  s += static_cast<char>('A' + col_of(p));
  s += std::to_string(row_of(p) + 1);
  return s;
}

std::optional<Point> parse_point(std::string_view s) {
  if (s.size() < 2 || s.size() > 3) return std::nullopt;
  const char letter = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
  if (letter < 'A' || letter >= 'A' + kBoardSize) return std::nullopt;
  int rank = 0;
  for (size_t i = 1; i < s.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) return std::nullopt;
    rank = rank * 10 + (s[i] - '0');
  }
  if (s[1] == '0') return std::nullopt;  // no leading zeros
  const int row = rank - 1;
  const int col = letter - 'A';
  if (!on_board(row, col)) return std::nullopt;
  return make_point(row, col);
}

const char* direction_name(int dir) {
  switch (dir) {
    case HORIZONTAL: return "HORIZONTAL";
    case VERTICAL: return "VERTICAL";
    case DIAG: return "DIAG";
    case ANTI_DIAG: return "ANTI_DIAG";
    default: return "?";
  }
}

std::optional<int> parse_direction(std::string_view s) {
  if (s == "H" || s == "HORIZONTAL") return HORIZONTAL;
  if (s == "V" || s == "VERTICAL") return VERTICAL;
  if (s == "D" || s == "DIAG") return DIAG;
  if (s == "A" || s == "ANTI_DIAG") return ANTI_DIAG;
  return std::nullopt;
}

}  // namespace dz
