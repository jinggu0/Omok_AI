#include "board/transform.h"

#include <array>

namespace dz {
namespace {

// (dr, dc) relative to H8 -> transformed (dr, dc). Table in RULES.md §1.5.
void apply_raw(int t, int dr, int dc, int& out_dr, int& out_dc) {
  switch (t) {
    case 0: out_dr = dr;  out_dc = dc;  break;  // identity
    case 1: out_dr = dc;  out_dc = -dr; break;  // rot90 (counter-clockwise)
    case 2: out_dr = -dr; out_dc = -dc; break;  // rot180
    case 3: out_dr = -dc; out_dc = dr;  break;  // rot270
    case 4: out_dr = dr;  out_dc = -dc; break;  // mirror (left-right)
    case 5: out_dr = -dc; out_dc = -dr; break;  // rot90 after mirror
    case 6: out_dr = -dr; out_dc = dc;  break;  // rot180 after mirror (up-down)
    case 7: out_dr = dc;  out_dc = dr;  break;  // rot270 after mirror
    default: out_dr = dr; out_dc = dc; break;
  }
}

struct Tables {
  std::array<std::array<Point, kNumPoints>, kNumTransforms> point{};
  std::array<std::array<int, kNumTransforms>, kNumTransforms> compose{};
  std::array<int, kNumTransforms> inverse{};

  Tables() {
    for (int t = 0; t < kNumTransforms; ++t) {
      for (int p = 0; p < kNumPoints; ++p) {
        int dr = 0, dc = 0;
        apply_raw(t, row_of(static_cast<Point>(p)) - kCenterRowCol,
                  col_of(static_cast<Point>(p)) - kCenterRowCol, dr, dc);
        point[t][p] = make_point(dr + kCenterRowCol, dc + kCenterRowCol);
      }
    }
    // Derive composition/inverse by brute force over all points.
    for (int a = 0; a < kNumTransforms; ++a) {
      for (int b = 0; b < kNumTransforms; ++b) {
        compose[a][b] = -1;
        for (int c = 0; c < kNumTransforms; ++c) {
          bool same = true;
          for (int p = 0; p < kNumPoints && same; ++p) {
            same = point[a][point[b][p]] == point[c][p];
          }
          if (same) { compose[a][b] = c; break; }
        }
      }
    }
    for (int a = 0; a < kNumTransforms; ++a) {
      for (int b = 0; b < kNumTransforms; ++b) {
        if (compose[a][b] == 0) inverse[a] = b;
      }
    }
  }
};

const Tables& tables() {
  static const Tables t;
  return t;
}

}  // namespace

Point transform_point(int t, Point p) { return tables().point[t][p]; }
int compose_transforms(int a, int b) { return tables().compose[a][b]; }
int inverse_transform(int t) { return tables().inverse[t]; }

const char* transform_name(int t) {
  static const char* kNames[kNumTransforms] = {
      "identity", "rot90", "rot180", "rot270",
      "mirror", "mirror_rot90", "mirror_rot180", "mirror_rot270"};
  return (t >= 0 && t < kNumTransforms) ? kNames[t] : "?";
}

}  // namespace dz
