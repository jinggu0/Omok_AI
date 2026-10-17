// D4 symmetry transforms about H8 (RULES.md §1.5). Transform ids 0..7 are fixed.
#pragma once

#include "board/coord.h"

namespace dz {

inline constexpr int kNumTransforms = 8;

// Image of point p under transform t.
Point transform_point(int t, Point p);
// compose(a, b) = transform equal to "apply b, then a".
int compose_transforms(int a, int b);
int inverse_transform(int t);
const char* transform_name(int t);

}  // namespace dz
