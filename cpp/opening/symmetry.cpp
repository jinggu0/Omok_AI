#include "opening/symmetry.h"

#include <algorithm>
#include <vector>

namespace dz {
namespace {

std::vector<Point> stones_of(const Board& b, Cell cell) {
  std::vector<Point> out;
  for (int p = 0; p < kNumPoints; ++p) {
    if (b.at(static_cast<Point>(p)) == cell) out.push_back(static_cast<Point>(p));
  }
  return out;
}

bool whites_preserved(const Board& b, const std::vector<Point>& whites, int t) {
  for (Point w : whites) {
    if (b.at(transform_point(t, w)) != WHITE_STONE) return false;
  }
  return true;
}

// Under transform t, is board + black q equivalent to board + black q' for some q'? Returns q'
// (kNoPoint if t does not map the position onto a position of that form). RULES.md §10.2 E1.
Point position_image(const Board& b, const std::vector<Point>& blacks,
                     const std::vector<Point>& whites, Point q, int t) {
  if (!whites_preserved(b, whites, t)) return kNoPoint;
  // Image X = t(B ∪ {q}); need B ⊆ X, then q' = X \ B.
  Point leftover = kNoPoint;
  int leftovers = 0;
  auto visit = [&](Point x) {
    const Point tx = transform_point(t, x);
    if (b.at(tx) != BLACK_STONE) {
      leftover = tx;
      ++leftovers;
    }
  };
  for (Point x : blacks) visit(x);
  visit(q);
  // |X| = |B| + 1 and t is injective, so B ⊆ X iff exactly one image point is not black.
  if (leftovers != 1 || b.at(leftover) != EMPTY) return kNoPoint;
  return leftover;
}

struct ProposalContext {
  std::array<Point, kNumPoints> cls{};  // class id of each admissible candidate, kNoPoint otherwise
  int total_classes = 0;
};

ProposalContext build_context(const Board& b, const Taraguchi10Params& params,
                              const BlackRuleOptions& black_opt) {
  ProposalContext ctx;
  ctx.cls.fill(kNoPoint);
  std::vector<bool> seen(kNumPoints, false);
  for (int i = 0; i < kNumPoints; ++i) {
    const Point p = static_cast<Point>(i);
    if (!b.empty_at(p)) continue;
    if (params.proposal_region != 0 && !in_central(p, params.proposal_region)) continue;
    if (is_forbidden(black_verdict(b, p, black_opt))) continue;
    const Point c = canonical_proposal(b, p, params.proposal_symmetry);
    ctx.cls[p] = c;
    if (!seen[c]) {
      seen[c] = true;
      ++ctx.total_classes;
    }
  }
  return ctx;
}

bool fail(std::string* why, const std::string& msg) {
  if (why) *why = msg;
  return false;
}

bool admissible_with_context(const Board& board, const ProposalContext& ctx,
                             std::span<const Point> offered, Point p,
                             const Taraguchi10Params& params, std::string* why) {
  if (!is_valid_point(p)) return fail(why, "invalid point");
  if (static_cast<int>(offered.size()) >= params.n_proposals) return fail(why, "proposal set is full");
  if (!board.empty_at(p)) return fail(why, to_notation(p) + " is occupied");
  if (std::find(offered.begin(), offered.end(), p) != offered.end()) {
    return fail(why, to_notation(p) + " already proposed");
  }
  if (params.proposal_region != 0 && !in_central(p, params.proposal_region)) {
    return fail(why, to_notation(p) + " outside proposal region");
  }
  if (ctx.cls[p] == kNoPoint) return fail(why, to_notation(p) + " is not a legal black move");
  std::vector<Point> used;
  for (Point q : offered) {
    if (ctx.cls[q] == kNoPoint) return fail(why, "existing proposal " + to_notation(q) + " is invalid");
    if (ctx.cls[q] == ctx.cls[p]) {
      return fail(why, to_notation(p) + " is symmetric to " + to_notation(q));
    }
    if (std::find(used.begin(), used.end(), ctx.cls[q]) != used.end()) {
      return fail(why, "existing proposals are mutually symmetric");
    }
    used.push_back(ctx.cls[q]);
  }
  const int needed_after = params.n_proposals - static_cast<int>(offered.size()) - 1;
  const int free_after = ctx.total_classes - static_cast<int>(used.size()) - 1;
  if (free_after < needed_after) return fail(why, "not enough non-symmetric points left");
  return true;
}

}  // namespace

uint8_t stabilizer_mask(const Board& board) {
  uint8_t mask = 1;
  for (int t = 1; t < kNumTransforms; ++t) {
    bool ok = true;
    for (int p = 0; p < kNumPoints && ok; ++p) {
      ok = board.at(transform_point(t, static_cast<Point>(p))) == board.at(static_cast<Point>(p));
    }
    if (ok) mask = static_cast<uint8_t>(mask | (1u << t));
  }
  return mask;
}

bool proposals_equivalent(const Board& board, Point q1, Point q2, SymmetryMode mode) {
  if (q1 == q2) return true;
  if (mode == SymmetryMode::POINTWISE_D4) {
    for (int t = 0; t < kNumTransforms; ++t) {
      if (transform_point(t, q1) == q2) return true;
    }
    return false;
  }
  const auto blacks = stones_of(board, BLACK_STONE);
  const auto whites = stones_of(board, WHITE_STONE);
  for (int t = 0; t < kNumTransforms; ++t) {
    if (position_image(board, blacks, whites, q1, t) == q2) return true;
  }
  return false;
}

bool proposals_stabilizer_equivalent(const Board& board, Point q1, Point q2) {
  const uint8_t stab = stabilizer_mask(board);
  for (int t = 0; t < kNumTransforms; ++t) {
    if ((stab & (1u << t)) && transform_point(t, q1) == q2) return true;
  }
  return false;
}

Point canonical_proposal(const Board& board, Point q, SymmetryMode mode) {
  Point best = q;
  if (mode == SymmetryMode::POINTWISE_D4) {
    for (int t = 0; t < kNumTransforms; ++t) best = std::min(best, transform_point(t, q));
    return best;
  }
  const auto blacks = stones_of(board, BLACK_STONE);
  const auto whites = stones_of(board, WHITE_STONE);
  for (int t = 0; t < kNumTransforms; ++t) {
    const Point img = position_image(board, blacks, whites, q, t);
    if (img != kNoPoint) best = std::min(best, img);
  }
  return best;
}

bool proposal_admissible(const Board& board, std::span<const Point> offered, Point p,
                         const Taraguchi10Params& params, const BlackRuleOptions& black_opt,
                         std::string* why) {
  const ProposalContext ctx = build_context(board, params, black_opt);
  return admissible_with_context(board, ctx, offered, p, params, why);
}

bool validate_proposals(const Board& board, std::span<const Point> proposals,
                        const Taraguchi10Params& params, const BlackRuleOptions& black_opt,
                        std::string* why) {
  if (static_cast<int>(proposals.size()) != params.n_proposals) {
    return fail(why, "expected " + std::to_string(params.n_proposals) + " proposals");
  }
  const ProposalContext ctx = build_context(board, params, black_opt);
  for (size_t i = 0; i < proposals.size(); ++i) {
    if (!admissible_with_context(board, ctx, proposals.first(i), proposals[i], params, why)) return false;
  }
  return true;
}

std::vector<Point> admissible_proposals(const Board& board, std::span<const Point> offered,
                                        const Taraguchi10Params& params,
                                        const BlackRuleOptions& black_opt) {
  const ProposalContext ctx = build_context(board, params, black_opt);
  std::vector<Point> out;
  for (int i = 0; i < kNumPoints; ++i) {
    if (admissible_with_context(board, ctx, offered, static_cast<Point>(i), params, nullptr)) {
      out.push_back(static_cast<Point>(i));
    }
  }
  return out;
}

}  // namespace dz
