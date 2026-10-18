#include "tactics/vcf.h"

#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace dz::tactics {

const char* vcf_status_name(VcfResult::Status s) {
  switch (s) {
    case VcfResult::Status::FOUND: return "FOUND";
    case VcfResult::Status::NONE: return "NONE";
    case VcfResult::Status::BUDGET: return "BUDGET";
  }
  return "?";
}

namespace {

class VcfSearch {
 public:
  VcfSearch(const TPos& pos, const VcfLimits& limits)
      : pos_(pos), limits_(limits), attacker_(pos.to_move()), defender_(opposite(pos.to_move())) {}

  VcfResult run() {
    VcfResult out;
    const Outcome r = or_node(0);
    out.status = r == Outcome::FOUND ? VcfResult::Status::FOUND
                 : r == Outcome::ABORT ? VcfResult::Status::BUDGET
                                       : VcfResult::Status::NONE;
    if (r == Outcome::FOUND) out.pv = pv_;
    out.nodes = nodes_;
    out.depth_limited = depth_limited_;
    return out;
  }

 private:
  enum class Outcome { FOUND, NONE, ABORT };

  Outcome or_node(int depth) {
    if (++nodes_ > limits_.max_nodes) return Outcome::ABORT;
    if (limits_.max_ms > 0 && (nodes_ & 255) == 0 &&
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count() > limits_.max_ms) {
      return Outcome::ABORT;
    }
    Board& b = pos_.board();
    const Rules& rules = pos_.rules();
    std::vector<Point> buf;

    five_points(b, attacker_, rules, buf);
    if (!buf.empty()) {
      pv_ = {buf.front()};
      return Outcome::FOUND;
    }
    const uint64_t key = pos_.key();
    const int remaining = limits_.max_depth - depth;
    if (auto it = failed_.find(key); it != failed_.end() && it->second >= remaining) return Outcome::NONE;
    if (depth >= limits_.max_depth) {
      depth_limited_ = true;
      return Outcome::NONE;
    }

    std::vector<Point> opp;
    five_points(b, defender_, rules, opp);
    std::vector<Point> candidates;
    if (opp.size() < 2) {
      four_moves(b, attacker_, rules, candidates);
      if (opp.size() == 1) {  // must block, and the block must itself be a four
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [&](Point m) { return m != opp.front(); }),
                         candidates.end());
      }
    }

    for (Point m : candidates) {
      pos_.play(m);
      five_points_near(b, m, attacker_, rules, buf);
      Outcome r = Outcome::NONE;
      if (buf.size() >= 2) {
        // Defender blocks one (if it legally can), attacker completes the other.
        pv_ = {m, is_legal_move(b, buf[0], defender_, rules) ? buf[0] : kNoPoint, buf[1]};
        r = Outcome::FOUND;
      } else if (buf.size() == 1) {
        const Point c = buf.front();
        if (!is_legal_move(b, c, defender_, rules)) {
          pv_ = {m, kNoPoint, c};  // forbidden block point: black cannot defend
          r = Outcome::FOUND;
        } else {
          pos_.play(c);
          r = or_node(depth + 1);
          pos_.undo(c);
          if (r == Outcome::FOUND) pv_.insert(pv_.begin(), {m, c});
        }
      }
      pos_.undo(m);
      if (r != Outcome::NONE) return r;
    }
    auto& slot = failed_[key];
    slot = std::max(slot, remaining);
    return Outcome::NONE;
  }

  TPos pos_;
  VcfLimits limits_;
  Color attacker_, defender_;
  uint64_t nodes_ = 0;
  bool depth_limited_ = false;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
  std::unordered_map<uint64_t, int> failed_;
  std::vector<Point> pv_;
};

}  // namespace

VcfResult solve_vcf(const TPos& pos, const VcfLimits& limits) { return VcfSearch(pos, limits).run(); }

}  // namespace dz::tactics
