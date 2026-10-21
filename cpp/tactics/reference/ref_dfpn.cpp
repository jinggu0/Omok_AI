// FROZEN reference copy of cpp/tactics/dfpn.cpp at ee1d011 (differential oracle).
#include "tactics/reference/ref.h"

#include <algorithm>
#include <chrono>

namespace dz::tactics::ref {
namespace {

constexpr uint32_t kInf = 1u << 30;
inline uint32_t sat_add(uint32_t a, uint32_t b) {
  return static_cast<uint32_t>(std::min<uint64_t>(kInf, uint64_t{a} + b));
}

struct Entry {
  uint32_t pn = 1, dn = 1;
  Point best = kNoPoint;  // proven OR node: winning move
};

enum class Gen { EXPAND, PROVEN, DISPROVEN };

class Dfpn {
 public:
  Dfpn(const TPos& pos, DfpnMode mode, const DfpnLimits& limits)
      : pos_(pos),
        limits_(limits),
        attacker_(mode == DfpnMode::PROVE_WIN ? pos.to_move() : opposite(pos.to_move())),
        defender_(opposite(attacker_)) {}

  DfpnResult run(DfpnMode mode) {
    mid(kInf, kInf, 0);
    const Entry root = lookup(pos_.key());
    DfpnResult out;
    out.nodes = nodes_;
    out.budget_exhausted = abort_;
    if (root.pn == 0) {
      out.result = mode == DfpnMode::PROVE_WIN ? ProofResult::WIN : ProofResult::LOSS;
      if (mode == DfpnMode::PROVE_WIN) out.best = root.best;
      extract_pv(out.pv);
      for (const auto& [key, e] : tt_) {
        if (e.pn == 0 && e.best != kNoPoint) out.proof_moves.emplace(key, e.best);
      }
    }
    return out;
  }

 private:
  bool is_or() const { return pos_.to_move() == attacker_; }
  void play(Point m) { m == kNoPoint ? pos_.pass() : pos_.play(m); }
  void undo(Point m) { m == kNoPoint ? pos_.undo_pass() : pos_.undo(m); }
  Entry lookup(uint64_t key) const {
    auto it = tt_.find(key);
    return it == tt_.end() ? Entry{} : it->second;
  }

  // Defender move d loses by force: after d the defender has no five point and the attacker has
  // a legal move creating two five points. Checked on the real board (see dfpn.h).
  bool refuted(Point d, const std::vector<Point>& double_threats) {
    Board& b = pos_.board();
    const Rules& rules = pos_.rules();
    play(d);
    bool result = false;
    std::vector<Point> buf;
    bool defender_five = false;
    if (d != kNoPoint) {
      ref::five_points_near(b, d, defender_, rules, buf);
      defender_five = !buf.empty();
    }
    if (!defender_five) {
      for (Point e : double_threats) {
        if (e == d || !ref::is_legal_move(b, e, attacker_, rules)) continue;
        b.place(e, attacker_);
        ref::five_points_near(b, e, attacker_, rules, buf);
        b.remove(e);
        if (buf.size() >= 2) {
          result = true;
          break;
        }
      }
    }
    undo(d);
    return result;
  }

  Gen generate(std::vector<Point>& moves, Point& win_move, int depth, bool& depth_cut) {
    Board& b = pos_.board();
    const Rules& rules = pos_.rules();
    std::vector<Point> buf;
    moves.clear();
    if (is_or()) {
      ref::five_points(b, attacker_, rules, buf);
      if (!buf.empty()) {
        win_move = buf.front();
        return Gen::PROVEN;
      }
      if (depth >= limits_.max_depth) {
        depth_cut = true;
        return Gen::DISPROVEN;
      }
      ref::five_points(b, defender_, rules, buf);
      if (buf.size() >= 2) return Gen::DISPROVEN;
      if (buf.size() == 1) {
        const Point p = buf.front();
        if (!ref::is_legal_move(b, p, attacker_, rules)) return Gen::DISPROVEN;
        if (limits_.fours_only) {
          b.place(p, attacker_);
          std::vector<Point> f;
          ref::five_points_near(b, p, attacker_, rules, f);
          b.remove(p);
          if (f.empty()) return Gen::DISPROVEN;
        }
        moves.push_back(p);
        return Gen::EXPAND;
      }
      if (limits_.fours_only) ref::four_moves(b, attacker_, rules, moves);
      else ref::threat_moves(b, attacker_, rules, moves);
      return moves.empty() ? Gen::DISPROVEN : Gen::EXPAND;
    }

    // AND node.
    ref::five_points(b, defender_, rules, buf);
    if (!buf.empty()) return Gen::DISPROVEN;  // defender wins first
    ref::five_points(b, attacker_, rules, buf);
    if (buf.size() >= 2) return Gen::PROVEN;
    if (buf.size() == 1) {
      const Point c = buf.front();
      if (!ref::is_legal_move(b, c, defender_, rules)) return Gen::PROVEN;  // forbidden induction
      moves.push_back(c);
      return Gen::EXPAND;
    }
    if (limits_.fours_only) return Gen::DISPROVEN;
    std::vector<Point> double_threats;
    ref::double_threat_moves(b, attacker_, rules, double_threats);
    int legal = 0;
    for (int i = 0; i < kNumPoints; ++i) {
      const Point d = static_cast<Point>(i);
      if (!ref::is_legal_move(b, d, defender_, rules)) continue;
      ++legal;
      if (!refuted(d, double_threats)) moves.push_back(d);
    }
    if (rules.pass_allowed) {
      ++legal;
      if (!refuted(kNoPoint, double_threats)) moves.push_back(kNoPoint);
    }
    if (legal == 0) return Gen::DISPROVEN;  // no move at all: not an attacker win (draw rules)
    return moves.empty() ? Gen::PROVEN : Gen::EXPAND;
  }

  void mid(uint32_t thpn, uint32_t thdn, int depth) {
    const uint64_t key = pos_.key();
    if (++nodes_ > limits_.max_nodes ||
        (limits_.max_ms > 0 && (nodes_ & 63) == 0 &&
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count() >
             limits_.max_ms)) {
      abort_ = true;
      return;
    }
    std::vector<Point> moves;
    Point win = kNoPoint;
    bool depth_cut = false;
    const Gen g = generate(moves, win, depth, depth_cut);
    if (g == Gen::PROVEN) {
      tt_[key] = {0, kInf, win};
      return;
    }
    if (g == Gen::DISPROVEN) {
      if (!depth_cut) tt_[key] = {kInf, 0, kNoPoint};
      return;
    }

    struct Child {
      Point move;
      uint64_t key;
      uint32_t pn, dn;
    };
    std::vector<Child> ch;
    ch.reserve(moves.size());
    for (Point m : moves) {
      play(m);
      const uint64_t k = pos_.key();
      undo(m);
      const Entry e = lookup(k);
      ch.push_back({m, k, e.pn, e.dn});
    }
    const bool or_node = is_or();
    while (true) {
      uint32_t pn, dn;
      if (or_node) {
        pn = kInf;
        dn = 0;
        for (const auto& c : ch) { pn = std::min(pn, c.pn); dn = sat_add(dn, c.dn); }
      } else {
        pn = 0;
        dn = kInf;
        for (const auto& c : ch) { pn = sat_add(pn, c.pn); dn = std::min(dn, c.dn); }
      }
      if (pn >= thpn || dn >= thdn || abort_) {
        Entry e{pn, dn, kNoPoint};
        if (or_node && pn == 0) {
          for (const auto& c : ch) if (c.pn == 0) { e.best = c.move; break; }
        }
        tt_[key] = e;
        return;
      }
      // Most-proving child and the runner-up value.
      size_t best = 0;
      uint32_t second = kInf;
      for (size_t i = 0; i < ch.size(); ++i) {
        const uint32_t v = or_node ? ch[i].pn : ch[i].dn;
        const uint32_t bv = or_node ? ch[best].pn : ch[best].dn;
        if (i == 0) continue;
        if (v < bv) { second = bv; best = i; }
        else if (v < second) { second = v; }
      }
      if (ch.size() == 1) second = kInf;
      uint32_t c_thpn, c_thdn;
      if (or_node) {
        c_thpn = static_cast<uint32_t>(std::min<uint64_t>(thpn, uint64_t{second} + 1));
        c_thdn = thdn - dn + ch[best].dn;
      } else {
        c_thdn = static_cast<uint32_t>(std::min<uint64_t>(thdn, uint64_t{second} + 1));
        c_thpn = thpn - pn + ch[best].pn;
      }
      play(ch[best].move);
      mid(c_thpn, c_thdn, depth + 1);
      undo(ch[best].move);
      const Entry e = lookup(ch[best].key);
      ch[best].pn = e.pn;
      ch[best].dn = e.dn;
      if (abort_) {
        // Store the current (unfinished) numbers so the root sees an unproven value.
        tt_[key] = {std::max<uint32_t>(pn, 1), std::max<uint32_t>(dn, 1), kNoPoint};
        return;
      }
    }
  }

  void extract_pv(std::vector<Point>& pv) {
    std::vector<Point> played;
    std::vector<Point> moves;
    for (int ply = 0; ply < 80; ++ply) {
      const Entry e = lookup(pos_.key());
      if (e.pn != 0) break;
      Point next = kNoPoint;
      if (is_or()) {
        if (e.best == kNoPoint) break;  // proven by an immediate win found at generation
        next = e.best;
      } else {
        Point win = kNoPoint;
        bool cut = false;
        if (generate(moves, win, 0, cut) != Gen::EXPAND) break;
        bool found = false;
        for (Point m : moves) {
          play(m);
          const bool proven = lookup(pos_.key()).pn == 0;
          undo(m);
          if (proven) { next = m; found = true; break; }
        }
        if (!found) break;
      }
      pv.push_back(next);
      play(next);
      played.push_back(next);
    }
    // Final immediate win of the attacker, if the line ends at an attacker node.
    if (is_or()) {
      std::vector<Point> f;
      ref::five_points(pos_.board(), attacker_, pos_.rules(), f);
      if (!f.empty()) pv.push_back(f.front());
    }
    for (auto it = played.rbegin(); it != played.rend(); ++it) undo(*it);
  }

  TPos pos_;
  DfpnLimits limits_;
  Color attacker_, defender_;
  std::unordered_map<uint64_t, Entry> tt_;
  uint64_t nodes_ = 0;
  bool abort_ = false;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

}  // namespace

DfpnResult solve_dfpn(const TPos& pos, DfpnMode mode, const DfpnLimits& limits) {
  return Dfpn(pos, mode, limits).run(mode);
}

}  // namespace dz::tactics::ref
