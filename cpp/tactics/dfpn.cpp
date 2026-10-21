#include "tactics/dfpn.h"

#include <algorithm>
#include <chrono>

#include "tactics/tprof.h"

namespace dz::tactics {
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
    DZ_TPROF_SCOPE(TT_LOOKUP);
    auto it = tt_.find(key);
    if (it != tt_.end()) DZ_TPROF_COUNT(TT_HIT);
    return it == tt_.end() ? Entry{} : it->second;
  }

  // Defender move d loses by force: after d the defender has no five point and the attacker has
  // a legal move creating two five points. Checked on the real board (see dfpn.h).
  bool refuted(Point d, const std::vector<Point>& double_threats) {
    DZ_TPROF_SCOPE(REFUTED);
    DZ_TPROF_COUNT(PLAY_UNDO);
    Board& b = pos_.board();
    const Rules& rules = pos_.rules();
    play(d);
    bool result = false;
    std::vector<Point>& buf = refute_buf_;
    bool defender_five = false;
    if (d != kNoPoint) {
      five_points_near(b, d, defender_, rules, buf);
      defender_five = !buf.empty();
    }
    if (!defender_five) {
      for (Point e : double_threats) {
        if (e == d || !is_legal_move(b, e, attacker_, rules)) continue;
        b.place(e, attacker_);
        five_points_near(b, e, attacker_, rules, buf);
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

  // Exact equivalent of refuted(d) for a stone d (not pass) under the default black options, from
  // per-AND-node data: dt_info_[i] = attacker double-threat move e and its five points after e on the
  // current board. Why it is exact:
  //  * Defender five after d: new_five_points() skips every direction whose 9-cell window holds
  //    fewer than 3 own stones, so without such a window there is no five point; otherwise the
  //    original check is run unchanged.
  //  * Attacker five points of e after d: a defender stone never extends or shortens attacker runs
  //    (neighbours() stops at any non-own cell, EMPTY or opponent alike) and does not change the
  //    9-cell own-stone windows, so the only effect of d is to occupy one of e's five points.
  //    Hence count = |five points of e before d| - [d is one of them].
  //  * Attacker legality of e after d is checked exactly (white: always legal on an empty point;
  //    black: black_verdict_cells on a copy of the board with d placed), because recursive
  //    false-forbidden judgement has no locality bound.
  // The existence test is order independent, so cheap conditions are evaluated first.
  struct DtInfo {
    Point e;
    int n;
    Point fp[16];
  };
  static bool own_window(const uint8_t* c, int idx, uint8_t own) {
    for (int dir = 0; dir < kNumDirections; ++dir) {
      const int step = Board::kStep[dir];
      int n = 0;
      for (int k = 1; k <= 4; ++k) n += (c[idx + k * step] == own) + (c[idx - k * step] == own);
      if (n >= 3) return true;
    }
    return false;
  }
  bool prepare_fast_refute(const std::vector<Point>& double_threats) {
    Board& b = pos_.board();
    const Rules& rules = pos_.rules();
    if (!rules.black.exact_five_precedence || !rules.black.five_beats_overline_other_dir) return false;
    dt_info_.clear();
    for (Point e : double_threats) {
      b.place(e, attacker_);
      five_points_near(b, e, attacker_, rules, refute_buf_);
      b.remove(e);
      if (refute_buf_.size() > 16) return false;
      DtInfo t{e, static_cast<int>(refute_buf_.size()), {}};
      std::copy(refute_buf_.begin(), refute_buf_.end(), t.fp);
      dt_info_.push_back(t);
    }
    if (attacker_ == Color::BLACK) scratch_ = b.cells();
    return true;
  }
  bool refuted_fast(Point d) {
    DZ_TPROF_SCOPE(REFUTED);
    Board& b = pos_.board();
    const Rules& rules = pos_.rules();
    const int di = Board::index_of(d);
    if (own_window(b.cells().data(), di, cell_of(defender_))) {
      b.place(d, defender_);
      five_points_near(b, d, defender_, rules, refute_buf_);
      b.remove(d);
      if (!refute_buf_.empty()) return false;
    }
    for (const DtInfo& t : dt_info_) {
      if (t.e == d) continue;
      int n = t.n;
      for (int k = 0; k < t.n; ++k) n -= t.fp[k] == d;
      if (n < 2) continue;
      if (attacker_ == Color::BLACK) {
        scratch_[di] = cell_of(defender_);
        const bool forbidden = is_forbidden(black_verdict_cells(scratch_, Board::index_of(t.e), rules.black));
        scratch_[di] = EMPTY;
        if (forbidden) continue;
      }
      return true;
    }
    return false;
  }

  Gen generate(std::vector<Point>& moves, Point& win_move, int depth, bool& depth_cut) {
    Board& b = pos_.board();
    const Rules& rules = pos_.rules();
    std::vector<Point> buf;
    moves.clear();
    if (is_or()) {
      DZ_TPROF_SCOPE(DF_GENERATE_OR);
      {
        DZ_TPROF_SCOPE(TERMINAL_CHECK);
        five_points(b, attacker_, rules, buf);
      }
      if (!buf.empty()) {
        win_move = buf.front();
        return Gen::PROVEN;
      }
      if (depth >= limits_.max_depth) {
        depth_cut = true;
        return Gen::DISPROVEN;
      }
      {
        DZ_TPROF_SCOPE(OPP_FIVE_SCAN);
        five_points(b, defender_, rules, buf);
      }
      if (buf.size() >= 2) return Gen::DISPROVEN;
      if (buf.size() == 1) {
        const Point p = buf.front();
        if (!is_legal_move(b, p, attacker_, rules)) return Gen::DISPROVEN;
        if (limits_.fours_only) {
          b.place(p, attacker_);
          std::vector<Point> f;
          five_points_near(b, p, attacker_, rules, f);
          b.remove(p);
          if (f.empty()) return Gen::DISPROVEN;
        }
        moves.push_back(p);
        return Gen::EXPAND;
      }
      if (limits_.fours_only) four_moves(b, attacker_, rules, moves);
      else threat_moves(b, attacker_, rules, moves);
      return moves.empty() ? Gen::DISPROVEN : Gen::EXPAND;
    }

    // AND node.
    DZ_TPROF_SCOPE(DF_GENERATE_AND);
    {
      DZ_TPROF_SCOPE(TERMINAL_CHECK);
      five_points(b, defender_, rules, buf);
    }
    if (!buf.empty()) return Gen::DISPROVEN;  // defender wins first
    {
      DZ_TPROF_SCOPE(OPP_FIVE_SCAN);
      five_points(b, attacker_, rules, buf);
    }
    if (buf.size() >= 2) return Gen::PROVEN;
    if (buf.size() == 1) {
      const Point c = buf.front();
      if (!is_legal_move(b, c, defender_, rules)) return Gen::PROVEN;  // forbidden induction
      moves.push_back(c);
      return Gen::EXPAND;
    }
    if (limits_.fours_only) return Gen::DISPROVEN;
    std::vector<Point>& double_threats = dt_buf_;
    double_threat_moves(b, attacker_, rules, double_threats);
    int legal = 0;
    if (limits_.lazy_legality) {
      // Same move set and order as the eager loop below: a move is kept iff it is legal and not
      // refuted, so legality is only checked for the (few) unrefuted empty points. refuted() on an
      // illegal point is harmless (play/undo restore everything) and its answer is discarded.
      const bool fast = limits_.fast_refute && prepare_fast_refute(double_threats);
      for (int i = 0; i < kNumPoints; ++i) {
        const Point d = static_cast<Point>(i);
        if (!b.empty_at(d) || (fast ? refuted_fast(d) : refuted(d, double_threats))) continue;
        bool legal_d;
        {
          DZ_TPROF_SCOPE(DEFENDER_LEGAL);
          legal_d = is_legal_move(b, d, defender_, rules);
        }
        if (legal_d) moves.push_back(d);
      }
      legal = moves.empty() ? 0 : 1;
      for (int i = 0; legal == 0 && i < kNumPoints; ++i) {  // only when every empty point is refuted
        if (is_legal_move(b, static_cast<Point>(i), defender_, rules)) legal = 1;
      }
    } else {
      for (int i = 0; i < kNumPoints; ++i) {
        const Point d = static_cast<Point>(i);
        bool legal_d;
        {
          DZ_TPROF_SCOPE(DEFENDER_LEGAL);
          legal_d = is_legal_move(b, d, defender_, rules);
        }
        if (!legal_d) continue;
        ++legal;
        if (!refuted(d, double_threats)) moves.push_back(d);
      }
    }
    if (rules.pass_allowed) {
      ++legal;
      if (!refuted(kNoPoint, double_threats)) moves.push_back(kNoPoint);
    }
    if (legal == 0) return Gen::DISPROVEN;  // no move at all: not an attacker win (draw rules)
    return moves.empty() ? Gen::PROVEN : Gen::EXPAND;
  }

  void mid(uint32_t thpn, uint32_t thdn, int depth) {
    DZ_TPROF_SCOPE(DF_NODE);
    const uint64_t key = pos_.key();
    if (++nodes_ > limits_.max_nodes ||
        (limits_.max_ms > 0 && (nodes_ & 63) == 0 &&
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count() >
             limits_.max_ms)) {
      abort_ = true;
      return;
    }
    std::vector<Point> local_moves;
    const std::vector<Point>* mv = &local_moves;
    Point win = kNoPoint;
    bool depth_cut = false;
    Gen g;
    // Generation cache: generate() depends on the position (key) and, at OR nodes only, on the depth
    // cut, which the original code checks right after the attacker's immediate-win test. A cached
    // OR entry that is not an immediate win is therefore re-cut here at the depth limit, and results
    // of a depth-cut generation are never cached, so the cache returns exactly what generate() would.
    const auto cached = limits_.gen_cache ? gen_cache_.find(key) : gen_cache_.end();
    if (cached != gen_cache_.end()) {
      const GenEntry& c = cached->second;
      if (is_or() && c.g != Gen::PROVEN && depth >= limits_.max_depth) {
        g = Gen::DISPROVEN;
        depth_cut = true;
      } else {
        g = c.g;
        win = c.win;
        mv = &c.moves;
      }
    } else {
      g = generate(local_moves, win, depth, depth_cut);
      if (limits_.gen_cache && !depth_cut) {
        mv = &gen_cache_.emplace(key, GenEntry{g, win, std::move(local_moves)}).first->second.moves;
      }
    }
    const std::vector<Point>& moves = *mv;
    if (g == Gen::PROVEN) {
      DZ_TPROF_SCOPE(TT_STORE);
      tt_[key] = {0, kInf, win};
      return;
    }
    if (g == Gen::DISPROVEN) {
      DZ_TPROF_SCOPE(TT_STORE);
      if (!depth_cut) tt_[key] = {kInf, 0, kNoPoint};
      return;
    }

    struct Child {
      Point move;
      uint64_t key;
      uint32_t pn, dn;
    };
    std::vector<Child> ch;
    DZ_TPROF_ADD(CANDIDATES, moves.size());
    ch.reserve(moves.size());
    for (Point m : moves) {
      DZ_TPROF_SCOPE(CHILD_KEYS);
      DZ_TPROF_COUNT(PLAY_UNDO);
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
      DZ_TPROF_COUNT(CHILD_SELECT);
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
      DZ_TPROF_COUNT(PLAY_UNDO);
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
      five_points(pos_.board(), attacker_, pos_.rules(), f);
      if (!f.empty()) pv.push_back(f.front());
    }
    for (auto it = played.rbegin(); it != played.rend(); ++it) undo(*it);
  }

  struct GenEntry {
    Gen g;
    Point win;
    std::vector<Point> moves;
  };

  TPos pos_;
  DfpnLimits limits_;
  Color attacker_, defender_;
  std::unordered_map<uint64_t, Entry> tt_;
  std::unordered_map<uint64_t, GenEntry> gen_cache_;  // node-based: references stay valid on rehash
  std::vector<Point> refute_buf_, dt_buf_;           // scratch (generate/refuted are not re-entrant)
  std::vector<DtInfo> dt_info_;                      // fast_refute data of the AND node being generated
  Board::Cells scratch_{};                           // board copy for black legality in refuted_fast
  uint64_t nodes_ = 0;
  bool abort_ = false;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

}  // namespace

DfpnResult solve_dfpn(const TPos& pos, DfpnMode mode, const DfpnLimits& limits) {
  return Dfpn(pos, mode, limits).run(mode);
}

}  // namespace dz::tactics
