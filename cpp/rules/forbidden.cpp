#include "rules/forbidden.h"

#include <array>
#include <sstream>

namespace dz {
namespace {

constexpr int kMaxFoursPerDir = 8;
constexpr int kMaxExtPerDir = 6;
constexpr size_t kMaxRecursionLog = 20000;

inline int run_len(const uint8_t* c, int idx, int step, uint8_t own) {
  int n = 1;
  for (int i = idx + step; c[i] == own; i += step) ++n;
  for (int i = idx - step; c[i] == own; i -= step) ++n;
  return n;
}

// Fours through idx (own stone already at idx) in one direction.
// A four is identified by its stone set S (RULES.md §3.3); completions sharing S are one four.
struct DirFours {
  int count = 0;
  std::array<uint16_t, kMaxFoursPerDir> mask{};  // bit (offset + 5) for offsets in [-5, 5]
  std::array<std::array<int, 2>, kMaxFoursPerDir> comp{};
  std::array<int, kMaxFoursPerDir> ncomp{};
};

// exact: completion must give a run of exactly 5 (black); otherwise >= 5 (white).
// For white, runs longer than the [-5, 5] window are clipped in the mask; white fours are only
// used for tactics, never for forbidden-move judgement.
inline void fours_in_dir(const uint8_t* c, int idx, int step, uint8_t own, bool exact, DirFours& out) {
  out.count = 0;
  for (int k = -4; k <= 4; ++k) {
    if (k == 0) continue;
    const int e = idx + k * step;
    if (c[e] != EMPTY) continue;
    int l = 0;
    for (int i = e - step; c[i] == own; i -= step) ++l;
    int r = 0;
    for (int i = e + step; c[i] == own; i += step) ++r;
    const int len = 1 + l + r;
    if (exact ? len != 5 : len < 5) continue;
    if (k - l > 0 || k + r < 0) continue;  // the five must contain idx
    uint16_t mask = 0;
    const int lo = k - l < -5 ? -5 : k - l;
    const int hi = k + r > 5 ? 5 : k + r;
    for (int o = lo; o <= hi; ++o) {
      if (o != k) mask = static_cast<uint16_t>(mask | (1u << (o + 5)));
    }
    int slot = -1;
    for (int s = 0; s < out.count; ++s) {
      if (out.mask[s] == mask) { slot = s; break; }
    }
    if (slot < 0) {
      if (out.count >= kMaxFoursPerDir) continue;  // unreachable on 15x15; defensive
      slot = out.count++;
      out.mask[slot] = mask;
      out.ncomp[slot] = 0;
    }
    if (out.ncomp[slot] < 2) out.comp[slot][out.ncomp[slot]++] = e;
  }
}

// With own stones at idx and e (e at offset k from idx): is there a straight four in this
// direction containing both? Straight four = exactly 4 contiguous with two completion points.
inline bool straight_four_through(const uint8_t* c, int e, int k, int step, uint8_t own, bool exact) {
  int l = 0;
  for (int i = e - step; c[i] == own; i -= step) ++l;
  int r = 0;
  for (int i = e + step; c[i] == own; i += step) ++r;
  if (1 + l + r != 4) return false;
  if (k - l > 0 || k + r < 0) return false;  // must contain idx (offset 0)
  const int a = e - (l + 1) * step;
  const int b = e + (r + 1) * step;
  if (c[a] != EMPTY || c[b] != EMPTY) return false;
  if (exact && (c[a - step] == own || c[b + step] == own)) return false;  // completion = overline
  return true;
}

// Collects straight-four extension points per direction (apparent threes) for own stone at idx.
// Directions that already hold a four or a run >= 5 are skipped (ASSUMPTIONS A-R3).
struct Extensions {
  std::array<std::array<int, kMaxExtPerDir>, kNumDirections> pt{};
  std::array<int, kNumDirections> n{};
  int dirs = 0;
};

inline void collect_extensions(uint8_t* c, int idx, uint8_t own, bool exact,
                               const std::array<bool, kNumDirections>& skip, Extensions& out) {
  out.dirs = 0;
  for (int d = 0; d < kNumDirections; ++d) {
    out.n[d] = 0;
    if (skip[d]) continue;
    const int step = Board::kStep[d];
    for (int k = -3; k <= 3; ++k) {
      if (k == 0) continue;
      const int e = idx + k * step;
      if (c[e] != EMPTY) continue;
      c[e] = own;
      const bool s = straight_four_through(c, e, k, step, own, exact);
      c[e] = EMPTY;
      if (s && out.n[d] < kMaxExtPerDir) out.pt[d][out.n[d]++] = e;
    }
    if (out.n[d] > 0) ++out.dirs;
  }
}

template <bool kExplain>
class BlackChecker {
 public:
  BlackChecker(uint8_t* cells, const BlackRuleOptions& opt, ForbiddenInfo* info)
      : c_(cells), opt_(opt), info_(info) {}

  MoveVerdict verdict(int idx, int depth) {
    c_[idx] = BLACK_STONE;
    const bool top = kExplain && depth == 0;

    bool five = false, over = false;
    std::array<bool, kNumDirections> skip{};
    for (int d = 0; d < kNumDirections; ++d) {
      const int rl = run_len(c_, idx, Board::kStep[d], BLACK_STONE);
      if (rl == 5) {
        five = true;
        if (top) info_->five_dirs.push_back(d);
      } else if (rl > 5) {
        over = true;
        if (top) info_->overline_dirs.push_back(d);
      }
      skip[d] = rl >= 5;
    }

    bool decided = false;
    MoveVerdict result = MoveVerdict::LEGAL;
    if (opt_.exact_five_precedence) {
      if (five && (opt_.five_beats_overline_other_dir || !over)) {
        result = MoveVerdict::WIN_FIVE;
        decided = true;
      } else if (over) {
        result = MoveVerdict::FORBIDDEN_OVERLINE;
        decided = true;
      }
    } else if (over) {
      result = MoveVerdict::FORBIDDEN_OVERLINE;
      decided = true;
    }
    if (decided && !top) return finish(idx, result);

    int total_fours = 0;
    for (int d = 0; d < kNumDirections; ++d) {
      DirFours df;
      fours_in_dir(c_, idx, Board::kStep[d], BLACK_STONE, /*exact=*/true, df);
      total_fours += df.count;
      if (df.count > 0) skip[d] = true;
      if (top) record_fours(idx, d, df);
    }
    if (!decided && total_fours >= 2) {
      result = MoveVerdict::FORBIDDEN_DOUBLE_FOUR;
      decided = true;
      if (!top) return finish(idx, result);
    }

    Extensions ext;
    collect_extensions(c_, idx, BLACK_STONE, /*exact=*/true, skip, ext);
    const MoveVerdict no33 =
        (!opt_.exact_five_precedence && five) ? MoveVerdict::WIN_FIVE : MoveVerdict::LEGAL;
    if (ext.dirs < 2 && !top) return finish(idx, decided ? result : no33);

    int real = 0;
    int remaining = ext.dirs;
    for (int d = 0; d < kNumDirections; ++d) {
      if (ext.n[d] == 0) continue;
      bool is_real = false;
      ThreeInfo ti;
      ti.dir = d;
      for (int j = 0; j < ext.n[d]; ++j) {
        const int e = ext.pt[d][j];
        size_t slot = 0;
        bool logged = false;
        if constexpr (kExplain) logged = reserve_step(e, idx, depth + 1, slot);  // pre-order trace
        const MoveVerdict ev = verdict(e, depth + 1);  // RIF 9.3: legality of the extension
        if constexpr (kExplain) {
          if (logged) info_->recursion[slot].verdict = ev;
        }
        if (top) ti.extensions.push_back({Board::point_of_index(e), ev});
        if (ev == MoveVerdict::LEGAL) {
          is_real = true;
          if (!top) break;
        }
      }
      if (top) {
        ti.real = is_real;
        info_->threes.push_back(std::move(ti));
      }
      if (is_real) ++real;
      --remaining;
      if (!top && (real >= 2 || real + remaining < 2)) break;
    }
    if (!decided) result = real >= 2 ? MoveVerdict::FORBIDDEN_DOUBLE_THREE : no33;
    return finish(idx, result);
  }

 private:
  MoveVerdict finish(int idx, MoveVerdict v) {
    c_[idx] = EMPTY;
    return v;
  }

  void record_fours(int idx, int d, const DirFours& df) {
    for (int s = 0; s < df.count; ++s) {
      FourInfo fi;
      fi.dir = d;
      for (int o = -5; o <= 5; ++o) {
        if (df.mask[s] & (1u << (o + 5))) {
          fi.stones.push_back(Board::point_of_index(idx + o * Board::kStep[d]));
        }
      }
      for (int j = 0; j < df.ncomp[s]; ++j) fi.completions.push_back(Board::point_of_index(df.comp[s][j]));
      info_->fours.push_back(std::move(fi));
    }
  }

  bool reserve_step(int e, int parent, int depth, size_t& slot) {
    if (info_->recursion.size() >= kMaxRecursionLog) {
      info_->recursion_truncated = true;
      return false;
    }
    slot = info_->recursion.size();
    info_->recursion.push_back(
        {Board::point_of_index(e), Board::point_of_index(parent), depth, MoveVerdict::LEGAL});
    return true;
  }

  uint8_t* c_;
  const BlackRuleOptions& opt_;
  ForbiddenInfo* info_;
};

}  // namespace

const char* verdict_name(MoveVerdict v) {
  switch (v) {
    case MoveVerdict::WIN_FIVE: return "WIN_FIVE";
    case MoveVerdict::LEGAL: return "LEGAL";
    case MoveVerdict::FORBIDDEN_OVERLINE: return "FORBIDDEN_OVERLINE";
    case MoveVerdict::FORBIDDEN_DOUBLE_FOUR: return "FORBIDDEN_DOUBLE_FOUR";
    case MoveVerdict::FORBIDDEN_DOUBLE_THREE: return "FORBIDDEN_DOUBLE_THREE";
  }
  return "?";
}

std::optional<MoveVerdict> parse_verdict(std::string_view s) {
  if (s == "WIN_FIVE" || s == "FIVE") return MoveVerdict::WIN_FIVE;
  if (s == "LEGAL") return MoveVerdict::LEGAL;
  if (s == "FORBIDDEN_OVERLINE" || s == "OVERLINE") return MoveVerdict::FORBIDDEN_OVERLINE;
  if (s == "FORBIDDEN_DOUBLE_FOUR" || s == "DOUBLE_FOUR") return MoveVerdict::FORBIDDEN_DOUBLE_FOUR;
  if (s == "FORBIDDEN_DOUBLE_THREE" || s == "DOUBLE_THREE") return MoveVerdict::FORBIDDEN_DOUBLE_THREE;
  return std::nullopt;
}

MoveVerdict black_verdict_cells(Board::Cells& cells, int idx, const BlackRuleOptions& opt) {
  BlackChecker<false> checker(cells.data(), opt, nullptr);
  return checker.verdict(idx, 0);
}

MoveVerdict black_verdict(const Board& board, Point p, const BlackRuleOptions& opt) {
  Board::Cells scratch = board.cells();
  return black_verdict_cells(scratch, Board::index_of(p), opt);
}

ForbiddenInfo explain_black_move(const Board& board, Point p, const BlackRuleOptions& opt) {
  ForbiddenInfo info;
  info.point = p;
  Board::Cells scratch = board.cells();
  BlackChecker<true> checker(scratch.data(), opt, &info);
  info.verdict = checker.verdict(Board::index_of(p), 0);
  return info;
}

int ForbiddenInfo::real_three_count() const {
  int n = 0;
  for (const auto& t : threes) n += t.real ? 1 : 0;
  return n;
}

std::string ForbiddenInfo::to_string() const {
  std::ostringstream os;
  os << "BLACK " << to_notation(point) << ": " << verdict_name(verdict) << "\n";
  for (int d : five_dirs) os << "  " << direction_name(d) << ": exact five\n";
  for (int d : overline_dirs) os << "  " << direction_name(d) << ": overline\n";
  for (const auto& f : fours) {
    os << "  " << direction_name(f.dir) << ": four {";
    for (size_t i = 0; i < f.stones.size(); ++i) os << (i ? "," : "") << to_notation(f.stones[i]);
    os << "} completion";
    for (Point c : f.completions) os << " " << to_notation(c);
    os << (f.completions.size() >= 2 ? " (straight four)" : "") << "\n";
  }
  for (const auto& t : threes) {
    os << "  " << direction_name(t.dir) << ": three;";
    for (const auto& e : t.extensions) {
      os << " extension " << to_notation(e.point) << " -> straight four (" << verdict_name(e.verdict) << ");";
    }
    os << " => " << (t.real ? "real" : "not real") << "\n";
  }
  os << "  fours=" << four_count() << " apparent_threes=" << apparent_three_count()
     << " real_threes=" << real_three_count() << "\n";
  if (!recursion.empty()) {
    os << "  recursion path:\n";
    for (const auto& s : recursion) {
      os << "    " << std::string(static_cast<size_t>(s.depth) * 2, ' ') << to_notation(s.point)
         << " (extends " << to_notation(s.parent) << ") = " << verdict_name(s.verdict) << "\n";
    }
    if (recursion_truncated) os << "    ... (truncated)\n";
  }
  return os.str();
}

bool makes_five(const Board& board, Point p, Color c) {
  const auto& cells = board.cells();
  const int idx = Board::index_of(p);
  const uint8_t own = cell_of(c);
  Board::Cells scratch = cells;
  scratch[idx] = own;
  for (int d = 0; d < kNumDirections; ++d) {
    const int rl = run_len(scratch.data(), idx, Board::kStep[d], own);
    if (c == Color::BLACK ? rl == 5 : rl >= 5) return true;
  }
  return false;
}

bool makes_overline(const Board& board, Point p, Color c) {
  Board::Cells scratch = board.cells();
  const int idx = Board::index_of(p);
  const uint8_t own = cell_of(c);
  scratch[idx] = own;
  for (int d = 0; d < kNumDirections; ++d) {
    if (run_len(scratch.data(), idx, Board::kStep[d], own) >= 6) return true;
  }
  return false;
}

int count_fours(const Board& board, Point p, Color c) {
  Board::Cells scratch = board.cells();
  const int idx = Board::index_of(p);
  const uint8_t own = cell_of(c);
  scratch[idx] = own;
  int total = 0;
  for (int d = 0; d < kNumDirections; ++d) {
    DirFours df;
    fours_in_dir(scratch.data(), idx, Board::kStep[d], own, c == Color::BLACK, df);
    total += df.count;
  }
  return total;
}

int count_threes(const Board& board, Point p, Color c, const BlackRuleOptions& opt) {
  if (c == Color::BLACK) return explain_black_move(board, p, opt).real_three_count();
  Board::Cells scratch = board.cells();
  const int idx = Board::index_of(p);
  scratch[idx] = WHITE_STONE;
  std::array<bool, kNumDirections> skip{};
  for (int d = 0; d < kNumDirections; ++d) {
    const int step = Board::kStep[d];
    DirFours df;
    fours_in_dir(scratch.data(), idx, step, WHITE_STONE, false, df);
    skip[d] = df.count > 0 || run_len(scratch.data(), idx, step, WHITE_STONE) >= 5;
  }
  Extensions ext;
  collect_extensions(scratch.data(), idx, WHITE_STONE, /*exact=*/false, skip, ext);
  return ext.dirs;
}

}  // namespace dz
