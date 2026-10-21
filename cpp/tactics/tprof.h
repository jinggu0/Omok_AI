// Tactical-solver profiling hooks (PROGRESS Phase 8 E3b). Active only when compiled with DZ_TPROF
// (the separate dojozero_core_tprof library used by dojozero_tprof); otherwise every macro is empty.
// Each section records inclusive wall time and call count; nested sections are reported inclusive.
#pragma once

#include <cstdint>

#ifdef DZ_TPROF
#include <chrono>
#endif

namespace dz::tactics::tprof {

enum Section : int {
  VCF_NODE,          // one VCF OR node (inclusive of children)
  DF_NODE,           // one df-pn mid() call (inclusive of children)
  DF_GENERATE_OR,    // move generation at a df-pn OR node
  DF_GENERATE_AND,   // move generation at a df-pn AND node
  TERMINAL_CHECK,    // side-to-move immediate-win scan at node entry
  OPP_FIVE_SCAN,     // opponent five-point scan (forced defense detection)
  FIVE_POINTS,       // five_points() full-board scan (all callers)
  FIVE_POINTS_NEAR,  // five_points_near() local scan
  FOUR_MOVES,        // four_moves() candidate generation (incl. ordering)
  DOUBLE_THREATS,    // double_threat_moves()
  THREAT_MOVES,      // threat_moves() (fours + threes)
  COUNT_THREES,      // count_threes() inside threat_moves
  CANDIDATE_SORT,    // stable_sort of candidates by five-point count
  DEFENDER_LEGAL,    // is_legal_move() over all points at AND nodes
  REFUTED,           // refuted(d): play, defender five check, attacker double threat search
  BLACK_VERDICT,     // black forbidden verdict (all callers inside the solver)
  PLAY_UNDO,         // TPos::play + undo pairs (Board place/remove incl. 8 hashes)
  TT_LOOKUP,         // TT / failed-map lookups
  TT_STORE,          // TT / failed-map stores
  CHILD_KEYS,        // df-pn: play/undo per child to compute its key + lookup
  CHILD_SELECT,      // df-pn: most-proving child selection loop
  TT_HIT,            // count only: lookups that found an entry
  CANDIDATES,        // count only: sum of generated candidate moves (per node: / node calls)
  kSections
};

inline const char* const kNames[kSections] = {
    "vcf_node", "dfpn_node", "dfpn_generate_or", "dfpn_generate_and", "terminal_check", "opp_five_scan",
    "five_points", "five_points_near", "four_moves", "double_threat_moves", "threat_moves", "count_threes",
    "candidate_sort", "defender_legal_loop", "refuted", "black_verdict", "play_undo", "tt_lookup", "tt_store",
    "child_keys", "child_select", "tt_hit", "candidates"};

struct Counters {
  double ns[kSections] = {};
  uint64_t calls[kSections] = {};
};
Counters& counters();  // defined in tprof.cpp (thread_local)
void reset();

#ifdef DZ_TPROF
struct Scope {
  Section s;
  std::chrono::steady_clock::time_point t0;
  explicit Scope(Section sec) : s(sec), t0(std::chrono::steady_clock::now()) {}
  ~Scope() {
    auto& c = counters();
    c.ns[s] += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    ++c.calls[s];
  }
};
#define DZ_TPROF_CAT2(a, b) a##b
#define DZ_TPROF_CAT(a, b) DZ_TPROF_CAT2(a, b)
#define DZ_TPROF_SCOPE(sec) ::dz::tactics::tprof::Scope DZ_TPROF_CAT(dz_tprof_, __LINE__)(::dz::tactics::tprof::sec)
#define DZ_TPROF_COUNT(sec) (++::dz::tactics::tprof::counters().calls[::dz::tactics::tprof::sec])
#define DZ_TPROF_ADD(sec, n) (::dz::tactics::tprof::counters().calls[::dz::tactics::tprof::sec] += (n))
#else
#define DZ_TPROF_SCOPE(sec)
#define DZ_TPROF_COUNT(sec)
#define DZ_TPROF_ADD(sec, n)
#endif

}  // namespace dz::tactics::tprof
