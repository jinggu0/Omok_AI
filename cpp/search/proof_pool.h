// Asynchronous tactical proofs for the MCTS tree (proof-aware MCTS, PROGRESS Phase 8 E3d / Case A).
//
// A process-wide pool of solver threads runs analyze_tactics on positions that search trees submit
// (the side to move is the attacker: WIN = the side to move has a forced win, LOSS = it cannot
// stop an immediate win). Trees never share nodes with the workers: a job carries a copy of the
// position and its result is posted to the submitting tree's mailbox, which the tree drains on its
// own thread. One pool serves every tree in the process (self-play runs hundreds of trees).
//
// ProofCache keeps exact results (WIN / LOSS: proofs, valid under any budget) separately from
// budget-limited attempts (UNKNOWN at n nodes: only "do not retry below n"). UNKNOWN is never
// stored or returned as a result.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "game/game_state.h"
#include "tactics/engine.h"
#include "tactics/verify.h"

namespace dz::search {

struct ProofOutcome {
  int32_t node = -1;
  tactics::ProofResult result = tactics::ProofResult::UNKNOWN;  // for the side to move
  uint64_t nodes = 0;
  uint64_t budget = 0;        // VCT node budget of this attempt
  bool from_cache = false;
  bool skipped = false;       // deadline passed before the job started
  Point move = kNoPoint;      // winning move of the side to move (WIN only; may be absent on cache hits)
};

// Results for one tree; shared with the pool so a job finishing after its tree is gone is harmless.
struct ProofMailbox {
  std::mutex mu;
  std::vector<ProofOutcome> done;
  bool closed = false;
};

struct ProofJob {
  std::shared_ptr<ProofMailbox> box;
  int32_t node = -1;
  GameState state;
  uint64_t vcf_nodes = 0, vct_nodes = 0;
  std::chrono::steady_clock::time_point deadline{};  // epoch value = no deadline (node budgets only)
};

class ProofCache {
 public:
  struct Entry {
    int8_t exact = 0;          // +1 side to move wins, -1 side to move loses, 0 none
    Point move = kNoPoint;     // winning move for exact == +1
    uint64_t tried_vct = 0;    // largest budget that ended UNKNOWN
  };
  static ProofCache& instance();
  static uint64_t key_of(const GameState& g);
  Entry get(uint64_t key);
  void put_exact(uint64_t key, int8_t exact, Point move = kNoPoint);
  void put_unknown(uint64_t key, uint64_t vct_budget);
  void clear();
  size_t size();

 private:
  std::mutex mu_;
  std::unordered_map<uint64_t, Entry> map_;
};

class ProofPool {
 public:
  // The pool grows to the largest thread count requested so far (never shrinks).
  static ProofPool& instance(int threads);
  void submit(ProofJob job);
  ~ProofPool();

 private:
  void ensure_threads(int n);
  void worker();
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<ProofJob> queue_;
  std::vector<std::thread> threads_;
  bool stop_ = false;
};

// Runs one job synchronously (used by the pool; exposed for tests).
ProofOutcome run_proof_job(const ProofJob& job);

}  // namespace dz::search
