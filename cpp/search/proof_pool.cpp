#include "search/proof_pool.h"

#include <algorithm>
#include <optional>

namespace dz::search {

// ---- ProofCache ----

ProofCache& ProofCache::instance() {
  static ProofCache cache;
  return cache;
}

uint64_t ProofCache::key_of(const GameState& g) {
  const auto& b = g.profile().black;
  const uint64_t rules = (b.exact_five_precedence ? 0x9E3779B97F4A7C15ull : 0) ^
                         (b.five_beats_overline_other_dir ? 0xC2B2AE3D27D4EB4Full : 0) ^
                         (g.profile().pass_allowed ? 0x165667B19E3779F9ull : 0);
  return tactics::position_key(g.board(), g.side_to_move()) ^ rules;
}

ProofCache::Entry ProofCache::get(uint64_t key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = map_.find(key);
  return it == map_.end() ? Entry{} : it->second;
}

void ProofCache::put_exact(uint64_t key, int8_t exact, Point move) {
  std::lock_guard<std::mutex> lk(mu_);
  if (map_.size() > 2'000'000) map_.clear();  // memory bound; proofs are recomputable
  Entry& e = map_[key];
  e.exact = exact;
  if (move != kNoPoint) e.move = move;
}

void ProofCache::put_unknown(uint64_t key, uint64_t vct_budget) {
  std::lock_guard<std::mutex> lk(mu_);
  if (map_.size() > 2'000'000) map_.clear();
  Entry& e = map_[key];
  e.tried_vct = std::max(e.tried_vct, vct_budget);
}

void ProofCache::clear() {
  std::lock_guard<std::mutex> lk(mu_);
  map_.clear();
}

size_t ProofCache::size() {
  std::lock_guard<std::mutex> lk(mu_);
  return map_.size();
}

// ---- jobs ----

ProofOutcome run_proof_job(const ProofJob& job) {
  ProofOutcome out;
  out.node = job.node;
  out.budget = job.vct_nodes;
  const bool timed = job.deadline.time_since_epoch().count() != 0;
  double ms = 0;
  if (timed) {
    ms = std::chrono::duration<double, std::milli>(job.deadline - std::chrono::steady_clock::now()).count();
    if (ms < 1.0) {
      out.skipped = true;
      return out;
    }
  }
  ProofCache& cache = ProofCache::instance();
  const uint64_t key = ProofCache::key_of(job.state);
  const ProofCache::Entry e = cache.get(key);
  if (e.exact != 0) {
    out.result = e.exact > 0 ? tactics::ProofResult::WIN : tactics::ProofResult::LOSS;
    out.move = e.move;
    out.from_cache = true;
    return out;
  }
  if (e.tried_vct >= job.vct_nodes) {  // already UNKNOWN at this budget (node budgets only)
    out.from_cache = true;
    return out;
  }
  tactics::TacticalLimits lim{job.vcf_nodes, job.vct_nodes, 0, 40, ms};
  lim.carry_unused_time = timed;  // VCT may use time a fast VCF leaves unused
  const tactics::TacticalDecision d = tactics::analyze_tactics(job.state, lim);
  out.result = d.result;
  out.nodes = d.nodes;
  if (d.result == tactics::ProofResult::WIN && d.move) out.move = *d.move;
  if (d.result == tactics::ProofResult::WIN) cache.put_exact(key, +1, out.move);
  else if (d.result == tactics::ProofResult::LOSS) cache.put_exact(key, -1);
  else if (!timed) cache.put_unknown(key, job.vct_nodes);  // a clock-limited UNKNOWN says nothing about the budget
  return out;
}

// ---- ProofPool ----

ProofPool& ProofPool::instance(int threads) {
  static ProofPool pool;
  pool.ensure_threads(threads);
  return pool;
}

void ProofPool::ensure_threads(int n) {
  std::lock_guard<std::mutex> lk(mu_);
  while (static_cast<int>(threads_.size()) < n) threads_.emplace_back([this] { worker(); });
}

void ProofPool::submit(ProofJob job) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push_back(std::move(job));
  }
  cv_.notify_one();
}

void ProofPool::worker() {
  while (true) {
    std::optional<ProofJob> job;  // GameState has no default constructor
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
      if (stop_ && queue_.empty()) return;
      job.emplace(std::move(queue_.front()));
      queue_.pop_front();
    }
    {
      std::lock_guard<std::mutex> lk(job->box->mu);
      if (job->box->closed) continue;  // the tree is gone
    }
    const ProofOutcome out = run_proof_job(*job);
    std::lock_guard<std::mutex> lk(job->box->mu);
    if (!job->box->closed) job->box->done.push_back(out);
  }
}

ProofPool::~ProofPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    queue_.clear();
  }
  cv_.notify_all();
  for (auto& t : threads_) {
    if (t.joinable()) t.join();
  }
}

}  // namespace dz::search
