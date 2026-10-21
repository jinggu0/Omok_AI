// Tactical-solver profiling and benchmarking on fixed corpora (PROGRESS Phase 8 E3b).
//
//   dojozero_tprof  --corpus benchmarks/perf/after_blunder.json --mode vct --nodes 50000 --out p.json
//   dojozero_tbench --corpus benchmarks/perf/filter_workload.json --mode analyze --ms 125 --threads 8
//
// Same source, two builds: dojozero_tprof links the DZ_TPROF library (per-section inclusive time and
// call counts, allocation counts); dojozero_tbench links the production library (timings without
// instrumentation). --impl ref runs the frozen ee1d011 solver (dz::tactics::ref) in the same binary.
//
// Modes (side to move = attacker):
//   vcf      solve_vcf(nodes, depth 40)
//   vct      solve_dfpn(PROVE_WIN, nodes, depth 40)            (the filter's VCT stage)
//   analyze  analyze_tactics(vcf_nodes, vct_nodes, loss 0, ms) (the root defense filter call)
// Output: JSON with per-position results and aggregates; with DZ_TPROF also the section table.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "game/position_spec.h"
#include "tactics/engine.h"
#include "tactics/reference/ref.h"
#include "tactics/tprof.h"

using namespace dz;
using namespace dz::tactics;
using nlohmann::json;
using Clock = std::chrono::steady_clock;

// ---- allocation counting (this executable only) ----
static std::atomic<uint64_t> g_allocs{0};
static std::atomic<uint64_t> g_alloc_bytes{0};
void* operator new(std::size_t n) {
  g_allocs.fetch_add(1, std::memory_order_relaxed);
  g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

std::map<std::string, std::string> parse_flags(int argc, char** argv) {
  std::map<std::string, std::string> f;
  for (int i = 1; i + 1 < argc; i += 2) f[std::string(argv[i]).substr(2)] = argv[i + 1];
  return f;
}

ProfilePtr profile_named(const std::string& n) {
  if (n == "RENJU_RIF") return make_profile(RuleProfile::renju_rif());
  if (n == "TARAGUCHI10") return make_profile(RuleProfile::taraguchi10());
  if (n == "OMOKDOJANG_DAN") return make_profile(RuleProfile::omokdojang_dan());
  if (n == "RENJU_GOMOCUP") return make_profile(RuleProfile::renju_gomocup());
  throw std::runtime_error("unknown profile " + n);
}

struct Item {
  std::string id;
  GameState state;
};

std::vector<Item> load_corpus(const std::string& path, std::string* sha) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  const json doc = json::parse(in);
  *sha = doc.value("sha256", std::string("UNKNOWN"));
  std::vector<Item> out;
  for (const auto& p : doc.at("positions")) {
    json spec = {{"black", p.at("black")}, {"white", p.at("white")}};
    if (p.contains("side") && !p["side"].is_null()) spec["side"] = p["side"];
    std::string err;
    auto ps = parse_position_spec(spec, &err);
    if (!ps) throw std::runtime_error(p.at("id").get<std::string>() + ": " + err);
    auto g = load_position(*ps, profile_named(p.value("profile", std::string("RENJU_RIF"))), &err);
    if (!g) throw std::runtime_error(p.at("id").get<std::string>() + ": " + err);
    out.push_back({p.at("id").get<std::string>(), *g});
  }
  return out;
}

struct Run {
  std::string result;  // WIN / NONE / UNKNOWN / BUDGET ...
  uint64_t nodes = 0;
  double ms = 0;
};


bool g_gen_cache = true, g_lazy_legality = true, g_fast_refute = true, g_carry = true;  // --gen-cache 0 / --lazy 0 (live df-pn only)

Run run_one(const GameState& g, const std::string& mode, bool ref, uint64_t nodes, uint64_t nodes2, double ms) {
  Run r;
  const auto t0 = Clock::now();
  if (mode == "analyze") {
    TacticalLimits lim{nodes, nodes2, 0, 40, ms};
    lim.carry_unused_time = g_carry;
    const TacticalDecision d = ref ? ref::analyze_tactics(g, lim) : analyze_tactics(g, lim);
    r.result = proof_name(d.result);
    r.nodes = d.nodes;
  } else {
    auto pos = TPos::from_state(g);
    if (!pos) {
      r.result = "TERMINAL";
    } else if (mode == "vcf") {
      const VcfLimits lim{nodes, 40, ms};
      const VcfResult v = ref ? ref::solve_vcf(*pos, lim) : solve_vcf(*pos, lim);
      r.result = vcf_status_name(v.status);
      r.nodes = v.nodes;
    } else {
      DfpnLimits lim{nodes, 40, false, ms};
      lim.gen_cache = g_gen_cache;
      lim.lazy_legality = g_lazy_legality;
      lim.fast_refute = g_fast_refute;
      const DfpnResult d = ref ? ref::solve_dfpn(*pos, DfpnMode::PROVE_WIN, lim)
                               : solve_dfpn(*pos, DfpnMode::PROVE_WIN, lim);
      r.result = proof_name(d.result);
      r.nodes = d.nodes;
    }
  }
  r.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  return r;
}

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
}

}  // namespace

int main(int argc, char** argv) {
  auto f = parse_flags(argc, argv);
  const std::string corpus = f.at("corpus");
  const std::string mode = f.count("mode") ? f["mode"] : "vct";
  const bool ref = f.count("impl") && f["impl"] == "ref";
  // --impl pair: each position is solved by the reference and then by the live solver back to back
  // (same thread, same moment), so machine-state drift (battery, thermal limits) hits both alike.
  const bool pair = f.count("impl") && f["impl"] == "pair";
#ifdef _WIN32
  // Opt this process out of Windows power throttling (EcoQoS), which otherwise may park a background
  // benchmark on slow cores. Process-local; no system setting is changed.
  PROCESS_POWER_THROTTLING_STATE pt{};
  pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  pt.StateMask = 0;
  SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof(pt));
#endif
  const uint64_t nodes = f.count("nodes") ? std::stoull(f["nodes"]) : (mode == "vcf" ? 100000 : 50000);
  const uint64_t nodes2 = f.count("nodes2") ? std::stoull(f["nodes2"]) : 50000;  // analyze: VCT nodes
  const double ms = f.count("ms") ? std::stod(f["ms"]) : 0.0;
  const int threads = f.count("threads") ? std::stoi(f["threads"]) : 1;
  const int limit = f.count("limit") ? std::stoi(f["limit"]) : 1 << 30;
  if (f.count("gen-cache")) g_gen_cache = f["gen-cache"] != "0";
  if (f.count("lazy")) g_lazy_legality = f["lazy"] != "0";
  if (f.count("fast-refute")) g_fast_refute = f["fast-refute"] != "0";
  if (f.count("carry")) g_carry = f["carry"] != "0";
  std::string sha;
  std::vector<Item> items = load_corpus(corpus, &sha);
  if (static_cast<int>(items.size()) > limit) items.erase(items.begin() + limit, items.end());

#ifdef DZ_TPROF
  // Timer overhead per scope (two clock reads + accumulate), to discount fine-grained sections.
  tprof::reset();
  const auto c0 = Clock::now();
  for (int i = 0; i < 200000; ++i) { DZ_TPROF_SCOPE(TT_HIT); }
  const double scope_ns = std::chrono::duration<double, std::nano>(Clock::now() - c0).count() / 200000;
  tprof::reset();
  tprof::Counters total;
#endif

  std::vector<Run> runs(items.size()), ref_runs(pair ? items.size() : 0);
  const uint64_t alloc0 = g_allocs.load();
  const auto t0 = Clock::now();
  std::atomic<size_t> next{0};
  std::mutex mu;
  auto worker = [&]() {
    for (size_t i; (i = next.fetch_add(1)) < items.size();) {
#ifdef DZ_TPROF
      tprof::reset();
#endif
      if (pair) ref_runs[i] = run_one(items[i].state, mode, true, nodes, nodes2, ms);
      runs[i] = run_one(items[i].state, mode, ref, nodes, nodes2, ms);
#ifdef DZ_TPROF
      std::lock_guard<std::mutex> lk(mu);
      for (int s = 0; s < tprof::kSections; ++s) {
        total.ns[s] += tprof::counters().ns[s];
        total.calls[s] += tprof::counters().calls[s];
      }
#endif
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  const uint64_t allocs = g_allocs.load() - alloc0;

  uint64_t total_nodes = 0;
  double sum_ms = 0;
  std::map<std::string, int> results;
  std::vector<double> ms_all, ms_proof, nodes_proof;
  json per = json::array();
  for (size_t i = 0; i < items.size(); ++i) {
    const Run& r = runs[i];
    total_nodes += r.nodes;
    sum_ms += r.ms;
    ++results[r.result];
    ms_all.push_back(r.ms);
    if (r.result == "WIN" || r.result == "FOUND") {
      ms_proof.push_back(r.ms);
      nodes_proof.push_back(static_cast<double>(r.nodes));
    }
    json row = {{"id", items[i].id}, {"result", r.result}, {"nodes", r.nodes}, {"ms", r.ms}};
    if (pair) row["ref"] = {{"result", ref_runs[i].result}, {"nodes", ref_runs[i].nodes}, {"ms", ref_runs[i].ms}};
    per.push_back(row);
  }
  json out = {
      {"corpus", corpus}, {"corpus_sha256", sha}, {"mode", mode}, {"impl", pair ? "pair(ref then live)" : ref ? "ref(ee1d011)" : "live"},
      {"nodes_budget", nodes}, {"vct_nodes_budget", nodes2}, {"ms_budget", ms}, {"threads", threads}, {"gen_cache", g_gen_cache}, {"lazy_legality", g_lazy_legality}, {"fast_refute", g_fast_refute}, {"carry_unused_time", g_carry},
      {"positions", items.size()}, {"wall_ms", wall_ms}, {"solver_ms_sum", sum_ms}, {"total_nodes", total_nodes},
      {"nodes_per_s_per_thread", sum_ms > 0 ? total_nodes / (sum_ms / 1000.0) : 0.0},
      {"nodes_per_s_aggregate", wall_ms > 0 ? total_nodes / (wall_ms / 1000.0) : 0.0},
      {"results", results}, {"median_ms", pct(ms_all, 0.5)}, {"p95_ms", pct(ms_all, 0.95)},
      {"proved", ms_proof.size()}, {"median_ms_to_proof", pct(ms_proof, 0.5)}, {"p95_ms_to_proof", pct(ms_proof, 0.95)},
      {"median_nodes_to_proof", pct(nodes_proof, 0.5)}, {"p95_nodes_to_proof", pct(nodes_proof, 0.95)},
      {"allocations", allocs}, {"allocations_per_node", total_nodes ? double(allocs) / total_nodes : 0.0},
#ifdef DZ_TPROF
      {"build", "DZ_TPROF (instrumented)"},
#else
      {"build", "production"},
#endif
  };
#ifdef DZ_TPROF
  json sec = json::object();
  for (int s = 0; s < tprof::kSections; ++s) {
    sec[tprof::kNames[s]] = {{"calls", total.calls[s]}, {"total_ms", total.ns[s] / 1e6},
                             {"avg_ns", total.calls[s] ? total.ns[s] / total.calls[s] : 0.0},
                             {"share_of_solver_time", sum_ms > 0 ? total.ns[s] / 1e6 / sum_ms : 0.0}};
  }
  out["sections_inclusive"] = sec;
  out["timer_overhead_ns_per_scope"] = scope_ns;
#endif
  if (pair) {
    double ref_ms = 0;
    uint64_t ref_nodes = 0;
    int mismatch = 0;
    std::vector<double> ratio;
    for (size_t i = 0; i < items.size(); ++i) {
      ref_ms += ref_runs[i].ms;
      ref_nodes += ref_runs[i].nodes;
      mismatch += ref_runs[i].result != runs[i].result || (ms == 0 && ref_runs[i].nodes != runs[i].nodes);
      if (ref_runs[i].ms > 5) ratio.push_back(ref_runs[i].ms / std::max(runs[i].ms, 1e-3));
    }
    out["pair"] = {{"ref_solver_ms_sum", ref_ms}, {"ref_total_nodes", ref_nodes},
                   {"ref_nodes_per_s_per_thread", ref_ms > 0 ? ref_nodes / (ref_ms / 1000.0) : 0.0},
                   {"speedup_total_time", sum_ms > 0 ? ref_ms / sum_ms : 0.0},
                   {"speedup_nps", (sum_ms > 0 && ref_ms > 0 && ref_nodes > 0)
                                       ? (total_nodes / (sum_ms / 1000.0)) / (ref_nodes / (ref_ms / 1000.0)) : 0.0},
                   {"per_position_speedup_median", pct(ratio, 0.5)}, {"per_position_speedup_p10", pct(ratio, 0.1)},
                   {"per_position_speedup_p90", pct(ratio, 0.9)},
                   {"mismatch_result_or_nodes", mismatch}};
  }
  out["per_position"] = per;
  if (f.count("out")) {
    std::ofstream(f["out"]) << out.dump(1);
  }
  out.erase("per_position");
  std::printf("%s\n", out.dump(1).c_str());
  return 0;
}
