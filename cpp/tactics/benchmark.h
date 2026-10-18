// Tactical benchmark runner (EVALUATION.md §3, §9): benchmarks/{tactical,vcf,vct}/*.json.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "tactics/engine.h"
#include "tactics/verify.h"

namespace dz::tactics {

struct ProblemOutcome {
  std::string id;
  std::string category;
  bool pass = false;           // matched expectation and every proof claim verified
  bool false_claim = false;    // claimed WIN/LOSS that the verifier REFUTED, or wrong-sign claim
  bool solved = false;         // expected WIN/LOSS/FOUND and the solver proved it
  std::string claimed;         // solver answer
  std::string verification;    // VERIFIED / REFUTED / TOO_LARGE / -
  std::string message;
  uint64_t nodes = 0;
  double ms = 0;
};

struct BenchmarkSettings {
  uint64_t budget_nodes = 200000;  // per solver call unless the problem overrides it
  VerifyLimits verify{80, 5000000};  // same limits the generator verified with
  bool verify_claims = true;
};

ProblemOutcome run_problem(const nlohmann::json& problem, const BenchmarkSettings& settings = {});
std::vector<ProblemOutcome> run_suite_file(const std::string& path, const BenchmarkSettings& settings = {});

}  // namespace dz::tactics
