#include "tactics/benchmark.h"

#include <algorithm>
#include <chrono>
#include <fstream>

#include <nlohmann/json.hpp>

#include "game/position_spec.h"

namespace dz::tactics {

using nlohmann::json;

namespace {

std::optional<RuleProfile> profile_named(const std::string& name) {
  if (name == "RENJU_RIF") return RuleProfile::renju_rif();
  if (name == "TARAGUCHI10") return RuleProfile::taraguchi10();
  if (name == "OMOKDOJANG_DAN") return RuleProfile::omokdojang_dan();
  if (name == "RENJU_GOMOCUP") return RuleProfile::renju_gomocup();
  return std::nullopt;
}

ProofMoveFn oracle(const std::unordered_map<uint64_t, Point>& proof) {
  return [&proof](uint64_t key) -> std::optional<Point> {
    auto it = proof.find(key);
    if (it == proof.end()) return std::nullopt;
    return it->second;
  };
}

bool in_list(const json& list, Point p) {
  for (const auto& s : list) {
    if (parse_point(s.get<std::string>()) == p) return true;
  }
  return false;
}

}  // namespace

ProblemOutcome run_problem(const json& p, const BenchmarkSettings& settings) {
  ProblemOutcome out;
  out.id = p.value("id", std::string("?"));
  out.category = p.value("category", std::string("?"));
  out.verification = "-";
  auto fail = [&](const std::string& msg) {
    out.pass = false;
    out.message = msg;
    return out;
  };

  const auto profile = profile_named(p.value("profile", std::string("RENJU_RIF")));
  if (!profile) return fail("unknown profile");
  std::string error;
  auto spec = parse_position_spec(p.at("position"), &error);
  if (!spec) return fail("position: " + error);
  auto g = load_position(*spec, make_profile(*profile), &error);
  if (!g) return fail("position rejected: " + error);
  auto pos = TPos::from_state(*g);
  if (!pos) return fail("not an ongoing NORMAL_RENJU position");

  const uint64_t budget = p.value("budget_nodes", settings.budget_nodes);
  const std::string query = p.at("query").get<std::string>();
  const json& exp = p.at("expected");
  const std::string want = exp.at("result").get<std::string>();

  std::optional<Point> first;
  std::string reason;
  std::unordered_map<uint64_t, Point> proof;
  bool claims_win = false, claims_loss = false;
  bool exhaustive_none = false;

  const auto t0 = std::chrono::steady_clock::now();
  if (query == "vcf") {
    const VcfResult r = solve_vcf(*pos, {budget, 40});
    out.nodes = r.nodes;
    out.claimed = vcf_status_name(r.status);
    if (r.status == VcfResult::Status::FOUND) {
      claims_win = true;
      first = r.pv.front();
      proof = proof_from_pv(*g, r.pv);
    }
    exhaustive_none = r.status == VcfResult::Status::NONE && !r.depth_limited;
  } else if (query == "vct" || query == "loss") {
    const DfpnResult r = solve_dfpn(*pos, query == "vct" ? DfpnMode::PROVE_WIN : DfpnMode::PROVE_LOSS, {budget, 40, false});
    out.nodes = r.nodes;
    out.claimed = proof_name(r.result);
    claims_win = r.result == ProofResult::WIN;
    claims_loss = r.result == ProofResult::LOSS;
    if (claims_win) first = r.best;
    proof = r.proof_moves;
  } else if (query == "decide") {
    const TacticalDecision d = analyze_tactics(*g, {budget, budget, budget / 2, 40});
    out.nodes = d.nodes;
    out.claimed = std::string(proof_name(d.result)) + "/" + tactic_reason_name(d.reason);
    claims_win = d.result == ProofResult::WIN;
    claims_loss = d.result == ProofResult::LOSS;
    first = d.move;
    reason = tactic_reason_name(d.reason);
    proof = d.proof_moves;
    out.claimed = std::string(proof_name(d.result)) + "/" + reason;
  } else {
    return fail("unknown query " + query);
  }
  out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  // Independent verification of every proof claim.
  if (settings.verify_claims && (claims_win || claims_loss)) {
    const VerifyStatus v = claims_win ? verify_forced_win(*g, oracle(proof), settings.verify)
                                      : verify_forced_loss(*g, oracle(proof), settings.verify);
    out.verification = verify_status_name(v);
    if (v == VerifyStatus::REFUTED) {
      out.false_claim = true;
      return fail("PROOF REFUTED by the independent verifier");
    }
    if (v == VerifyStatus::TOO_LARGE) return fail("claim could not be verified within limits");
  }

  // Expectation.
  bool match = false;
  if (want == "WIN" || want == "FOUND") match = claims_win;
  else if (want == "LOSS") match = claims_loss;
  else if (want == "NOT_WIN") match = !claims_win;
  else if (want == "NONE") match = exhaustive_none;
  else if (want == "UNKNOWN") match = !claims_win && !claims_loss;
  else return fail("unknown expected result " + want);
  out.solved = (want == "WIN" || want == "FOUND" || want == "LOSS") && match;
  if (!match) return fail("expected " + want + ", got " + out.claimed);
  if (exp.contains("reason") && exp["reason"].get<std::string>() != reason) {
    return fail("expected reason " + exp["reason"].get<std::string>() + ", got " + reason);
  }
  if (exp.contains("move") && (!first || parse_point(exp["move"].get<std::string>()) != first)) {
    return fail("expected move " + exp["move"].get<std::string>());
  }
  if (exp.contains("first_moves") && (!first || !in_list(exp["first_moves"], *first))) {
    return fail("first move " + (first ? to_notation(*first) : std::string("none")) + " not in expected list");
  }
  out.pass = true;
  return out;
}

std::vector<ProblemOutcome> run_suite_file(const std::string& path, const BenchmarkSettings& settings) {
  std::ifstream in(path);
  const json doc = json::parse(in);
  std::vector<ProblemOutcome> out;
  for (const auto& p : doc.at("problems")) out.push_back(run_problem(p, settings));
  return out;
}

}  // namespace dz::tactics
