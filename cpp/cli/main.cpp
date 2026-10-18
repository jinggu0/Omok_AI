// dojozero_engine: rule-engine commands available in Phase 2 (ARCHITECTURE.md §6).
//
//   dojozero_engine rules-test [--root <repo>] [--fuzz <comparisons>]
//   dojozero_engine forbidden  [--profile <name|file.json>] --position '<json>' [--point H8]
//   dojozero_engine legal      [--profile <name|file.json>] --position '<json>'
//   dojozero_engine tactics    [--profile <name|file.json>] --position '<json>' [--budget N] [--verify]
//   dojozero_engine benchmark  --suite <file.json|dir> [--budget N]
//
// <json> is a position spec (game/position_spec.h), e.g. '{"moves":["H8","keep","I9"]}' or
// '{"black":["F8","G8","H6","H7"],"white":["A15","C15","E15","G15"]}'.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>

#include <nlohmann/json.hpp>

#include "game/position_spec.h"
#include "rules/reference_rules.h"
#include "tactics/benchmark.h"
#include "tactics/engine.h"
#include "tactics/verify.h"
#include "tools/random_positions.h"

using namespace dz;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

std::map<std::string, std::string> parse_flags(int argc, char** argv, int start) {
  std::map<std::string, std::string> f;
  for (int i = start; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) continue;
    f[k.substr(2)] = (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ? argv[++i] : "true";
  }
  return f;
}

ProfilePtr load_profile(const std::string& spec) {
  if (spec.empty() || spec == "RENJU_RIF") return make_profile(RuleProfile::renju_rif());
  if (spec == "TARAGUCHI10") return make_profile(RuleProfile::taraguchi10());
  if (spec == "OMOKDOJANG_DAN") return make_profile(RuleProfile::omokdojang_dan());
  std::ifstream in(spec);
  if (!in) throw std::runtime_error("cannot open profile " + spec);
  std::string error;
  auto p = RuleProfile::from_json(json::parse(in), &error);
  if (!p) throw std::runtime_error(error);
  return make_profile(*p);
}

GameState load_state(const std::map<std::string, std::string>& f) {
  ProfilePtr profile = load_profile(f.count("profile") ? f.at("profile") : "");
  if (!f.count("position")) throw std::runtime_error("--position is required");
  std::string error;
  auto spec = parse_position_spec(json::parse(f.at("position")), &error);
  if (!spec) throw std::runtime_error(error);
  auto g = load_position(*spec, profile, &error);
  if (!g) throw std::runtime_error("rejected position: " + error);
  return *g;
}

int cmd_forbidden(const std::map<std::string, std::string>& f) {
  const GameState g = load_state(f);
  std::cout << g.to_string();
  if (f.count("point")) {
    auto p = parse_point(f.at("point"));
    if (!p || !g.board().empty_at(*p)) throw std::runtime_error("--point must be an empty point");
    std::cout << g.forbidden(*p).to_string();
    return 0;
  }
  int n = 0;
  for (int i = 0; i < kNumPoints; ++i) {
    const Point p = static_cast<Point>(i);
    if (!g.board().empty_at(p)) continue;
    const MoveVerdict v = g.black_verdict(p);
    if (is_forbidden(v)) {
      std::cout << to_notation(p) << " " << verdict_name(v) << "\n";
      ++n;
    }
  }
  std::cout << n << " forbidden point(s) for black\n";
  return 0;
}

int cmd_legal(const std::map<std::string, std::string>& f) {
  const GameState g = load_state(f);
  std::cout << g.to_string();
  const auto actions = g.legal_actions();
  std::cout << actions.size() << " legal action(s):";
  for (const Action& a : actions) std::cout << ' ' << to_string(a);
  std::cout << "\n";
  return 0;
}

// Replays the JSON forbidden cases (fast + oracle + D4) and a short differential fuzz.
int cmd_rules_test(const std::map<std::string, std::string>& f) {
  const fs::path root = f.count("root") ? fs::path(f.at("root")) : fs::current_path();
  int cases = 0, failures = 0;
  for (const auto& entry : fs::directory_iterator(root / "tests" / "forbidden" / "cases")) {
    if (entry.path().extension() != ".json") continue;
    std::ifstream in(entry.path());
    const json doc = json::parse(in);
    for (const auto& c : doc.at("cases")) {
      ++cases;
      std::string error;
      auto spec = parse_position_spec(c.at("position"), &error);
      Board b;
      for (Point p : spec->black) b.place(p, Color::BLACK);
      for (Point p : spec->white) b.place(p, Color::WHITE);
      const Point q = *parse_point(c.at("query").get<std::string>());
      BlackRuleOptions opt;
      if (c.contains("options")) {
        opt.exact_five_precedence = c["options"].value("exact_five_precedence", true);
        opt.five_beats_overline_other_dir = c["options"].value("five_beats_overline_other_dir", true);
      }
      const auto want = *parse_verdict(c.at("expected").at("verdict").get<std::string>());
      bool ok = black_verdict(b, q, opt) == want && ref::black_verdict(b, q, opt) == want;
      for (int t = 1; t < kNumTransforms && ok; ++t) {
        ok = black_verdict(b.transformed(t), transform_point(t, q), opt) == want;
      }
      if (!ok) {
        ++failures;
        std::cout << "FAIL " << c.at("id").get<std::string>() << "\n";
      }
    }
  }
  const long long fuzz = f.count("fuzz") ? std::atoll(f.at("fuzz").c_str()) : 20000;
  std::mt19937_64 rng(1);
  long long compared = 0, mismatches = 0;
  while (compared < fuzz) {
    const Board b = tools::random_dense_board(rng);
    for (int i = 0; i < kNumPoints && compared < fuzz; ++i) {
      const Point p = static_cast<Point>(i);
      if (!b.empty_at(p) || !tools::near_black(b, p)) continue;
      ++compared;
      if (black_verdict(b, p) != ref::black_verdict(b, p)) ++mismatches;
    }
  }
  std::cout << "json cases: " << cases << " run, " << failures << " failed\n"
            << "differential fuzz: " << compared << " compared, " << mismatches << " mismatches\n";
  return failures == 0 && mismatches == 0 ? 0 : 1;
}

int cmd_tactics(const std::map<std::string, std::string>& f) {
  const GameState g = load_state(f);
  std::cout << g.to_string();
  const uint64_t budget = f.count("budget") ? std::strtoull(f.at("budget").c_str(), nullptr, 10) : 100000;
  const tactics::TacticalDecision d = tactics::analyze_tactics(g, {budget, budget, budget / 2, 40});
  std::cout << d.to_string() << "\n";
  if (f.count("verify") && d.result != tactics::ProofResult::UNKNOWN) {
    const tactics::ProofMoveFn oracle = [&](uint64_t key) -> std::optional<Point> {
      auto it = d.proof_moves.find(key);
      if (it == d.proof_moves.end()) return std::nullopt;
      return it->second;
    };
    const auto v = d.result == tactics::ProofResult::WIN ? tactics::verify_forced_win(g, oracle)
                                                         : tactics::verify_forced_loss(g, oracle);
    std::cout << "independent verification: " << tactics::verify_status_name(v) << "\n";
    return v == tactics::VerifyStatus::REFUTED ? 1 : 0;
  }
  return 0;
}

int cmd_benchmark(const std::map<std::string, std::string>& f) {
  if (!f.count("suite")) throw std::runtime_error("--suite is required");
  std::vector<fs::path> files;
  const fs::path suite = f.at("suite");
  if (fs::is_directory(suite)) {
    for (const auto& e : fs::recursive_directory_iterator(suite)) {
      if (e.path().extension() == ".json") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
  } else {
    files.push_back(suite);
  }
  tactics::BenchmarkSettings settings;
  if (f.count("budget")) settings.budget_nodes = std::strtoull(f.at("budget").c_str(), nullptr, 10);
  int total = 0, passed = 0, false_claims = 0, expected_proofs = 0, solved = 0;
  double ms = 0;
  for (const auto& file : files) {
    for (const auto& o : tactics::run_suite_file(file.string(), settings)) {
      ++total;
      passed += o.pass;
      false_claims += o.false_claim;
      solved += o.solved;
      ms += o.ms;
      std::cout << (o.pass ? "PASS " : "FAIL ") << file.filename().string() << " " << o.id << " [" << o.category
                << "] claimed=" << o.claimed << " verify=" << o.verification << " nodes=" << o.nodes
                << " ms=" << static_cast<int>(o.ms) << (o.message.empty() ? "" : " : " + o.message) << "\n";
    }
  }
  for (const auto& file : files) {
    std::ifstream in(file);
    const json doc = json::parse(in);  // keep alive: .at() returns a reference into it
    for (const auto& p : doc.at("problems")) {
      const std::string r = p.at("expected").at("result").get<std::string>();
      expected_proofs += (r == "WIN" || r == "FOUND" || r == "LOSS") ? 1 : 0;
    }
  }
  std::cout << "problems=" << total << " passed=" << passed << " false_claims=" << false_claims
            << " solve_rate=" << solved << "/" << expected_proofs << " solver_time_ms=" << static_cast<int>(ms)
            << "\n";
  return passed == total && false_claims == 0 ? 0 : 1;
}

void usage() {
  std::cerr << "usage: dojozero_engine <rules-test|forbidden|legal|tactics|benchmark> [options]\n"
               "  rules-test [--root <repo>] [--fuzz N]\n"
               "  forbidden  [--profile NAME|file] --position JSON [--point H8]\n"
               "  legal      [--profile NAME|file] --position JSON\n"
               "  tactics    [--profile NAME|file] --position JSON [--budget N] [--verify]\n"
               "  benchmark  --suite FILE|DIR [--budget N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string cmd = argv[1];
  const auto flags = parse_flags(argc, argv, 2);
  try {
    if (cmd == "rules-test") return cmd_rules_test(flags);
    if (cmd == "forbidden") return cmd_forbidden(flags);
    if (cmd == "legal") return cmd_legal(flags);
    if (cmd == "tactics") return cmd_tactics(flags);
    if (cmd == "benchmark") return cmd_benchmark(flags);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  usage();
  return 2;
}
