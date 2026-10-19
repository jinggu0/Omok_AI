#include "rules/rule_profile.h"

#include <set>

#include <nlohmann/json.hpp>

namespace dz {

using nlohmann::json;

const char* player_name(PlayerId p) { return p == PlayerId::P0 ? "P0" : "P1"; }
const char* role_name(Role r) { return r == Role::BLACK_HOLDER ? "BLACK_HOLDER" : "WHITE_HOLDER"; }

namespace {

const char* opening_rule_name(OpeningRule r) {
  switch (r) {
    case OpeningRule::NONE: return "NONE";
    case OpeningRule::FIXED_POSITION: return "FIXED_POSITION";
    case OpeningRule::TARAGUCHI_10: return "TARAGUCHI_10";
  }
  return "?";
}

bool fail(std::string* error, const std::string& msg) {
  if (error) *error = msg;
  return false;
}

bool check_keys(const json& j, const std::set<std::string>& allowed, const std::string& where,
                std::string* error) {
  if (!j.is_object()) return fail(error, where + ": expected object");
  for (const auto& [k, v] : j.items()) {
    if (!allowed.count(k)) return fail(error, where + ": unknown key '" + k + "'");
  }
  return true;
}

bool parse_role(const json& v, Role& out) {
  if (v == "BLACK_HOLDER") { out = Role::BLACK_HOLDER; return true; }
  if (v == "WHITE_HOLDER") { out = Role::WHITE_HOLDER; return true; }
  return false;
}

}  // namespace

RuleProfile RuleProfile::renju_rif() {
  RuleProfile p;
  p.id = "RENJU_RIF";
  p.opening_rule = OpeningRule::NONE;
  p.pass_allowed = true;  // RIF Art. 5, 10
  return p;
}

RuleProfile RuleProfile::renju_gomocup() {
  RuleProfile p = renju_rif();
  p.id = "RENJU_GOMOCUP";
  p.pass_allowed = false;
  return p;
}

RuleProfile RuleProfile::taraguchi10() {
  RuleProfile p = renju_rif();
  p.id = "TARAGUCHI10";
  p.opening_rule = OpeningRule::TARAGUCHI_10;
  return p;
}

RuleProfile RuleProfile::omokdojang_dan() {
  RuleProfile p = taraguchi10();
  p.id = "OMOKDOJANG_DAN";
  p.pass_allowed = false;  // UNVERIFIED_APP_BEHAVIOR (C-06)
  return p;
}

json RuleProfile::to_json() const {
  json j;
  j["id"] = id;
  j["board_size"] = kBoardSize;
  j["black_win_condition"] = kBlackWin;
  j["white_win_condition"] = kWhiteWin;
  j["black_forbidden_rules"] = json::array({"OVERLINE", "DOUBLE_FOUR", "DOUBLE_THREE"});
  j["white_forbidden_rules"] = json::array();
  j["opening_rule"] = opening_rule_name(opening_rule);
  j["pass_allowed"] = pass_allowed;
  j["illegal_move_behavior"] = illegal_move_behavior == IllegalMoveBehavior::REJECT ? "REJECT" : "LOSS";
  j["exact_five_precedence"] = black.exact_five_precedence;
  j["five_beats_overline_other_dir"] = black.five_beats_overline_other_dir;
  json t;
  t["region"] = taraguchi.region;
  t["n_proposals"] = taraguchi.n_proposals;
  t["choice_actor"] = role_name(taraguchi.choice_actor);
  t["swap_before_direct5_actor"] = role_name(taraguchi.swap_before_direct5_actor);
  t["proposal_symmetry"] = taraguchi.proposal_symmetry == SymmetryMode::POSITION ? "POSITION" : "POINTWISE_D4";
  t["proposal_region"] = taraguchi.proposal_region;
  j["taraguchi"] = t;
  return j;
}

std::optional<RuleProfile> RuleProfile::from_json(const json& j, std::string* error) {
  static const std::set<std::string> kTop = {
      "id", "board_size", "black_win_condition", "white_win_condition", "black_forbidden_rules",
      "white_forbidden_rules", "opening_rule", "pass_allowed", "illegal_move_behavior",
      "exact_five_precedence", "five_beats_overline_other_dir", "taraguchi", "description"};
  static const std::set<std::string> kTara = {"region", "n_proposals", "choice_actor",
                                              "swap_before_direct5_actor", "proposal_symmetry",
                                              "proposal_region"};
  try {
    if (!check_keys(j, kTop, "profile", error)) return std::nullopt;
    RuleProfile p;
    p.id = j.at("id").get<std::string>();
    if (j.value("board_size", kBoardSize) != kBoardSize) {
      fail(error, "only board_size 15 is supported");
      return std::nullopt;
    }
    if (j.value("black_win_condition", std::string(kBlackWin)) != kBlackWin ||
        j.value("white_win_condition", std::string(kWhiteWin)) != kWhiteWin) {
      fail(error, "only Renju win conditions are supported");
      return std::nullopt;
    }
    if (j.contains("black_forbidden_rules")) {
      auto rules = j["black_forbidden_rules"].get<std::set<std::string>>();
      if (rules != std::set<std::string>{"OVERLINE", "DOUBLE_FOUR", "DOUBLE_THREE"}) {
        fail(error, "black_forbidden_rules must be exactly OVERLINE, DOUBLE_FOUR, DOUBLE_THREE");
        return std::nullopt;
      }
    }
    if (j.contains("white_forbidden_rules") && !j["white_forbidden_rules"].empty()) {
      fail(error, "white_forbidden_rules must be empty");
      return std::nullopt;
    }
    const std::string opening = j.value("opening_rule", std::string("NONE"));
    if (opening == "NONE") p.opening_rule = OpeningRule::NONE;
    else if (opening == "FIXED_POSITION") p.opening_rule = OpeningRule::FIXED_POSITION;
    else if (opening == "TARAGUCHI_10") p.opening_rule = OpeningRule::TARAGUCHI_10;
    else { fail(error, "unknown opening_rule " + opening); return std::nullopt; }
    p.pass_allowed = j.value("pass_allowed", p.pass_allowed);
    const std::string imb = j.value("illegal_move_behavior", std::string("REJECT"));
    if (imb == "REJECT") p.illegal_move_behavior = IllegalMoveBehavior::REJECT;
    else if (imb == "LOSS") p.illegal_move_behavior = IllegalMoveBehavior::LOSS;
    else { fail(error, "unknown illegal_move_behavior " + imb); return std::nullopt; }
    p.black.exact_five_precedence = j.value("exact_five_precedence", true);
    p.black.five_beats_overline_other_dir = j.value("five_beats_overline_other_dir", true);
    if (j.contains("taraguchi")) {
      const json& t = j["taraguchi"];
      if (!check_keys(t, kTara, "taraguchi", error)) return std::nullopt;
      if (t.contains("region")) p.taraguchi.region = t["region"].get<std::array<int, 5>>();
      for (int n : p.taraguchi.region) {
        if (n < 1 || n > 15 || n % 2 == 0) { fail(error, "region sizes must be odd 1..15"); return std::nullopt; }
      }
      p.taraguchi.n_proposals = t.value("n_proposals", 10);
      if (p.taraguchi.n_proposals < 1 || p.taraguchi.n_proposals > kMaxProposals) {
        fail(error, "n_proposals out of range");
        return std::nullopt;
      }
      if (t.contains("choice_actor") && !parse_role(t["choice_actor"], p.taraguchi.choice_actor)) {
        fail(error, "bad choice_actor");
        return std::nullopt;
      }
      if (t.contains("swap_before_direct5_actor") &&
          !parse_role(t["swap_before_direct5_actor"], p.taraguchi.swap_before_direct5_actor)) {
        fail(error, "bad swap_before_direct5_actor");
        return std::nullopt;
      }
      const std::string sym = t.value("proposal_symmetry", std::string("POSITION"));
      if (sym == "POSITION") p.taraguchi.proposal_symmetry = SymmetryMode::POSITION;
      else if (sym == "POINTWISE_D4") p.taraguchi.proposal_symmetry = SymmetryMode::POINTWISE_D4;
      else { fail(error, "bad proposal_symmetry"); return std::nullopt; }
      p.taraguchi.proposal_region = t.value("proposal_region", 0);
      if (p.taraguchi.proposal_region != 0 &&
          (p.taraguchi.proposal_region % 2 == 0 || p.taraguchi.proposal_region > 15)) {
        fail(error, "proposal_region must be 0 or odd <= 15");
        return std::nullopt;
      }
    }
    return p;
  } catch (const json::exception& e) {
    fail(error, std::string("profile json error: ") + e.what());
    return std::nullopt;
  }
}

uint64_t RuleProfile::hash() const {
  const std::string s = to_json().dump();  // nlohmann sorts object keys
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char ch : s) {
    h ^= ch;
    h *= 0x100000001B3ULL;
  }
  return h;
}

}  // namespace dz
