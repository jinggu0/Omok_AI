#include <gtest/gtest.h>

#include "common/test_util.h"
#include "rules/rule_profile.h"

namespace dz {
namespace {

using nlohmann::json;

TEST(RuleProfile, ConfigFilesMatchBuiltins) {
  const auto dir = test::source_dir() / "configs" / "rules";
  const std::pair<const char*, RuleProfile> expected[] = {
      {"renju_rif.json", RuleProfile::renju_rif()},
      {"taraguchi10.json", RuleProfile::taraguchi10()},
      {"omokdojang_dan.json", RuleProfile::omokdojang_dan()},
  };
  for (const auto& [file, builtin] : expected) {
    std::string error;
    auto p = RuleProfile::from_json(test::read_json(dir / file), &error);
    ASSERT_TRUE(p) << file << ": " << error;
    EXPECT_EQ(p->to_json(), builtin.to_json()) << file;
    EXPECT_EQ(p->hash(), builtin.hash()) << file;
  }
}

TEST(RuleProfile, RoundTripAndHashSensitivity) {
  RuleProfile p = RuleProfile::omokdojang_dan();
  std::string error;
  auto back = RuleProfile::from_json(p.to_json(), &error);
  ASSERT_TRUE(back) << error;
  EXPECT_EQ(back->hash(), p.hash());
  RuleProfile q = p;
  q.taraguchi.swap_before_direct5_actor = Role::WHITE_HOLDER;
  EXPECT_NE(q.hash(), p.hash());
  EXPECT_NE(RuleProfile::renju_rif().hash(), RuleProfile::taraguchi10().hash());
}

TEST(RuleProfile, RejectsUnknownOrUnsupported) {
  std::string error;
  json j = RuleProfile::renju_rif().to_json();
  j["pass_alowed"] = true;  // typo must not be silently ignored
  EXPECT_FALSE(RuleProfile::from_json(j, &error));
  EXPECT_NE(error.find("unknown key"), std::string::npos);

  j = RuleProfile::renju_rif().to_json();
  j["board_size"] = 19;
  EXPECT_FALSE(RuleProfile::from_json(j, &error));

  j = RuleProfile::renju_rif().to_json();
  j["white_forbidden_rules"] = json::array({"OVERLINE"});
  EXPECT_FALSE(RuleProfile::from_json(j, &error));

  j = RuleProfile::renju_rif().to_json();
  j["black_forbidden_rules"] = json::array({"OVERLINE"});
  EXPECT_FALSE(RuleProfile::from_json(j, &error));

  j = RuleProfile::taraguchi10().to_json();
  j["taraguchi"]["region"] = json::array({1, 3, 4, 7, 9});
  EXPECT_FALSE(RuleProfile::from_json(j, &error));

  j = RuleProfile::taraguchi10().to_json();
  j["taraguchi"]["choice_actor"] = "NOBODY";
  EXPECT_FALSE(RuleProfile::from_json(j, &error));
}

}  // namespace
}  // namespace dz
