// Data-driven forbidden-move cases: tests/forbidden/cases/*.json.
// Each case is checked with the fast checker, the explain path, the reference oracle, and under
// all 8 D4 transforms.
#include <gtest/gtest.h>

#include "common/test_util.h"
#include "rules/forbidden.h"
#include "rules/reference_rules.h"

namespace dz {
namespace {

using nlohmann::json;

struct ForbiddenCase {
  std::string file;
  std::string id;
  json data;
};

std::vector<ForbiddenCase> load_cases() {
  std::vector<ForbiddenCase> out;
  for (const auto& path : test::json_files_in(test::source_dir() / "tests" / "forbidden" / "cases")) {
    const json doc = test::read_json(path);
    for (const auto& c : doc.at("cases")) {
      out.push_back({path.filename().string(), c.at("id").get<std::string>(), c});
    }
  }
  return out;
}

std::string sanitize(const std::string& id) {
  std::string s;
  for (char ch : id) s += std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_';
  return s;
}

class ForbiddenCaseTest : public ::testing::TestWithParam<ForbiddenCase> {};

TEST_P(ForbiddenCaseTest, MatchesExpectation) {
  const ForbiddenCase& tc = GetParam();
  std::string error;
  auto spec = parse_position_spec(tc.data.at("position"), &error);
  ASSERT_TRUE(spec) << error;
  const Board board = test::board_from_spec(*spec);
  const Point q = test::P(tc.data.at("query").get<std::string>().c_str());
  ASSERT_TRUE(board.empty_at(q)) << "query point must be empty";

  BlackRuleOptions opt;
  if (tc.data.contains("options")) {
    const json& o = tc.data["options"];
    opt.exact_five_precedence = o.value("exact_five_precedence", true);
    opt.five_beats_overline_other_dir = o.value("five_beats_overline_other_dir", true);
  }
  const json& exp = tc.data.at("expected");
  const auto expected = parse_verdict(exp.at("verdict").get<std::string>());
  ASSERT_TRUE(expected);

  const ForbiddenInfo info = explain_black_move(board, q, opt);
  const std::string context = "\n" + board.to_string() + info.to_string();

  EXPECT_EQ(black_verdict(board, q, opt), *expected) << context;
  EXPECT_EQ(info.verdict, *expected) << context;
  EXPECT_EQ(ref::black_verdict(board, q, opt), *expected) << "reference oracle disagrees" << context;

  if (exp.contains("fours")) {
    EXPECT_EQ(info.four_count(), exp["fours"].get<int>()) << context;
    EXPECT_EQ(ref::count_black_fours(board, q), exp["fours"].get<int>()) << context;
  }
  if (exp.contains("apparent_threes")) {
    EXPECT_EQ(info.apparent_three_count(), exp["apparent_threes"].get<int>()) << context;
    EXPECT_EQ(ref::count_black_apparent_threes(board, q), exp["apparent_threes"].get<int>()) << context;
  }
  if (exp.contains("real_threes")) {
    EXPECT_EQ(info.real_three_count(), exp["real_threes"].get<int>()) << context;
    EXPECT_EQ(ref::count_black_real_threes(board, q, opt), exp["real_threes"].get<int>()) << context;
  }
  for (int t = 1; t < kNumTransforms; ++t) {
    EXPECT_EQ(black_verdict(board.transformed(t), transform_point(t, q), opt), *expected)
        << "transform " << transform_name(t) << context;
  }
}

INSTANTIATE_TEST_SUITE_P(Json, ForbiddenCaseTest, ::testing::ValuesIn(load_cases()),
                         [](const ::testing::TestParamInfo<ForbiddenCase>& info) {
                           return sanitize(info.param.id);
                         });

TEST(ForbiddenCases, CaseFilesPresent) { EXPECT_GE(load_cases().size(), 40u); }

}  // namespace
}  // namespace dz
