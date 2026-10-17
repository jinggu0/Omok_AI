// Omokdojang compatibility regression runner (docs/ASSUMPTIONS.md §2.1).
// Collects tests/compatibility/omokdojang/*.json (evidence gathered manually from the app).
// A mismatch means the engine default differs from observed app behaviour: update the matching
// C-xx entry and the profile, do not "fix" the case.
#include <gtest/gtest.h>

#include <set>

#include "common/test_util.h"
#include "game/position_spec.h"

namespace dz {
namespace {

using nlohmann::json;

std::filesystem::path compat_dir() { return test::source_dir() / "tests" / "compatibility" / "omokdojang"; }

void run_case(const std::filesystem::path& path) {
  SCOPED_TRACE(path.filename().string());
  const json c = test::read_json(path);
  ASSERT_EQ(c.at("schema_version").get<int>(), 1);
  const std::string compat_id = c.at("compat_id").get<std::string>();
  ProfilePtr profile = test::profile_by_name(c.value("profile", std::string("OMOKDOJANG_DAN")));
  std::string error;
  auto spec = parse_position_spec(c.at("position"), &error);
  ASSERT_TRUE(spec) << error;
  auto g = load_position(*spec, profile, &error);
  if (c.contains("expected_load_error")) {
    EXPECT_FALSE(g) << compat_id << ": app rejects this position";
    return;
  }
  ASSERT_TRUE(g) << error;
  if (c.contains("side")) {
    ASSERT_EQ(color_name(g->side_to_move()), c["side"].get<std::string>()) << compat_id;
  }
  if (c.contains("query")) {
    const Point q = test::P(c["query"].at("point").get<std::string>().c_str());
    if (c.contains("expected_legal")) {
      EXPECT_EQ(g->is_legal(Action::place(q)), c["expected_legal"].get<bool>()) << compat_id;
    }
    if (c.contains("expected_forbidden")) {
      const std::string want = c["expected_forbidden"].get<std::string>();
      const MoveVerdict v = g->black_verdict(q);
      if (want == "NONE") {
        EXPECT_FALSE(is_forbidden(v)) << compat_id << " got " << verdict_name(v);
      } else {
        EXPECT_EQ(std::string(verdict_name(v)), "FORBIDDEN_" + want) << compat_id << "\n"
                                                                     << g->forbidden(q).to_string();
      }
    }
  }
  if (c.contains("expected_legal_moves")) {
    std::set<std::string> want, got;
    for (const auto& m : c["expected_legal_moves"]) want.insert(m.get<std::string>());
    for (const Action& a : g->legal_actions()) got.insert(to_string(a));
    EXPECT_EQ(got, want) << compat_id;
  }
  if (c.contains("expected_phase")) {
    EXPECT_EQ(phase_name(g->phase()), c["expected_phase"].get<std::string>()) << compat_id;
  }
  if (c.contains("apply")) {
    for (const auto& m : c["apply"]) {
      auto a = parse_action_in_phase(m.get<std::string>(), g->phase());
      ASSERT_TRUE(a);
      ASSERT_TRUE(g->apply(*a)) << m;
    }
  }
  if (c.contains("expected_result")) {
    EXPECT_EQ(result_name(g->outcome().result), c["expected_result"].get<std::string>()) << compat_id;
  }
}

TEST(OmokdojangCompatibility, AllCollectedCases) {
  const auto files = test::json_files_in(compat_dir());
  if (files.empty()) {
    GTEST_SKIP() << "no app evidence collected yet in " << compat_dir().string()
                 << " (all C-xx items are UNVERIFIED_APP_BEHAVIOR)";
  }
  for (const auto& f : files) run_case(f);
}

// The template must stay loadable so collectors can copy it.
TEST(OmokdojangCompatibility, TemplateIsValid) {
  const auto path = compat_dir() / "TEMPLATE.json.example";
  ASSERT_TRUE(std::filesystem::exists(path));
  run_case(path);
}

}  // namespace
}  // namespace dz
