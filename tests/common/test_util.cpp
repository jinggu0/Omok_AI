#include "common/test_util.h"

#include <algorithm>
#include <fstream>

#include "tools/random_positions.h"

#include <gtest/gtest.h>

namespace dz::test {

std::filesystem::path source_dir() { return std::filesystem::path(DOJOZERO_SOURCE_DIR); }

Point P(const char* notation) {
  auto p = parse_point(notation);
  if (!p) {
    ADD_FAILURE() << "bad point " << notation;
    return 0;
  }
  return *p;
}

Board make_board(std::initializer_list<const char*> black, std::initializer_list<const char*> white) {
  Board b;
  for (const char* s : black) b.place(P(s), Color::BLACK);
  for (const char* s : white) b.place(P(s), Color::WHITE);
  return b;
}

nlohmann::json read_json(const std::filesystem::path& path) {
  std::ifstream in(path);
  return nlohmann::json::parse(in);
}

std::vector<std::filesystem::path> json_files_in(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> out;
  if (!std::filesystem::exists(dir)) return out;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    if (e.is_regular_file() && e.path().extension() == ".json") out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

ProfilePtr profile_by_name(const std::string& name) {
  if (name == "RENJU_RIF") return make_profile(RuleProfile::renju_rif());
  if (name == "TARAGUCHI10") return make_profile(RuleProfile::taraguchi10());
  if (name == "OMOKDOJANG_DAN") return make_profile(RuleProfile::omokdojang_dan());
  ADD_FAILURE() << "unknown profile " << name;
  return make_profile(RuleProfile::renju_rif());
}

Board board_from_spec(const PositionSpec& spec) {
  Board b;
  for (Point p : spec.black) b.place(p, Color::BLACK);
  for (Point p : spec.white) b.place(p, Color::WHITE);
  return b;
}

Board random_dense_board(std::mt19937_64& rng) { return tools::random_dense_board(rng); }

}  // namespace dz::test
