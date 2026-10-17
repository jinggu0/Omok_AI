#pragma once

#include <filesystem>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "board/board.h"
#include "game/game_state.h"
#include "game/position_spec.h"

namespace dz::test {

std::filesystem::path source_dir();
Point P(const char* notation);  // aborts the test on a bad point
Board make_board(std::initializer_list<const char*> black, std::initializer_list<const char*> white = {});
nlohmann::json read_json(const std::filesystem::path& path);
std::vector<std::filesystem::path> json_files_in(const std::filesystem::path& dir);
ProfilePtr profile_by_name(const std::string& name);  // RENJU_RIF, TARAGUCHI10, OMOKDOJANG_DAN
// Board from a position spec (direct setup only, no validation beyond overlap).
Board board_from_spec(const PositionSpec& spec);

// Random positions biased towards dense black clusters (for fuzzing the forbidden checker).
Board random_dense_board(std::mt19937_64& rng);

}  // namespace dz::test
