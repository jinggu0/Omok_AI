add_library(dojozero_core STATIC
  board/coord.cpp
  board/transform.cpp
  board/zobrist.cpp
  board/board.cpp
  rules/forbidden.cpp
  rules/reference_rules.cpp
  rules/rule_profile.cpp
  opening/opening_phase.cpp
  opening/symmetry.cpp
  game/action.cpp
  game/game_state.cpp
  game/position_spec.cpp
  tactics/threats.cpp
  tactics/vcf.cpp
  tactics/dfpn.cpp
  tactics/verify.cpp
  tactics/engine.cpp
  tactics/benchmark.cpp
)
target_include_directories(dojozero_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dojozero_core PUBLIC nlohmann_json::nlohmann_json)

if(DOJOZERO_BUILD_TOOLS)
  add_executable(dojozero_engine cli/main.cpp)
  target_link_libraries(dojozero_engine PRIVATE dojozero_core)

  add_executable(dojozero_bench tools/bench_rules.cpp)
  target_link_libraries(dojozero_bench PRIVATE dojozero_core)

  add_executable(dojozero_fuzz tools/fuzz_rules.cpp)
  target_link_libraries(dojozero_fuzz PRIVATE dojozero_core)

  add_executable(dojozero_tactics tools/tactics_tool.cpp)
  target_link_libraries(dojozero_tactics PRIVATE dojozero_core)
endif()
