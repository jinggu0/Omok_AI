add_executable(dojozero_tests
  common/test_util.cpp
  rules/test_coord_board.cpp
  rules/test_game_rules.cpp
  rules/test_position_spec.cpp
  rules/test_profile.cpp
  forbidden/test_forbidden_cases.cpp
  forbidden/test_forbidden_props.cpp
  forbidden/test_line_enum.cpp
  opening/test_opening.cpp
  opening/test_symmetry.cpp
  compatibility/test_compat.cpp
  tactics/test_threats.cpp
  tactics/test_solvers.cpp
)
target_include_directories(dojozero_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(dojozero_tests PRIVATE DOJOZERO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(dojozero_tests PRIVATE dojozero_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(dojozero_tests DISCOVERY_TIMEOUT 120 PROPERTIES TIMEOUT 1800)
