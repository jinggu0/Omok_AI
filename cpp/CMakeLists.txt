set(DOJOZERO_CORE_SOURCES
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
  features/features.cpp
  search/search.cpp
  search/bots.cpp
  protocol/piskvork_adapter.cpp
  protocol/gomocup_process.cpp
  protocol/external_engine.cpp
  tactics/tprof.cpp
  tactics/reference/ref_threats.cpp
  tactics/reference/ref_vcf.cpp
  tactics/reference/ref_dfpn.cpp
  tactics/reference/ref_engine.cpp
)
add_library(dojozero_core STATIC ${DOJOZERO_CORE_SOURCES})
target_include_directories(dojozero_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dojozero_core PUBLIC nlohmann_json::nlohmann_json)

if(WIN32)
  # Gomocup/Piskvork engine exe for GUIs: a launcher around the Python host (ADR-08 interim). No core
  # dependency and a static CRT, so the exe can be copied next to a GUI.
  add_executable(pbrain_dojozero cli/pbrain_launcher.cpp)
  set_target_properties(pbrain_dojozero PROPERTIES OUTPUT_NAME pbrain-DojoZero
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(pbrain_dojozero PRIVATE DOJOZERO_REPO_DIR="${PROJECT_SOURCE_DIR}")
endif()

if(DOJOZERO_BUILD_TOOLS)
  # Profiling build of the same sources (DZ_TPROF timers; never linked into production targets).
  add_library(dojozero_core_tprof STATIC ${DOJOZERO_CORE_SOURCES})
  target_include_directories(dojozero_core_tprof PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(dojozero_core_tprof PUBLIC nlohmann_json::nlohmann_json)
  target_compile_definitions(dojozero_core_tprof PUBLIC DZ_TPROF)
  add_executable(dojozero_tprof tools/tactics_profile.cpp)
  target_link_libraries(dojozero_tprof PRIVATE dojozero_core_tprof)
  add_executable(dojozero_tbench tools/tactics_profile.cpp)
  target_link_libraries(dojozero_tbench PRIVATE dojozero_core)

  add_executable(dojozero_engine cli/main.cpp)
  target_link_libraries(dojozero_engine PRIVATE dojozero_core)

  add_executable(dojozero_bench tools/bench_rules.cpp)
  target_link_libraries(dojozero_bench PRIVATE dojozero_core)

  add_executable(dojozero_fuzz tools/fuzz_rules.cpp)
  target_link_libraries(dojozero_fuzz PRIVATE dojozero_core)

  add_executable(dojozero_tactics tools/tactics_tool.cpp)
  target_link_libraries(dojozero_tactics PRIVATE dojozero_core)

  add_executable(dojozero_bench_search tools/bench_search.cpp)
  target_link_libraries(dojozero_bench_search PRIVATE dojozero_core)

  add_executable(dojozero_datagen tools/datagen.cpp)
  target_link_libraries(dojozero_datagen PRIVATE dojozero_core)
endif()
