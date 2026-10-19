// Gomocup engine client (ARCHITECTURE.md §3.12, PRD FR-20/FR-21a).
//
// ExternalEngine = GomocupProcess + codec. It reports what the engine said and how it failed; it
// does NOT judge legality — the caller (TournamentRunner) applies every move to the canonical
// GameState and classifies illegal / forbidden moves itself.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "game/game_state.h"
#include "protocol/gomocup_process.h"
#include "protocol/piskvork_adapter.h"

namespace dz::protocol {

enum class Failure { NONE, CRASH, TIMEOUT, MALFORMED_RESPONSE, PROTOCOL_ERROR, START_FAILED };
const char* failure_name(Failure f);

struct EngineConfig {
  std::string path;
  std::vector<std::string> args;
  std::string working_dir;
  std::string expected_sha256;  // empty = not checked here (the Python registry checks hashes)
  int rule = kRuleRenju;
  int timeout_turn_ms = 1000;
  int timeout_match_ms = 0;     // 0 = no match clock sent
  long long max_memory_bytes = 0;
  std::vector<std::pair<std::string, std::string>> extra_info;  // engine-specific INFO key value
  int start_timeout_ms = 10000;  // START -> OK (engines may load weights)
  int slack_ms = 1000;           // allowed beyond timeout_turn before TIMEOUT
  bool crlf = true;
};

struct EngineReply {
  Failure failure = Failure::NONE;
  std::optional<Point> move;     // set when failure == NONE
  int x = -1, y = -1;            // raw protocol coordinates (also for off-board moves)
  double elapsed_ms = 0;
  std::string detail;            // failure description / offending line
  std::vector<std::string> messages;  // MESSAGE / DEBUG lines seen while waiting
};

class ExternalEngine {
 public:
  explicit ExternalEngine(EngineConfig cfg) : cfg_(std::move(cfg)) {}

  // Launch, send INFO settings and START 15; expects OK. Returns NONE or the failure.
  Failure start(std::string* detail = nullptr);
  // Engine plays the first move on the empty board.
  EngineReply begin();
  // Opponent played `p`; engine replies.
  EngineReply turn(Point p);
  // Set up `moves` (in play order, alternating colours from black) and ask for a move; the engine
  // is the side to move after `moves`.
  EngineReply board(const std::vector<Point>& moves);
  // New game in the same process (RESTART -> OK). NONE or the failure.
  Failure restart(std::string* detail = nullptr);
  void set_time_left(int ms);
  std::string about();  // "" on failure
  void end();           // END, short grace period, then kill
  bool alive() { return proc_.alive(); }
  std::string stderr_tail() { return proc_.stderr_tail(); }
  long long pid() const { return proc_.pid(); }

 private:
  EngineReply await_move(double budget_ms);
  bool send(const std::string& line);

  EngineConfig cfg_;
  GomocupProcess proc_;
  bool started_ = false;
};

}  // namespace dz::protocol
