#include "protocol/external_engine.h"

#include <chrono>
#include <thread>

namespace dz::protocol {

const char* failure_name(Failure f) {
  switch (f) {
    case Failure::NONE: return "NONE";
    case Failure::CRASH: return "CRASH";
    case Failure::TIMEOUT: return "TIMEOUT";
    case Failure::MALFORMED_RESPONSE: return "MALFORMED_RESPONSE";
    case Failure::PROTOCOL_ERROR: return "PROTOCOL_ERROR";
    case Failure::START_FAILED: return "START_FAILED";
  }
  return "?";
}

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

bool ExternalEngine::send(const std::string& line) { return proc_.write_line(line); }

Failure ExternalEngine::start(std::string* detail) {
  GomocupProcess::Options o;
  o.path = cfg_.path;
  o.args = cfg_.args;
  o.working_dir = cfg_.working_dir;
  o.crlf = cfg_.crlf;
  std::string err;
  if (!proc_.start(o, &err)) {
    if (detail) *detail = err;
    return Failure::START_FAILED;
  }
  started_ = true;
  bool ok = send("INFO timeout_turn " + std::to_string(cfg_.timeout_turn_ms));
  first_move_ = true;
  match_ms_ = cfg_.timeout_match_ms > 0 ? cfg_.timeout_match_ms : 250LL * cfg_.timeout_turn_ms;
  used_ms_ = 0;
  ok &= send("INFO timeout_match " + std::to_string(match_ms_));
  ok &= send("INFO time_left " + std::to_string(match_ms_));
  if (cfg_.max_memory_bytes > 0) ok &= send("INFO max_memory " + std::to_string(cfg_.max_memory_bytes));
  ok &= send("INFO rule " + std::to_string(cfg_.rule));
  for (const auto& [k, v] : cfg_.extra_info) ok &= send("INFO " + k + " " + v);
  ok &= send("START 15");
  if (!ok) {
    if (detail) *detail = "could not write to the engine";
    return Failure::CRASH;
  }
  const auto t0 = Clock::now();
  for (;;) {
    const int left = static_cast<int>(cfg_.start_timeout_ms - ms_since(t0));
    std::string line;
    const auto st = proc_.read_line(std::max(0, left), line);
    if (st == GomocupProcess::ReadStatus::TIMEOUT) {
      if (detail) *detail = "no OK after START";
      return proc_.alive() ? Failure::TIMEOUT : Failure::CRASH;
    }
    if (st == GomocupProcess::ReadStatus::CLOSED) {
      if (detail) *detail = "engine exited during START: " + proc_.stderr_tail();
      return Failure::CRASH;
    }
    if (st == GomocupProcess::ReadStatus::TOO_LONG) {
      if (detail) *detail = "over-long line during START";
      return Failure::MALFORMED_RESPONSE;
    }
    const Reply r = parse_reply(line);
    if (r.kind == ReplyKind::OK) return Failure::NONE;
    if (r.kind == ReplyKind::MESSAGE || r.kind == ReplyKind::DEBUG) continue;
    if (detail) *detail = "START reply: " + line;
    return r.kind == ReplyKind::ERROR || r.kind == ReplyKind::UNKNOWN ? Failure::PROTOCOL_ERROR
                                                                      : Failure::MALFORMED_RESPONSE;
  }
}

EngineReply ExternalEngine::await_move(double budget_ms) {
  EngineReply out;
  const auto t0 = Clock::now();
  for (;;) {
    const int left = static_cast<int>(budget_ms - ms_since(t0));
    std::string line;
    const auto st = proc_.read_line(std::max(0, left), line);
    out.elapsed_ms = ms_since(t0);
    if (st == GomocupProcess::ReadStatus::TIMEOUT) {
      out.failure = proc_.alive() ? Failure::TIMEOUT : Failure::CRASH;
      out.detail = out.failure == Failure::CRASH ? "engine exited: " + proc_.stderr_tail() : "no move within the time limit";
      return out;
    }
    if (st == GomocupProcess::ReadStatus::CLOSED) {
      out.failure = Failure::CRASH;
      const auto code = proc_.exit_code();
      out.detail = "engine closed stdout" + (code ? " (exit code " + std::to_string(*code) + ")" : std::string()) +
                   ": " + proc_.stderr_tail();
      return out;
    }
    if (st == GomocupProcess::ReadStatus::TOO_LONG) {
      out.failure = Failure::MALFORMED_RESPONSE;
      out.detail = "over-long line";
      return out;
    }
    const Reply r = parse_reply(line);
    switch (r.kind) {
      case ReplyKind::MESSAGE:
      case ReplyKind::DEBUG:
      case ReplyKind::SUGGEST:
        out.messages.push_back(line);
        continue;
      case ReplyKind::MOVE:
        out.x = r.x;
        out.y = r.y;
        out.move = gomocup_to_point(r.x, r.y);
        if (!out.move) {
          out.failure = Failure::MALFORMED_RESPONSE;
          out.detail = "move off the board: " + line;
        }
        return out;
      case ReplyKind::ERROR:
      case ReplyKind::UNKNOWN:
      case ReplyKind::OK:
        out.failure = Failure::PROTOCOL_ERROR;
        out.detail = line;
        return out;
      case ReplyKind::OTHER:
        out.failure = Failure::MALFORMED_RESPONSE;
        out.detail = line;
        return out;
    }
  }
}

EngineReply ExternalEngine::timed_move(double budget_ms) {
  if (first_move_) budget_ms += cfg_.first_move_slack_ms;
  first_move_ = false;
  EngineReply r = await_move(budget_ms);
  used_ms_ += r.elapsed_ms;
  return r;
}

EngineReply ExternalEngine::begin() {
  if (!send_time_left() || !send("BEGIN")) return {Failure::CRASH, std::nullopt, -1, -1, 0, "could not write BEGIN", {}};
  return timed_move(cfg_.timeout_turn_ms + cfg_.slack_ms);
}

EngineReply ExternalEngine::turn(Point p) {
  if (!send_time_left() || !send("TURN " + format_move(p))) return {Failure::CRASH, std::nullopt, -1, -1, 0, "could not write TURN", {}};
  return timed_move(cfg_.timeout_turn_ms + cfg_.slack_ms);
}

EngineReply ExternalEngine::board(const std::vector<Point>& moves) {
  // After an even number of moves black is to move. Field 1 = the engine's (side to move) stones.
  const size_t mover_parity = moves.size() % 2;  // stones with index parity == mover_parity are the mover's
  bool ok = send_time_left();
  ok &= send("BOARD");
  for (size_t i = 0; i < moves.size(); ++i) ok &= send(format_board_line(moves[i], (i % 2) == mover_parity ? 1 : 2));
  ok &= send("DONE");
  if (!ok) return {Failure::CRASH, std::nullopt, -1, -1, 0, "could not write BOARD", {}};
  return timed_move(cfg_.timeout_turn_ms + cfg_.slack_ms);
}

bool ExternalEngine::send_time_left() {
  const long long left = std::max<long long>(0, match_ms_ - static_cast<long long>(used_ms_));
  return send("INFO time_left " + std::to_string(left));
}

Failure ExternalEngine::restart(std::string* detail) {
  used_ms_ = 0;
  first_move_ = true;
  if (!started_ || !send("RESTART")) {
    if (detail) *detail = "could not write RESTART";
    return Failure::CRASH;
  }
  const auto t0 = Clock::now();
  for (;;) {
    std::string line;
    const auto st = proc_.read_line(std::max(0, static_cast<int>(cfg_.start_timeout_ms - ms_since(t0))), line);
    if (st == GomocupProcess::ReadStatus::TIMEOUT) return proc_.alive() ? Failure::TIMEOUT : Failure::CRASH;
    if (st == GomocupProcess::ReadStatus::CLOSED) return Failure::CRASH;
    if (st == GomocupProcess::ReadStatus::TOO_LONG) return Failure::MALFORMED_RESPONSE;
    const Reply r = parse_reply(line);
    if (r.kind == ReplyKind::OK) return Failure::NONE;
    if (r.kind == ReplyKind::MESSAGE || r.kind == ReplyKind::DEBUG) continue;
    if (detail) *detail = "RESTART reply: " + line;
    return Failure::PROTOCOL_ERROR;
  }
}

void ExternalEngine::set_time_left(int ms) { send("INFO time_left " + std::to_string(ms)); }

std::string ExternalEngine::about() {
  if (!send("ABOUT")) return "";
  std::string line;
  for (int i = 0; i < 20; ++i) {
    if (proc_.read_line(2000, line) != GomocupProcess::ReadStatus::LINE) return "";
    const Reply r = parse_reply(line);
    if (r.kind != ReplyKind::MESSAGE && r.kind != ReplyKind::DEBUG) return line;
  }
  return "";
}

void ExternalEngine::end() {
  if (!started_) return;
  send("END");
  for (int i = 0; i < 50 && proc_.alive(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  proc_.kill();
  started_ = false;
}

}  // namespace dz::protocol
