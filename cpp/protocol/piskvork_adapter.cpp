#include "protocol/piskvork_adapter.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>

#include "game/position_spec.h"

namespace dz::protocol {

namespace {

std::string upper(std::string s) {
  for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

// Splits "a,b c" style argument lists (commas and spaces are both separators).
std::vector<std::string> split_args(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

bool to_int(const std::string& s, int& v) {
  if (s.empty() || s.size() > 11) return false;
  const char* b = s.data();
  const char* e = s.data() + s.size();
  auto [p, ec] = std::from_chars(b, e, v);
  return ec == std::errc() && p == e;
}

}  // namespace

std::optional<Point> gomocup_to_point(int x, int y) {
  if (x < 0 || x >= kBoardSize || y < 0 || y >= kBoardSize) return std::nullopt;
  return make_point(kBoardSize - 1 - y, x);  // y counts rows from the top
}

void point_to_gomocup(Point p, int& x, int& y) {
  x = col_of(p);
  y = kBoardSize - 1 - row_of(p);
}

std::string format_move(Point p) {
  int x, y;
  point_to_gomocup(p, x, y);
  return std::to_string(x) + "," + std::to_string(y);
}

std::string format_board_line(Point p, int field) { return format_move(p) + "," + std::to_string(field); }

Command parse_command(const std::string& line) {
  Command c;
  c.raw = trim(line);
  const size_t sp = c.raw.find(' ');
  const std::string head = upper(c.raw.substr(0, sp));
  const std::string rest = sp == std::string::npos ? "" : trim(c.raw.substr(sp + 1));
  const std::vector<std::string> args = split_args(rest);
  auto two_ints = [&](int& a, int& b) { return args.size() >= 2 && to_int(args[0], a) && to_int(args[1], b); };
  if (head == "START") {
    if (args.size() == 1 && to_int(args[0], c.size)) c.type = CommandType::START;
  } else if (head == "RECTSTART") {
    if (two_ints(c.width, c.height)) c.type = CommandType::RECTSTART;
  } else if (head == "RESTART") {
    c.type = CommandType::RESTART;
  } else if (head == "BEGIN") {
    c.type = CommandType::BEGIN;
  } else if (head == "TURN") {
    if (args.size() == 2 && two_ints(c.x, c.y)) c.type = CommandType::TURN;
  } else if (head == "TAKEBACK") {
    if (args.size() == 2 && two_ints(c.x, c.y)) c.type = CommandType::TAKEBACK;
  } else if (head == "BOARD") {
    c.type = CommandType::BOARD;
  } else if (head == "INFO") {
    const size_t s2 = rest.find(' ');
    c.key = rest.substr(0, s2);
    for (char& ch : c.key) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    c.value = s2 == std::string::npos ? "" : trim(rest.substr(s2 + 1));
    if (!c.key.empty()) c.type = CommandType::INFO;
  } else if (head == "END") {
    c.type = CommandType::END;
  } else if (head == "ABOUT") {
    c.type = CommandType::ABOUT;
  }
  return c;
}

Reply parse_reply(const std::string& line) {
  Reply r;
  const std::string t = trim(line);
  r.text = t;
  const size_t sp = t.find(' ');
  const std::string head = upper(t.substr(0, sp));
  const std::string rest = sp == std::string::npos ? "" : trim(t.substr(sp + 1));
  if (head == "OK" && rest.empty()) { r.kind = ReplyKind::OK; return r; }
  if (head == "ERROR") { r.kind = ReplyKind::ERROR; r.text = rest; return r; }
  if (head == "UNKNOWN") { r.kind = ReplyKind::UNKNOWN; r.text = rest; return r; }
  if (head == "MESSAGE") { r.kind = ReplyKind::MESSAGE; r.text = rest; return r; }
  if (head == "DEBUG") { r.kind = ReplyKind::DEBUG; r.text = rest; return r; }
  if (head == "SUGGEST") {
    const auto a = split_args(rest);
    if (a.size() == 2 && to_int(a[0], r.x) && to_int(a[1], r.y)) r.kind = ReplyKind::SUGGEST;
    return r;
  }
  // A move is exactly "x,y" (spaces tolerated around the comma).
  const auto a = split_args(t);
  int x, y;
  if (a.size() == 2 && t.find(',') != std::string::npos && to_int(a[0], x) && to_int(a[1], y)) {
    r.kind = ReplyKind::MOVE;
    r.x = x;
    r.y = y;
  }
  return r;
}

// ---------------------------------------------------------------------------------------------

namespace {
ProfilePtr gomocup_profile() { return make_profile(RuleProfile::renju_gomocup()); }
}  // namespace

EngineSession::EngineSession(ThinkFn think, std::string about) : think_(std::move(think)), about_(std::move(about)) {}

std::vector<std::string> EngineSession::think_and_reply() {
  if (!(rule_ & kRuleRenju) || (rule_ & kRuleContinuous))
    return {"ERROR DojoZero supports only Renju (INFO rule 4)"};
  if (!state_) return {"ERROR no game started"};
  if (state_->is_terminal()) return {"ERROR the game is already over"};
  const std::optional<Point> p = think_(*state_, limits_);
  if (!p || !state_->apply(Action::place(*p))) return {"ERROR engine produced no legal move"};
  return {format_move(*p)};
}

std::vector<std::string> EngineSession::finish_board() {
  in_board_ = false;
  std::vector<Point> own, opp;
  Point last_opp = kNoPoint;
  for (const auto& [p, f] : board_lines_) {
    if (f == 1) own.push_back(p);
    else if (f == 2) { opp.push_back(p); last_opp = p; }
    else return {"ERROR BOARD field must be 1 or 2 (continuous games are not supported)"};
  }
  // Renju: black moves first, so stone counts fix the colours (ASSUMPTIONS A-P3).
  PositionSpec spec;
  if (own.size() == opp.size()) {
    spec.black = own;
    spec.white = opp;
    spec.side_to_move = Color::BLACK;
  } else if (opp.size() == own.size() + 1) {
    spec.black = opp;
    spec.white = own;
    spec.side_to_move = Color::WHITE;
  } else {
    return {"ERROR BOARD stone counts do not fit a Renju game with this engine to move"};
  }
  spec.last_move = last_opp;
  std::string error;
  auto g = load_position(spec, gomocup_profile(), &error);
  if (!g) return {"ERROR invalid BOARD position: " + error};
  state_ = std::move(*g);
  return think_and_reply();
}

std::vector<std::string> EngineSession::feed(const std::string& line) {
  if (in_board_) {
    const std::string t = trim(line);
    if (upper(t) == "DONE") return finish_board();
    const auto a = split_args(t);
    int x, y, f;
    if (a.size() != 3 || !to_int(a[0], x) || !to_int(a[1], y) || !to_int(a[2], f)) {
      in_board_ = false;
      return {"ERROR malformed BOARD line: " + t};
    }
    const auto p = gomocup_to_point(x, y);
    if (!p) {
      in_board_ = false;
      return {"ERROR BOARD point off the board: " + t};
    }
    board_lines_.emplace_back(*p, f);
    return {};
  }
  const Command c = parse_command(line);
  switch (c.type) {
    case CommandType::START:
      if (c.size != kBoardSize) return {"ERROR only 15x15 boards are supported"};
      state_.emplace(gomocup_profile());
      return {"OK"};
    case CommandType::RECTSTART:
      return {"ERROR rectangular boards are not supported"};
    case CommandType::RESTART:
      state_.emplace(gomocup_profile());
      return {"OK"};
    case CommandType::BEGIN:
      if (!state_ || state_->board().stone_count() != 0) return {"ERROR BEGIN needs an empty board after START"};
      return think_and_reply();
    case CommandType::TURN: {
      if (!state_) return {"ERROR no game started"};
      const auto p = gomocup_to_point(c.x, c.y);
      if (!p || !state_->apply(Action::place(*p))) return {"ERROR illegal opponent move " + c.raw};
      return think_and_reply();
    }
    case CommandType::TAKEBACK: {
      const auto p = gomocup_to_point(c.x, c.y);
      if (!state_ || !p || !state_->can_undo() || state_->last_move() != *p) return {"ERROR cannot take back " + c.raw};
      state_->undo();
      return {"OK"};
    }
    case CommandType::BOARD:
      in_board_ = true;
      board_lines_.clear();
      return {};
    case CommandType::INFO: {
      int v = 0;
      const bool num = [&] {
        const std::string& s = c.value;
        if (s.empty() || s.size() > 11) return false;
        auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
        return ec == std::errc() && p == s.data() + s.size();
      }();
      if (!num) return {};
      if (c.key == "timeout_turn") limits_.timeout_turn_ms = v;
      else if (c.key == "timeout_match") limits_.timeout_match_ms = v;
      else if (c.key == "time_left") limits_.time_left_ms = v;
      else if (c.key == "max_memory") limits_.max_memory_bytes = v;
      else if (c.key == "rule") rule_ = v;
      return {};
    }
    case CommandType::END:
      quit_ = true;
      return {};
    case CommandType::ABOUT:
      return {about_};
    case CommandType::UNKNOWN:
      break;
  }
  return {"UNKNOWN " + c.raw};
}

}  // namespace dz::protocol
