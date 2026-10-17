#include "game/position_spec.h"

#include <algorithm>
#include <set>

#include <nlohmann/json.hpp>

#include "opening/symmetry.h"

namespace dz {

using nlohmann::json;

namespace {

bool fail(std::string* error, const std::string& msg) {
  if (error) *error = msg;
  return false;
}

bool parse_points(const json& arr, std::vector<Point>& out, const std::string& key, std::string* error) {
  if (!arr.is_array()) return fail(error, key + ": expected array");
  for (const auto& v : arr) {
    if (!v.is_string()) return fail(error, key + ": expected point strings");
    auto p = parse_point(v.get<std::string>());
    if (!p) return fail(error, key + ": bad point '" + v.get<std::string>() + "'");
    out.push_back(*p);
  }
  return true;
}

bool add_cell(char ch, Point p, PositionSpec& spec, std::string* error) {
  switch (ch) {
    case 'X': case 'x': case 'B': spec.black.push_back(p); return true;
    case 'O': case 'o': case 'W': spec.white.push_back(p); return true;
    case '.': case '_': case '+': return true;
    default: return fail(error, std::string("bad cell character '") + ch + "'");
  }
}

std::optional<PlayerId> parse_player(const json& v) {
  if (v == "P0") return PlayerId::P0;
  if (v == "P1") return PlayerId::P1;
  return std::nullopt;
}

// Longest run of `cell` through any stone of that color.
int longest_run(const Board& b, Cell cell) {
  int best = 0;
  for (int i = 0; i < kNumPoints; ++i) {
    const Point p = static_cast<Point>(i);
    if (b.at(p) != cell) continue;
    for (int d = 0; d < kNumDirections; ++d) {
      int n = 1;
      for (int k = 1;; ++k) {
        const int r = row_of(p) + k * kDirDRow[d], c = col_of(p) + k * kDirDCol[d];
        if (!on_board(r, c) || b.at(make_point(r, c)) != cell) break;
        ++n;
      }
      best = std::max(best, n);
    }
  }
  return best;
}

// Can the given stones be the opening moves with the given regions (some ordering)?
// regions[i] = Central(N) for the i-th stone of this color, 0 = anywhere.
bool regions_feasible(std::vector<Point> stones, const std::vector<int>& regions) {
  std::sort(stones.begin(), stones.end());
  do {
    bool ok = true;
    for (size_t i = 0; i < stones.size() && ok; ++i) {
      ok = regions[i] == 0 || in_central(stones[i], regions[i]);
    }
    if (ok) return true;
  } while (std::next_permutation(stones.begin(), stones.end()));
  return false;
}

}  // namespace

std::optional<PositionSpec> parse_position_spec(const json& j, std::string* error) {
  static const std::set<std::string> kKeys = {"moves", "black", "white", "lines", "diagram", "phase",
                                              "side", "assignment", "offered", "last_move"};
  if (!j.is_object()) { fail(error, "position: expected object"); return std::nullopt; }
  for (const auto& [k, v] : j.items()) {
    if (!kKeys.count(k)) { fail(error, "position: unknown key '" + k + "'"); return std::nullopt; }
  }
  PositionSpec spec;
  try {
    if (j.contains("moves")) spec.moves = j["moves"].get<std::vector<std::string>>();
    if (j.contains("black") && !parse_points(j["black"], spec.black, "black", error)) return std::nullopt;
    if (j.contains("white") && !parse_points(j["white"], spec.white, "white", error)) return std::nullopt;
    if (j.contains("lines")) {
      for (const auto& line : j["lines"]) {
        auto from = parse_point(line.at("from").get<std::string>());
        auto dir = parse_direction(line.at("dir").get<std::string>());
        const std::string cells = line.at("cells").get<std::string>();
        if (!from || !dir) { fail(error, "lines: bad from/dir"); return std::nullopt; }
        for (size_t i = 0; i < cells.size(); ++i) {
          const int r = row_of(*from) + static_cast<int>(i) * kDirDRow[*dir];
          const int c = col_of(*from) + static_cast<int>(i) * kDirDCol[*dir];
          if (!on_board(r, c)) {
            if (cells[i] == '.') continue;
            fail(error, "lines: stone off board");
            return std::nullopt;
          }
          if (!add_cell(cells[i], make_point(r, c), spec, error)) return std::nullopt;
        }
      }
    }
    if (j.contains("diagram")) {
      const auto rows = j["diagram"].get<std::vector<std::string>>();
      if (rows.size() != static_cast<size_t>(kBoardSize)) {
        fail(error, "diagram: expected 15 rows");
        return std::nullopt;
      }
      for (int i = 0; i < kBoardSize; ++i) {
        std::string row;
        for (char ch : rows[i]) if (ch != ' ') row += ch;
        if (row.size() != static_cast<size_t>(kBoardSize)) {
          fail(error, "diagram: row " + std::to_string(i) + " must have 15 cells");
          return std::nullopt;
        }
        for (int col = 0; col < kBoardSize; ++col) {
          if (!add_cell(row[col], make_point(kBoardSize - 1 - i, col), spec, error)) return std::nullopt;
        }
      }
    }
    if (j.contains("phase")) {
      spec.phase = parse_phase(j["phase"].get<std::string>());
      if (!spec.phase) { fail(error, "bad phase"); return std::nullopt; }
    }
    if (j.contains("side")) {
      const std::string s = j["side"].get<std::string>();
      if (s == "BLACK") spec.side_to_move = Color::BLACK;
      else if (s == "WHITE") spec.side_to_move = Color::WHITE;
      else { fail(error, "bad side"); return std::nullopt; }
    }
    if (j.contains("assignment")) {
      auto b = parse_player(j["assignment"].at("black"));
      auto w = parse_player(j["assignment"].at("white"));
      if (!b || !w) { fail(error, "bad assignment"); return std::nullopt; }
      spec.assignment = {*b, *w};
    }
    if (j.contains("offered") && !parse_points(j["offered"], spec.offered, "offered", error)) return std::nullopt;
    if (j.contains("last_move")) {
      auto p = parse_point(j["last_move"].get<std::string>());
      if (!p) { fail(error, "bad last_move"); return std::nullopt; }
      spec.last_move = *p;
    }
  } catch (const json::exception& e) {
    fail(error, std::string("position json error: ") + e.what());
    return std::nullopt;
  }
  return spec;
}

std::optional<GameState> load_position(const PositionSpec& spec, ProfilePtr profile, std::string* error) {
  if (!spec.assignment.valid()) { fail(error, "assignment: both colors held by one player"); return std::nullopt; }

  if (spec.moves) {
    if (!spec.black.empty() || !spec.white.empty() || spec.phase || !spec.offered.empty()) {
      fail(error, "moves cannot be combined with a direct setup");
      return std::nullopt;
    }
    GameState g(profile, spec.assignment);
    for (size_t i = 0; i < spec.moves->size(); ++i) {
      const std::string& s = (*spec.moves)[i];
      auto a = parse_action_in_phase(s, g.phase());
      if (!a) { fail(error, "move " + std::to_string(i + 1) + ": cannot parse '" + s + "'"); return std::nullopt; }
      if (!g.apply(*a)) {
        fail(error, "move " + std::to_string(i + 1) + " '" + s + "' is illegal in phase " + phase_name(g.phase()));
        return std::nullopt;
      }
    }
    return g;
  }

  Setup s;
  s.black = spec.black;
  s.white = spec.white;
  s.assignment = spec.assignment;
  s.offered = spec.offered;
  s.last_move = spec.last_move;
  s.phase = spec.phase.value_or(OpeningPhase::NORMAL_RENJU);

  std::set<Point> seen;
  for (Point p : s.black) {
    if (!seen.insert(p).second) { fail(error, "duplicate stone at " + to_notation(p)); return std::nullopt; }
  }
  for (Point p : s.white) {
    if (!seen.insert(p).second) { fail(error, "duplicate stone at " + to_notation(p)); return std::nullopt; }
  }
  if (s.last_move != kNoPoint && !seen.count(s.last_move)) {
    fail(error, "last_move is not a stone on the board");
    return std::nullopt;
  }

  const int nb = static_cast<int>(s.black.size()), nw = static_cast<int>(s.white.size());
  const bool taraguchi = profile->opening_rule == OpeningRule::TARAGUCHI_10;
  if (s.phase != OpeningPhase::NORMAL_RENJU && !taraguchi) {
    fail(error, "opening phases require the TARAGUCHI_10 opening rule");
    return std::nullopt;
  }

  if (s.phase == OpeningPhase::NORMAL_RENJU) {
    if (spec.side_to_move) {
      s.side_to_move = *spec.side_to_move;
    } else if (nb == nw) {
      s.side_to_move = Color::BLACK;
    } else if (nb == nw + 1) {
      s.side_to_move = Color::WHITE;
    } else {
      fail(error, "cannot infer side to move from stone counts");
      return std::nullopt;
    }
    if (!profile->pass_allowed) {
      const bool ok = s.side_to_move == Color::BLACK ? nb == nw : nb == nw + 1;
      if (!ok) { fail(error, "stone counts inconsistent with side to move (passes not allowed)"); return std::nullopt; }
    }
    if (taraguchi && (nb < 3 || nw < 3)) {
      fail(error, "NORMAL_RENJU under Taraguchi-10 needs at least 3 stones of each color");
      return std::nullopt;
    }
    if (!s.offered.empty()) { fail(error, "offered proposals only exist in proposal phases"); return std::nullopt; }
  } else {
    int eb = 0, ew = 0;
    phase_stone_counts(s.phase, eb, ew);
    if (nb != eb || nw != ew) {
      fail(error, std::string("phase ") + phase_name(s.phase) + " requires " + std::to_string(eb) +
                      " black / " + std::to_string(ew) + " white stones");
      return std::nullopt;
    }
    const auto& reg = profile->taraguchi.region;
    // Black: move 1 (region[0]), move 3 (region[2]), move 5 (region[4] if direct; a selected
    // proposal may be anywhere, which we cannot distinguish in WHITE_MOVE_6).
    std::vector<int> black_regions = {reg[0], reg[2],
                                      s.phase == OpeningPhase::WHITE_MOVE_6 ? 0 : reg[4]};
    black_regions.resize(nb);
    std::vector<int> white_regions = {reg[1], reg[3]};
    white_regions.resize(nw);
    if (!regions_feasible(s.black, black_regions) || !regions_feasible(s.white, white_regions)) {
      fail(error, "stones violate the Taraguchi-10 opening regions");
      return std::nullopt;
    }
    if (s.phase == OpeningPhase::OFFER_TEN_5TH_MOVES || s.phase == OpeningPhase::SELECT_ONE_OF_TEN) {
      Board b;
      for (Point p : s.black) b.place(p, Color::BLACK);
      for (Point p : s.white) b.place(p, Color::WHITE);
      const int n = profile->taraguchi.n_proposals;
      const int k = static_cast<int>(s.offered.size());
      if (s.phase == OpeningPhase::SELECT_ONE_OF_TEN ? k != n : k >= n) {
        fail(error, "offered set has the wrong size for the phase");
        return std::nullopt;
      }
      for (int i = 0; i < k; ++i) {
        std::string why;
        if (!proposal_admissible(b, std::span<const Point>(s.offered).first(i), s.offered[i],
                                 profile->taraguchi, profile->black, &why)) {
          fail(error, "invalid offered set: " + why);
          return std::nullopt;
        }
      }
    } else if (!s.offered.empty()) {
      fail(error, "offered proposals only exist in proposal phases");
      return std::nullopt;
    }
  }

  Board b;
  for (Point p : s.black) b.place(p, Color::BLACK);
  for (Point p : s.white) b.place(p, Color::WHITE);
  if (longest_run(b, BLACK_STONE) >= 5 || longest_run(b, WHITE_STONE) >= 5) {
    fail(error, "position already contains a five or overline");
    return std::nullopt;
  }
  return GameState(profile, s);
}

}  // namespace dz
