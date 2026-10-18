#include "tactics/verify.h"

#include "tactics/threats.h"

namespace dz::tactics {

const char* verify_status_name(VerifyStatus s) {
  switch (s) {
    case VerifyStatus::VERIFIED: return "VERIFIED";
    case VerifyStatus::REFUTED: return "REFUTED";
    case VerifyStatus::TOO_LARGE: return "TOO_LARGE";
  }
  return "?";
}

namespace {

class Verifier {
 public:
  Verifier(GameState g, Color attacker, const ProofMoveFn& proof, const VerifyLimits& limits,
           const VerifyHints& hints)
      : g_(std::move(g)), attacker_(attacker), proof_(proof), limits_(limits), hints_(hints) {}

  VerifyStatus run() {
    const VerifyStatus s = g_.side_to_move() == attacker_ ? attacker_node(0) : defender_node(0);
    return over_ ? VerifyStatus::TOO_LARGE : s;
  }

 private:
  bool attacker_won() const {
    const GameResult r = g_.outcome().result;
    return (r == GameResult::BLACK_WIN && attacker_ == Color::BLACK) ||
           (r == GameResult::WHITE_WIN && attacker_ == Color::WHITE);
  }

  bool step(const Action& a) {
    if (++applies_ > limits_.max_nodes) over_ = true;
    return g_.apply(a);
  }

  bool wins_by(const Action& a) {
    if (!step(a)) return false;
    const bool win = attacker_won();
    g_.undo();
    return win;
  }

  // Attacker to move: does some legal PLACE win at once? With hints, only the hinted moves are
  // tried (a missed hint can only make verification fail, never succeed wrongly).
  bool immediate_win() {
    if (hints_.immediate_wins) {
      for (Point p : hints_.immediate_wins(g_)) {
        if (wins_by(Action::place(p))) return true;
        if (over_) return false;
      }
      return false;
    }
    for (const Action& a : g_.legal_actions()) {
      if (a.type != ActionType::PLACE) continue;
      if (!step(a)) continue;
      const bool win = attacker_won();
      g_.undo();
      if (win) return true;
      if (over_) return false;
    }
    return false;
  }

  // Attacker plays a; does every defender reply leave an immediate attacker win?
  bool double_threat(const Action& a) {
    if (!step(a)) return false;
    bool ok = !g_.is_terminal() || attacker_won();
    if (ok && !g_.is_terminal()) {
      for (const Action& d : g_.legal_actions()) {
        if (!step(d)) { ok = false; break; }
        const bool good = !g_.is_terminal() && immediate_win();
        g_.undo();
        if (!good || over_) { ok = false; break; }
      }
    }
    g_.undo();
    return ok;
  }

  VerifyStatus attacker_node(int depth) {
    if (over_) return VerifyStatus::TOO_LARGE;
    if (g_.is_terminal()) return attacker_won() ? VerifyStatus::VERIFIED : VerifyStatus::REFUTED;
    if (depth > limits_.max_depth) { over_ = true; return VerifyStatus::TOO_LARGE; }
    if (immediate_win()) return VerifyStatus::VERIFIED;
    if (auto m = proof_(position_key(g_.board(), g_.side_to_move()))) {
      if (!step(Action::place(*m))) return VerifyStatus::REFUTED;  // proof move is illegal
      const VerifyStatus s = defender_node(depth + 1);
      g_.undo();
      return s;
    }
    if (hints_.double_threats) {
      for (Point e : hints_.double_threats(g_)) {
        if (double_threat(Action::place(e))) return VerifyStatus::VERIFIED;
        if (over_) return VerifyStatus::TOO_LARGE;
      }
    }
    for (const Action& a : g_.legal_actions()) {  // fallback: every legal move
      if (a.type != ActionType::PLACE) continue;
      if (double_threat(a)) return VerifyStatus::VERIFIED;
      if (over_) return VerifyStatus::TOO_LARGE;
    }
    return VerifyStatus::REFUTED;
  }

  VerifyStatus defender_node(int depth) {
    if (over_) return VerifyStatus::TOO_LARGE;
    if (g_.is_terminal()) return attacker_won() ? VerifyStatus::VERIFIED : VerifyStatus::REFUTED;
    const auto actions = g_.legal_actions();
    if (actions.empty()) return VerifyStatus::REFUTED;
    for (const Action& d : actions) {
      if (!step(d)) return VerifyStatus::REFUTED;
      const VerifyStatus s = attacker_node(depth + 1);
      g_.undo();
      if (s != VerifyStatus::VERIFIED) return s;
    }
    return VerifyStatus::VERIFIED;
  }

  GameState g_;
  Color attacker_;
  const ProofMoveFn& proof_;
  VerifyLimits limits_;
  const VerifyHints& hints_;
  uint64_t applies_ = 0;
  bool over_ = false;
};

}  // namespace

VerifyHints threat_detector_hints() {
  VerifyHints h;
  h.immediate_wins = [](const GameState& g) {
    std::vector<Point> out;
    five_points(g.board(), g.side_to_move(), Rules{g.profile().black, g.profile().pass_allowed}, out);
    return out;
  };
  h.double_threats = [](const GameState& g) {
    std::vector<Point> out;
    Board b = g.board();
    double_threat_moves(b, g.side_to_move(), Rules{g.profile().black, g.profile().pass_allowed}, out);
    return out;
  };
  return h;
}

VerifyStatus verify_forced_win(const GameState& g, const ProofMoveFn& proof_move, const VerifyLimits& limits,
                               const VerifyHints& hints) {
  return Verifier(g, g.side_to_move(), proof_move, limits, hints).run();
}

VerifyStatus verify_forced_loss(const GameState& g, const ProofMoveFn& proof_move, const VerifyLimits& limits,
                                const VerifyHints& hints) {
  return Verifier(g, opposite(g.side_to_move()), proof_move, limits, hints).run();
}

std::unordered_map<uint64_t, Point> proof_from_pv(const GameState& g, const std::vector<Point>& pv) {
  std::unordered_map<uint64_t, Point> out;
  GameState s = g;
  const Color attacker = g.side_to_move();
  for (Point m : pv) {
    if (m == kNoPoint) break;  // "any defender move": the attacker's completion is an immediate win
    if (s.side_to_move() == attacker) out.emplace(position_key(s.board(), s.side_to_move()), m);
    if (!s.apply(Action::place(m))) break;
  }
  return out;
}

}  // namespace dz::tactics
