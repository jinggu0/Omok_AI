#include "game/game_state.h"

#include <algorithm>
#include <cassert>
#include <sstream>

#include "board/zobrist.h"
#include "opening/symmetry.h"

namespace dz {

const char* result_name(GameResult r) {
  switch (r) {
    case GameResult::ONGOING: return "ONGOING";
    case GameResult::BLACK_WIN: return "BLACK_WIN";
    case GameResult::WHITE_WIN: return "WHITE_WIN";
    case GameResult::DRAW: return "DRAW";
  }
  return "?";
}

const char* end_reason_name(EndReason r) {
  switch (r) {
    case EndReason::NONE: return "NONE";
    case EndReason::FIVE: return "FIVE";
    case EndReason::FORBIDDEN_MOVE: return "FORBIDDEN_MOVE";
    case EndReason::BOARD_FULL: return "BOARD_FULL";
    case EndReason::DOUBLE_PASS: return "DOUBLE_PASS";
    case EndReason::NO_LEGAL_MOVE: return "NO_LEGAL_MOVE";
  }
  return "?";
}

namespace {

// Phase transition table (RULES.md §8.3) for PLACE and SWAP/KEEP actions.
OpeningPhase phase_after_place(OpeningPhase ph) {
  switch (ph) {
    case OpeningPhase::BLACK_MOVE_1: return OpeningPhase::SWAP_AFTER_1;
    case OpeningPhase::WHITE_MOVE_2: return OpeningPhase::SWAP_AFTER_2;
    case OpeningPhase::BLACK_MOVE_3: return OpeningPhase::SWAP_AFTER_3;
    case OpeningPhase::WHITE_MOVE_4: return OpeningPhase::CHOICE_AFTER_4;
    case OpeningPhase::BLACK_DIRECT_MOVE_5: return OpeningPhase::SWAP_AFTER_DIRECT_5;
    case OpeningPhase::WHITE_MOVE_6: return OpeningPhase::NORMAL_RENJU;
    default: return OpeningPhase::NORMAL_RENJU;
  }
}

OpeningPhase phase_after_swap_decision(OpeningPhase ph) {
  switch (ph) {
    case OpeningPhase::SWAP_AFTER_1: return OpeningPhase::WHITE_MOVE_2;
    case OpeningPhase::SWAP_AFTER_2: return OpeningPhase::BLACK_MOVE_3;
    case OpeningPhase::SWAP_AFTER_3: return OpeningPhase::WHITE_MOVE_4;
    case OpeningPhase::SWAP_BEFORE_DIRECT_5: return OpeningPhase::BLACK_DIRECT_MOVE_5;
    case OpeningPhase::SWAP_AFTER_DIRECT_5: return OpeningPhase::WHITE_MOVE_6;
    default: assert(false); return ph;
  }
}

}  // namespace

GameState::GameState(ProfilePtr profile, SideAssignment initial)
    : profile_(std::move(profile)), assignment_(initial) {
  phase_ = profile_->opening_rule == OpeningRule::TARAGUCHI_10 ? OpeningPhase::BLACK_MOVE_1
                                                                : OpeningPhase::NORMAL_RENJU;
  initial_.phase = phase_;
  initial_.assignment = initial;
  initial_.side_to_move = Color::BLACK;
  history_.reserve(256);
}

GameState::GameState(ProfilePtr profile, const Setup& setup)
    : profile_(std::move(profile)),
      phase_(setup.phase),
      side_(setup.side_to_move),
      assignment_(setup.assignment),
      last_move_(setup.last_move),
      initial_(setup) {
  for (Point p : setup.black) board_.place(p, Color::BLACK);
  for (Point p : setup.white) board_.place(p, Color::WHITE);
  for (Point p : setup.offered) offered_[n_offered_++] = p;
  if (phase_ != OpeningPhase::NORMAL_RENJU) side_ = next_stone_color(phase_);
  if (phase_ == OpeningPhase::NORMAL_RENJU) after_normal_move();
  history_.reserve(256);
}

Role GameState::actor_role() const {
  if (phase_ == OpeningPhase::NORMAL_RENJU) {
    return side_ == Color::BLACK ? Role::BLACK_HOLDER : Role::WHITE_HOLDER;
  }
  return phase_role(phase_, profile_->taraguchi);
}

Color GameState::side_to_move() const {
  return phase_ == OpeningPhase::NORMAL_RENJU ? side_ : next_stone_color(phase_);
}

std::optional<PlayerId> GameState::winner() const {
  if (outcome_.result == GameResult::BLACK_WIN) return assignment_.black_holder;
  if (outcome_.result == GameResult::WHITE_WIN) return assignment_.white_holder;
  return std::nullopt;
}

bool GameState::is_legal(const Action& a) const { return is_legal_impl(a, /*for_engine=*/false); }

bool GameState::is_legal_impl(const Action& a, bool for_engine, MoveVerdict* verdict_out) const {
  if (is_terminal()) return false;
  if (a.has_point() && !is_valid_point(a.point)) return false;
  if (!a.has_point() && a.point != kNoPoint) return false;

  if (is_placement_phase(phase_)) {
    if (a.type == ActionType::PASS) {
      return phase_ == OpeningPhase::NORMAL_RENJU && profile_->pass_allowed;
    }
    if (a.type != ActionType::PLACE) return false;
    if (!board_.empty_at(a.point)) return false;
    const int region = placement_region(phase_, profile_->taraguchi);
    if (phase_ != OpeningPhase::NORMAL_RENJU && region != 0 && !in_central(a.point, region)) return false;
    if (side_to_move() == Color::BLACK) {
      const MoveVerdict v = black_verdict(a.point);
      if (verdict_out) *verdict_out = v;
      if (is_forbidden(v)) return !for_engine && profile_->illegal_move_behavior == IllegalMoveBehavior::LOSS;
    }
    return true;
  }
  if (is_swap_phase(phase_)) return a.type == ActionType::SWAP || a.type == ActionType::KEEP_COLOR;
  switch (phase_) {
    case OpeningPhase::CHOICE_AFTER_4:
      return a.type == ActionType::CHOOSE_DIRECT_FIFTH || a.type == ActionType::CHOOSE_TEN_PROPOSALS;
    case OpeningPhase::OFFER_TEN_5TH_MOVES:
      return a.type == ActionType::PROPOSE_FIFTH &&
             proposal_admissible(board_, offered(), a.point, profile_->taraguchi, profile_->black);
    case OpeningPhase::SELECT_ONE_OF_TEN: {
      if (a.type != ActionType::SELECT_PROPOSED_FIFTH) return false;
      const auto off = offered();
      return std::find(off.begin(), off.end(), a.point) != off.end();
    }
    default:
      return false;
  }
}

bool GameState::apply(const Action& a) {
  MoveVerdict verdict = MoveVerdict::LEGAL;
  if (!is_legal_impl(a, /*for_engine=*/false, &verdict)) return false;
  UndoRecord rec{a, phase_, side_, assignment_, last_move_, chosen_fifth_, kNoPoint, n_offered_, passes_, outcome_};

  switch (a.type) {
    case ActionType::PLACE: {
      const Color c = side_to_move();
      rec.placed = a.point;
      history_.push_back(rec);
      place_stone(a.point, c, c == Color::BLACK ? std::optional<MoveVerdict>(verdict) : std::nullopt);
      last_move_ = a.point;
      passes_ = 0;
      if (phase_ == OpeningPhase::NORMAL_RENJU) {
        side_ = opposite(c);
      } else {
        phase_ = phase_after_place(phase_);
        side_ = phase_ == OpeningPhase::NORMAL_RENJU ? Color::BLACK : next_stone_color(phase_);
      }
      after_normal_move();
      return true;
    }
    case ActionType::PASS:
      history_.push_back(rec);
      ++passes_;
      side_ = opposite(side_);
      last_move_ = kNoPoint;
      if (passes_ >= 2) outcome_ = {GameResult::DRAW, EndReason::DOUBLE_PASS, kNoPoint};
      return true;
    case ActionType::SWAP:
      history_.push_back(rec);
      assignment_ = assignment_.swapped();
      phase_ = phase_after_swap_decision(phase_);
      return true;
    case ActionType::KEEP_COLOR:
      history_.push_back(rec);
      phase_ = phase_after_swap_decision(phase_);
      return true;
    case ActionType::CHOOSE_DIRECT_FIFTH:
      history_.push_back(rec);
      phase_ = OpeningPhase::SWAP_BEFORE_DIRECT_5;
      return true;
    case ActionType::CHOOSE_TEN_PROPOSALS:
      history_.push_back(rec);
      phase_ = OpeningPhase::OFFER_TEN_5TH_MOVES;
      return true;
    case ActionType::PROPOSE_FIFTH:
      history_.push_back(rec);
      offered_[n_offered_++] = a.point;
      if (n_offered_ == profile_->taraguchi.n_proposals) phase_ = OpeningPhase::SELECT_ONE_OF_TEN;
      return true;
    case ActionType::SELECT_PROPOSED_FIFTH:
      rec.placed = a.point;
      history_.push_back(rec);
      place_stone(a.point, Color::BLACK);
      chosen_fifth_ = a.point;
      last_move_ = a.point;
      phase_ = OpeningPhase::WHITE_MOVE_6;
      side_ = Color::WHITE;
      return true;
  }
  return false;
}

void GameState::undo() {
  assert(!history_.empty());
  const UndoRecord rec = history_.back();
  history_.pop_back();
  if (rec.placed != kNoPoint) board_.remove(rec.placed);
  phase_ = rec.phase;
  side_ = rec.side;
  assignment_ = rec.assignment;
  last_move_ = rec.last_move;
  chosen_fifth_ = rec.chosen_fifth;
  n_offered_ = rec.n_offered;
  passes_ = rec.passes;
  outcome_ = rec.outcome;
}

void GameState::place_stone(Point p, Color c, std::optional<MoveVerdict> known_black_verdict) {
  if (c == Color::BLACK) {
    const MoveVerdict v = known_black_verdict ? *known_black_verdict : black_verdict(p);
    board_.place(p, c);
    if (is_forbidden(v)) outcome_ = {GameResult::WHITE_WIN, EndReason::FORBIDDEN_MOVE, p};
    else if (v == MoveVerdict::WIN_FIVE) outcome_ = {GameResult::BLACK_WIN, EndReason::FIVE, p};
  } else {
    const bool five = makes_five(board_, p, Color::WHITE);
    board_.place(p, c);
    if (five) outcome_ = {GameResult::WHITE_WIN, EndReason::FIVE, p};
  }
}

void GameState::after_normal_move() {
  if (is_terminal()) return;
  if (board_.empty_count() == 0) {
    outcome_ = {GameResult::DRAW, EndReason::BOARD_FULL, kNoPoint};
    return;
  }
  // ASSUMPTIONS A-R5: black to move, every empty point forbidden, no pass -> draw.
  if (phase_ == OpeningPhase::NORMAL_RENJU && side_ == Color::BLACK && !profile_->pass_allowed &&
      !black_has_legal_move()) {
    outcome_ = {GameResult::DRAW, EndReason::NO_LEGAL_MOVE, kNoPoint};
  }
}

bool GameState::black_has_legal_move() const {
  Board::Cells scratch = board_.cells();
  for (int i = 0; i < kNumPoints; ++i) {
    const Point p = static_cast<Point>(i);
    if (!board_.empty_at(p)) continue;
    if (!is_forbidden(black_verdict_cells(scratch, Board::index_of(p), profile_->black))) return true;
  }
  return false;
}

std::vector<Action> GameState::legal_actions() const {
  std::vector<Action> out;
  legal_actions(out);
  return out;
}

void GameState::legal_actions(std::vector<Action>& out) const {
  out.clear();
  if (is_terminal()) return;
  if (is_placement_phase(phase_)) {
    const int region = phase_ == OpeningPhase::NORMAL_RENJU ? 0 : placement_region(phase_, profile_->taraguchi);
    const bool black = side_to_move() == Color::BLACK;
    Board::Cells scratch = board_.cells();
    for (int i = 0; i < kNumPoints; ++i) {
      const Point p = static_cast<Point>(i);
      if (!board_.empty_at(p)) continue;
      if (region != 0 && !in_central(p, region)) continue;
      if (black && is_forbidden(black_verdict_cells(scratch, Board::index_of(p), profile_->black))) continue;
      out.push_back(Action::place(p));
    }
    if (phase_ == OpeningPhase::NORMAL_RENJU && profile_->pass_allowed) out.push_back(Action::pass());
    return;
  }
  if (is_swap_phase(phase_)) {
    out.push_back(Action::swap());
    out.push_back(Action::keep());
    return;
  }
  switch (phase_) {
    case OpeningPhase::CHOICE_AFTER_4:
      out.push_back(Action::direct_fifth());
      out.push_back(Action::ten_proposals());
      return;
    case OpeningPhase::OFFER_TEN_5TH_MOVES:
      for (Point p : admissible_proposals(board_, offered(), profile_->taraguchi, profile_->black)) {
        out.push_back(Action::propose(p));
      }
      return;
    case OpeningPhase::SELECT_ONE_OF_TEN:
      for (Point p : offered()) out.push_back(Action::select(p));
      return;
    default:
      return;
  }
}

std::vector<Action> GameState::action_history() const {
  std::vector<Action> out;
  out.reserve(history_.size());
  for (const auto& r : history_) out.push_back(r.action);
  return out;
}

uint64_t GameState::hash_under(int t) const {
  uint64_t h = board_.hash(t) ^ zobrist::phase(static_cast<int>(phase_));
  if (side_to_move() == Color::WHITE) h ^= zobrist::side_to_move_white();
  if (phase_ == OpeningPhase::OFFER_TEN_5TH_MOVES || phase_ == OpeningPhase::SELECT_ONE_OF_TEN) {
    for (Point q : offered()) h ^= zobrist::offered(transform_point(t, q));
  }
  return h;
}

uint64_t GameState::canonical_hash(int* transform) const {
  uint64_t best = hash_under(0);
  int best_t = 0;
  for (int t = 1; t < kNumTransforms; ++t) {
    const uint64_t h = hash_under(t);
    if (h < best) {
      best = h;
      best_t = t;
    }
  }
  if (transform) *transform = best_t;
  return best;
}

GameState GameState::transformed(int t) const {
  Setup s = initial_;
  for (Point& p : s.black) p = transform_point(t, p);
  for (Point& p : s.white) p = transform_point(t, p);
  for (Point& p : s.offered) p = transform_point(t, p);
  if (s.last_move != kNoPoint) s.last_move = transform_point(t, s.last_move);
  GameState g(profile_, s);
  for (const auto& rec : history_) {
    Action a = rec.action;
    if (a.has_point()) a.point = transform_point(t, a.point);
    [[maybe_unused]] const bool ok = g.apply(a);
    assert(ok && "D4 transform must preserve legality");
  }
  return g;
}

std::string GameState::to_string() const {
  std::ostringstream os;
  os << board_.to_string();
  os << "phase=" << phase_name(phase_) << " actor=" << player_name(actor()) << " ("
     << role_name(actor_role()) << ") next_stone=" << color_name(side_to_move())
     << " black_holder=" << player_name(assignment_.black_holder)
     << " white_holder=" << player_name(assignment_.white_holder);
  if (n_offered_ > 0) {
    os << " offered=";
    for (Point q : offered()) os << to_notation(q) << ' ';
  }
  os << " result=" << result_name(outcome_.result);
  if (is_terminal()) os << " (" << end_reason_name(outcome_.reason) << ")";
  os << '\n';
  return os.str();
}

}  // namespace dz
