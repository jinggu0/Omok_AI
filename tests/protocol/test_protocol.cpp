// Gomocup / Piskvork codec and engine-side session (ARCHITECTURE.md §3.12). Process I/O and the
// ExternalEngine client are exercised from Python against a mock engine (tests/python).
#include <gtest/gtest.h>

#include "common/test_util.h"
#include "protocol/piskvork_adapter.h"

namespace dz::protocol {
namespace {

using test::P;

TEST(Protocol, CoordinateConversionRoundTripsEveryPoint) {
  for (int p = 0; p < kNumPoints; ++p) {
    int x, y;
    point_to_gomocup(static_cast<Point>(p), x, y);
    const auto back = gomocup_to_point(x, y);
    ASSERT_TRUE(back);
    EXPECT_EQ(*back, p);
  }
  int x, y;
  point_to_gomocup(P("H8"), x, y);
  EXPECT_EQ(x, 7);
  EXPECT_EQ(y, 7);
  point_to_gomocup(P("A1"), x, y);  // bottom-left in project notation = top-left origin's last row
  EXPECT_EQ(x, 0);
  EXPECT_EQ(y, 14);
  point_to_gomocup(P("O15"), x, y);
  EXPECT_EQ(x, 14);
  EXPECT_EQ(y, 0);
  EXPECT_FALSE(gomocup_to_point(15, 0));
  EXPECT_FALSE(gomocup_to_point(0, -1));
  EXPECT_EQ(format_move(P("A1")), "0,14");
  EXPECT_EQ(format_board_line(P("O15"), 2), "14,0,2");
}

TEST(Protocol, ParsesCommands) {
  EXPECT_EQ(parse_command("START 15").type, CommandType::START);
  EXPECT_EQ(parse_command("start 15").size, 15);
  EXPECT_EQ(parse_command("START x").type, CommandType::UNKNOWN);
  const Command t = parse_command("TURN 3,11");
  EXPECT_EQ(t.type, CommandType::TURN);
  EXPECT_EQ(t.x, 3);
  EXPECT_EQ(t.y, 11);
  EXPECT_EQ(parse_command("TURN 3").type, CommandType::UNKNOWN);
  EXPECT_EQ(parse_command("TURN 3,4,5").type, CommandType::UNKNOWN);
  const Command i = parse_command("INFO timeout_turn 1000");
  EXPECT_EQ(i.type, CommandType::INFO);
  EXPECT_EQ(i.key, "timeout_turn");
  EXPECT_EQ(i.value, "1000");
  EXPECT_EQ(parse_command("INFO Rule 4").key, "rule");
  EXPECT_EQ(parse_command("RECTSTART 20,15").type, CommandType::RECTSTART);
  EXPECT_EQ(parse_command("BOARD").type, CommandType::BOARD);
  EXPECT_EQ(parse_command("END").type, CommandType::END);
  EXPECT_EQ(parse_command("ABOUT").type, CommandType::ABOUT);
  EXPECT_EQ(parse_command("TAKEBACK 1,2").type, CommandType::TAKEBACK);
  EXPECT_EQ(parse_command("HELLO").type, CommandType::UNKNOWN);
}

TEST(Protocol, ParsesReplies) {
  Reply r = parse_reply("7,7");
  EXPECT_EQ(r.kind, ReplyKind::MOVE);
  EXPECT_EQ(r.x, 7);
  EXPECT_EQ(r.y, 7);
  EXPECT_EQ(parse_reply(" 7 , 8 \r").kind, ReplyKind::MOVE);
  EXPECT_EQ(parse_reply("OK").kind, ReplyKind::OK);
  EXPECT_EQ(parse_reply("ERROR bad").kind, ReplyKind::ERROR);
  EXPECT_EQ(parse_reply("ERROR bad").text, "bad");
  EXPECT_EQ(parse_reply("MESSAGE depth 5").kind, ReplyKind::MESSAGE);
  EXPECT_EQ(parse_reply("DEBUG x").kind, ReplyKind::DEBUG);
  EXPECT_EQ(parse_reply("UNKNOWN FOO").kind, ReplyKind::UNKNOWN);
  EXPECT_EQ(parse_reply("SUGGEST 1,2").kind, ReplyKind::SUGGEST);
  EXPECT_EQ(parse_reply("7 7").kind, ReplyKind::OTHER);  // a move needs a comma
  EXPECT_EQ(parse_reply("7,7,1").kind, ReplyKind::OTHER);
  EXPECT_EQ(parse_reply("a,b").kind, ReplyKind::OTHER);
  EXPECT_EQ(parse_reply("99999999999999,1").kind, ReplyKind::OTHER);
  EXPECT_EQ(parse_reply("").kind, ReplyKind::OTHER);
}

// Plays the first legal point from a fixed corner list, so tests know the reply.
std::optional<Point> corner_bot(const GameState& g, const ThinkLimits&) {
  for (const char* s : {"O15", "O13", "O11", "O9", "M15", "M13", "K15"}) {
    if (g.is_legal(Action::place(P(s)))) return P(s);
  }
  return std::nullopt;
}

TEST(Protocol, EngineSessionPlaysAGame) {
  EngineSession s(corner_bot, "name=\"Test\"");
  EXPECT_TRUE(s.feed("INFO timeout_turn 500").empty());
  EXPECT_TRUE(s.feed("INFO rule 4").empty());
  EXPECT_EQ(s.feed("START 15"), std::vector<std::string>{"OK"});
  EXPECT_EQ(s.limits().timeout_turn_ms, 500);
  EXPECT_EQ(s.feed("BEGIN"), std::vector<std::string>{"14,0"});  // O15
  EXPECT_EQ(s.feed("TURN 7,7"), std::vector<std::string>{"14,2"});  // opponent H8, engine O13
  ASSERT_NE(s.state(), nullptr);
  EXPECT_EQ(s.state()->board().stone_count(), 3);
  EXPECT_EQ(s.feed("TAKEBACK 14,2"), std::vector<std::string>{"OK"});
  EXPECT_EQ(s.state()->board().stone_count(), 2);
  EXPECT_EQ(s.feed("TAKEBACK 0,0")[0].rfind("ERROR", 0), 0u);
  EXPECT_EQ(s.feed("ABOUT"), std::vector<std::string>{"name=\"Test\""});
  EXPECT_EQ(s.feed("TURN 7,7")[0].rfind("ERROR", 0), 0u) << "occupied point";
  EXPECT_EQ(s.feed("FOO")[0], "UNKNOWN FOO");
  EXPECT_TRUE(s.feed("END").empty());
  EXPECT_TRUE(s.quit());
}

TEST(Protocol, EngineSessionBoardSetsColoursFromStoneCounts) {
  EngineSession s(corner_bot, "x");
  s.feed("START 15");
  // Black F8 G8 H6 H7, white A1 A3 A5: white (the engine, field 1) to move.
  EXPECT_TRUE(s.feed("BOARD").empty());
  for (const char* line : {"5,7,2", "0,14,1", "6,7,2", "0,12,1", "7,9,2", "0,10,1", "7,8,2"}) EXPECT_TRUE(s.feed(line).empty());
  EXPECT_EQ(s.feed("DONE"), std::vector<std::string>{"14,0"});
  ASSERT_NE(s.state(), nullptr);
  EXPECT_EQ(s.state()->side_to_move(), Color::BLACK);
  EXPECT_EQ(s.state()->board().stone_count(Color::WHITE), 4);
  // H8 would be a black double three: the canonical rules reject it, so the session reports an
  // illegal opponent move instead of continuing.
  EXPECT_EQ(s.feed("TURN 7,7")[0].rfind("ERROR", 0), 0u);
}

TEST(Protocol, EngineSessionRejectsUnsupportedSetups) {
  EngineSession s(corner_bot, "x");
  EXPECT_EQ(s.feed("START 19")[0].rfind("ERROR", 0), 0u);
  EXPECT_EQ(s.feed("RECTSTART 15,15")[0].rfind("ERROR", 0), 0u);
  s.feed("START 15");
  s.feed("INFO rule 1");  // exact five gomoku: not supported
  EXPECT_EQ(s.feed("BEGIN")[0].rfind("ERROR", 0), 0u);
  s.feed("INFO rule 4");
  s.feed("BOARD");
  EXPECT_EQ(s.feed("1,2")[0].rfind("ERROR", 0), 0u) << "malformed BOARD line";
  s.feed("BOARD");
  s.feed("7,7,1");
  s.feed("7,8,1");
  EXPECT_EQ(s.feed("DONE")[0].rfind("ERROR", 0), 0u) << "two own stones, no opponent stone";
  s.feed("BOARD");
  s.feed("7,7,3");
  EXPECT_EQ(s.feed("DONE")[0].rfind("ERROR", 0), 0u) << "continuous-game field";
}

TEST(Protocol, EngineSessionNeverPlaysAnIllegalMove) {
  // A buggy think function: the session must answer ERROR, not send the move.
  EngineSession s([](const GameState&, const ThinkLimits&) -> std::optional<Point> { return P("H8"); }, "x");
  s.feed("START 15");
  EXPECT_EQ(s.feed("BEGIN"), std::vector<std::string>{"7,7"});
  EXPECT_EQ(s.feed("TURN 0,0")[0].rfind("ERROR", 0), 0u) << "H8 is occupied";
}

}  // namespace
}  // namespace dz::protocol
