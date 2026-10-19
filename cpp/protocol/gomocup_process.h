// Child process with line-based stdin/stdout (ARCHITECTURE.md §3.12). No game knowledge.
// stdout and stderr are read by background threads into queues so reads can time out without
// blocking on a misbehaving engine. Output is untrusted: lines longer than kMaxLine are cut and
// flagged.
#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace dz::protocol {

class GomocupProcess {
 public:
  static constexpr size_t kMaxLine = 8192;

  struct Options {
    std::string path;
    std::vector<std::string> args;
    std::string working_dir;  // empty = inherit
    bool crlf = true;         // line terminator written to the engine
  };

  enum class ReadStatus { LINE, TIMEOUT, CLOSED, TOO_LONG };

  GomocupProcess() = default;
  GomocupProcess(const GomocupProcess&) = delete;
  GomocupProcess& operator=(const GomocupProcess&) = delete;
  ~GomocupProcess();

  bool start(const Options& opt, std::string* error);
  bool write_line(const std::string& line);
  // Next stdout line within timeout_ms (< 0: wait forever).
  ReadStatus read_line(int timeout_ms, std::string& line);
  bool alive();
  void kill();
  // Exit code once the process ended, else nullopt.
  std::optional<int> exit_code();
  std::string stderr_tail(size_t max_lines = 20);
  long long pid() const { return pid_; }

 private:
  struct Queue {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool closed = false;
    bool overflow = false;
  };
  void reader_loop(void* handle_or_fd, Queue* q);
  void join_readers();

  Options opt_;
  Queue out_, err_;
  std::thread out_thread_, err_thread_;
  long long pid_ = -1;
  bool running_ = false;
#ifdef _WIN32
  void* process_ = nullptr;
  void* job_ = nullptr;
  void* stdin_w_ = nullptr;
  void* stdout_r_ = nullptr;
  void* stderr_r_ = nullptr;
#else
  int stdin_w_ = -1;
  int stdout_r_ = -1;
  int stderr_r_ = -1;
  std::optional<int> status_;
#endif
};

}  // namespace dz::protocol
