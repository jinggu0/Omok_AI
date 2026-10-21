#include "protocol/gomocup_process.h"

#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace dz::protocol {

namespace {
constexpr const char* kTooLongMarker = "\x01TOO_LONG";
}

GomocupProcess::~GomocupProcess() {
  kill();
#ifdef _WIN32
  if (process_) CloseHandle(static_cast<HANDLE>(process_));
#endif
}

void GomocupProcess::reader_loop(void* src, Queue* q) {
  std::string cur;
  bool dropping = false;  // inside an over-long line: discard until the next newline
  char buf[4096];
  for (;;) {
#ifdef _WIN32
    DWORD n = 0;
    if (!ReadFile(static_cast<HANDLE>(src), buf, sizeof(buf), &n, nullptr) || n == 0) break;
#else
    const ssize_t n = ::read(static_cast<int>(reinterpret_cast<intptr_t>(src)), buf, sizeof(buf));
    if (n <= 0) break;
#endif
    std::lock_guard<std::mutex> lk(q->m);
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
      const char c = buf[i];
      if (c == '\n') {
        if (!cur.empty() && cur.back() == '\r') cur.pop_back();
        if (!dropping) q->lines.push_back(cur);
        dropping = false;
        cur.clear();
      } else if (!dropping) {
        cur += c;
        if (cur.size() > kMaxLine) {
          q->lines.push_back(kTooLongMarker);
          q->overflow = true;
          dropping = true;
          cur.clear();
        }
      }
    }
    q->cv.notify_all();
  }
  std::lock_guard<std::mutex> lk(q->m);
  if (!cur.empty() && !dropping) q->lines.push_back(cur);
  q->closed = true;
  q->cv.notify_all();
}

#ifdef _WIN32

namespace {
// CommandLineToArgvW-compatible quoting.
std::string quote_arg(const std::string& a) {
  if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
  std::string out = "\"";
  size_t backslashes = 0;
  for (char c : a) {
    if (c == '\\') {
      ++backslashes;
    } else if (c == '"') {
      out.append(backslashes * 2 + 1, '\\');
      out += '"';
      backslashes = 0;
    } else {
      out.append(backslashes, '\\');
      out += c;
      backslashes = 0;
    }
  }
  out.append(backslashes * 2, '\\');
  out += '"';
  return out;
}

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}
}  // namespace

bool GomocupProcess::start(const Options& opt, std::string* error) {
  opt_ = opt;
  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr, err_r = nullptr, err_w = nullptr;
  if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0) || !CreatePipe(&err_r, &err_w, &sa, 0)) {
    if (error) *error = "CreatePipe failed";
    return false;
  }
  SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);

  std::string cmd = quote_arg(opt.path);
  for (const auto& a : opt.args) cmd += " " + quote_arg(a);
  std::wstring wcmd = widen(cmd);
  std::wstring wdir = widen(opt.working_dir);

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = in_r;
  si.hStdOutput = out_w;
  si.hStdError = err_w;
  PROCESS_INFORMATION pi{};
  // A job object kills helper processes the engine may spawn when we kill the engine.
  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  if (job) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
  }
  const BOOL ok = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED,
                                 nullptr, wdir.empty() ? nullptr : wdir.c_str(), &si, &pi);
  CloseHandle(in_r);
  CloseHandle(out_w);
  CloseHandle(err_w);
  if (!ok) {
    if (error) *error = "CreateProcess failed for " + opt.path + " (error " + std::to_string(GetLastError()) + ")";
    CloseHandle(in_w);
    CloseHandle(out_r);
    CloseHandle(err_r);
    if (job) CloseHandle(job);
    return false;
  }
  if (job) AssignProcessToJobObject(job, pi.hProcess);
  {
    // Engines run windowless in the background, where Windows power throttling (EcoQoS) may put
    // them on slow cores; opt them out so every timed engine gets full speed (Phase 8 E3b).
    PROCESS_POWER_THROTTLING_STATE pt{};
    pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    pt.StateMask = 0;
    SetProcessInformation(pi.hProcess, ProcessPowerThrottling, &pt, sizeof(pt));
  }
  ResumeThread(pi.hThread);
  CloseHandle(pi.hThread);
  process_ = pi.hProcess;
  job_ = job;
  pid_ = pi.dwProcessId;
  stdin_w_ = in_w;
  stdout_r_ = out_r;
  stderr_r_ = err_r;
  running_ = true;
  out_thread_ = std::thread(&GomocupProcess::reader_loop, this, stdout_r_, &out_);
  err_thread_ = std::thread(&GomocupProcess::reader_loop, this, stderr_r_, &err_);
  return true;
}

bool GomocupProcess::write_line(const std::string& line) {
  if (!running_ || !stdin_w_) return false;
  const std::string s = line + (opt_.crlf ? "\r\n" : "\n");
  DWORD written = 0;
  return WriteFile(static_cast<HANDLE>(stdin_w_), s.data(), static_cast<DWORD>(s.size()), &written, nullptr) &&
         written == s.size();
}

bool GomocupProcess::alive() {
  if (!process_) return false;
  return WaitForSingleObject(static_cast<HANDLE>(process_), 0) == WAIT_TIMEOUT;
}

std::optional<int> GomocupProcess::exit_code() {
  if (!process_ || alive()) return std::nullopt;
  DWORD code = 0;
  GetExitCodeProcess(static_cast<HANDLE>(process_), &code);
  return static_cast<int>(code);
}

void GomocupProcess::kill() {
  if (process_) {
    if (alive()) TerminateProcess(static_cast<HANDLE>(process_), 1);
    WaitForSingleObject(static_cast<HANDLE>(process_), 5000);
  }
  if (job_) {
    CloseHandle(static_cast<HANDLE>(job_));  // kills any remaining process in the job
    job_ = nullptr;
  }
  if (stdin_w_) {
    CloseHandle(static_cast<HANDLE>(stdin_w_));
    stdin_w_ = nullptr;
  }
  join_readers();
  if (stdout_r_) { CloseHandle(static_cast<HANDLE>(stdout_r_)); stdout_r_ = nullptr; }
  if (stderr_r_) { CloseHandle(static_cast<HANDLE>(stderr_r_)); stderr_r_ = nullptr; }
  running_ = false;
}

#else  // POSIX (not exercised on the Windows development machine)

bool GomocupProcess::start(const Options& opt, std::string* error) {
  opt_ = opt;
  int in[2], out[2], err[2];
  if (pipe(in) || pipe(out) || pipe(err)) {
    if (error) *error = "pipe failed";
    return false;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    if (error) *error = "fork failed";
    return false;
  }
  if (pid == 0) {
    dup2(in[0], 0);
    dup2(out[1], 1);
    dup2(err[1], 2);
    close(in[1]);
    close(out[0]);
    close(err[0]);
    if (!opt.working_dir.empty() && chdir(opt.working_dir.c_str()) != 0) _exit(127);
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(opt.path.c_str()));
    for (const auto& a : opt.args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    execv(opt.path.c_str(), argv.data());
    _exit(127);
  }
  close(in[0]);
  close(out[1]);
  close(err[1]);
  signal(SIGPIPE, SIG_IGN);
  pid_ = pid;
  stdin_w_ = in[1];
  stdout_r_ = out[0];
  stderr_r_ = err[0];
  running_ = true;
  out_thread_ = std::thread(&GomocupProcess::reader_loop, this, reinterpret_cast<void*>(static_cast<intptr_t>(stdout_r_)), &out_);
  err_thread_ = std::thread(&GomocupProcess::reader_loop, this, reinterpret_cast<void*>(static_cast<intptr_t>(stderr_r_)), &err_);
  return true;
}

bool GomocupProcess::write_line(const std::string& line) {
  if (!running_ || stdin_w_ < 0) return false;
  const std::string s = line + (opt_.crlf ? "\r\n" : "\n");
  return ::write(stdin_w_, s.data(), s.size()) == static_cast<ssize_t>(s.size());
}

bool GomocupProcess::alive() {
  if (pid_ <= 0 || status_) return false;
  int st = 0;
  const pid_t r = waitpid(static_cast<pid_t>(pid_), &st, WNOHANG);
  if (r == 0) return true;
  status_ = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
  return false;
}

std::optional<int> GomocupProcess::exit_code() {
  if (alive()) return std::nullopt;
  return status_;
}

void GomocupProcess::kill() {
  if (pid_ > 0 && alive()) {
    ::kill(static_cast<pid_t>(pid_), SIGKILL);
    int st = 0;
    waitpid(static_cast<pid_t>(pid_), &st, 0);
    status_ = 128 + SIGKILL;
  }
  if (stdin_w_ >= 0) { close(stdin_w_); stdin_w_ = -1; }
  join_readers();
  if (stdout_r_ >= 0) { close(stdout_r_); stdout_r_ = -1; }
  if (stderr_r_ >= 0) { close(stderr_r_); stderr_r_ = -1; }
  running_ = false;
}

#endif

void GomocupProcess::join_readers() {
  if (out_thread_.joinable()) out_thread_.join();
  if (err_thread_.joinable()) err_thread_.join();
}

GomocupProcess::ReadStatus GomocupProcess::read_line(int timeout_ms, std::string& line) {
  std::unique_lock<std::mutex> lk(out_.m);
  auto ready = [&] { return !out_.lines.empty() || out_.closed; };
  if (timeout_ms < 0) {
    out_.cv.wait(lk, ready);
  } else if (!out_.cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready)) {
    return ReadStatus::TIMEOUT;
  }
  if (out_.lines.empty()) return ReadStatus::CLOSED;
  line = std::move(out_.lines.front());
  out_.lines.pop_front();
  if (line == kTooLongMarker) return ReadStatus::TOO_LONG;
  return ReadStatus::LINE;
}

std::string GomocupProcess::stderr_tail(size_t max_lines) {
  std::lock_guard<std::mutex> lk(err_.m);
  std::string s;
  const size_t n = err_.lines.size();
  for (size_t i = n > max_lines ? n - max_lines : 0; i < n; ++i) s += err_.lines[i] + "\n";
  return s;
}

}  // namespace dz::protocol
