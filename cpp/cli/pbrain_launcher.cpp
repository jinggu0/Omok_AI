// pbrain-DojoZero.exe: Gomocup/Piskvork engine executable for GUIs and tournament managers
// (PRD FR-21b, ADR-08 interim: a launcher around the Python host).
//
// It starts
//   <python> -m dojozero.protocol.pbrain --think engine:<model> --device <device> [<args>]
// with PYTHONPATH=<repo>/python and the working directory <repo>, hands the child its own
// stdin/stdout (handle inheritance: no copying, no re-encoding) and returns the child's exit code.
// The protocol codec stays the C++ EngineSession inside the Python host; nothing is parsed here.
//
// Configuration: pbrain-DojoZero.ini next to the exe (key=value, '#' comments), every key optional:
//   repo=C:/path/to/Omok_AI          default: the source tree this exe was built from
//   python=.venv/Scripts/python.exe  relative paths are relative to repo
//   model=runs/taraguchi_v1/models/gen_001.pt
//   device=xpu
//   rule_dialect=gomocup             "yixin" for Yixin Board (its INFO rule 2 = Renju)
//   opening_guard=1                  GUI usability guard for the first moves (0 = off)
//   max_move_ms=0                    cap on the per-move search time (0 = the GUI's limit only)
//   args=                            extra arguments for dojozero.protocol.pbrain
//   stderr=pbrain-DojoZero.log       child stderr: a log file (appended; relative to the exe
//                                    folder) or "inherit". A GUI that never reads stderr would
//                                    otherwise block the engine once the pipe fills.
#ifndef _WIN32
#error "pbrain-DojoZero launcher is Windows-only"
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <io.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#ifndef DOJOZERO_REPO_DIR
#define DOJOZERO_REPO_DIR ""
#endif

namespace fs = std::filesystem;

namespace {

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}

std::string narrow(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

std::string trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

// CommandLineToArgvW-compatible quoting (same rules as protocol/gomocup_process.cpp).
std::wstring quote_arg(const std::wstring& a) {
  if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
  std::wstring out = L"\"";
  size_t backslashes = 0;
  for (wchar_t c : a) {
    if (c == L'\\') {
      ++backslashes;
    } else if (c == L'"') {
      out.append(backslashes * 2 + 1, L'\\');
      out += L'"';
      backslashes = 0;
    } else {
      out.append(backslashes, L'\\');
      out += c;
      backslashes = 0;
    }
  }
  out.append(backslashes * 2, L'\\');
  out += L'"';
  return out;
}

fs::path exe_dir() {
  std::wstring buf(MAX_PATH, L'\0');
  for (;;) {
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n < buf.size()) {
      buf.resize(n);
      break;
    }
    buf.resize(buf.size() * 2);
  }
  return fs::path(buf).parent_path();
}

std::map<std::string, std::string> read_config(const fs::path& file) {
  std::map<std::string, std::string> cfg;
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.size() >= 3 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line = trim(line.substr(3));  // UTF-8 BOM
    if (line.empty() || line[0] == '#' || line[0] == ';') continue;
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    cfg[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
  }
  return cfg;
}

std::string get(const std::map<std::string, std::string>& cfg, const char* key, const char* def) {
  const auto it = cfg.find(key);
  return it == cfg.end() || it->second.empty() ? def : it->second;
}

fs::path resolve(const fs::path& base, const std::string& p) {
  fs::path q(widen(p));
  return q.is_absolute() ? q : base / q;
}

// Reports a start-up failure where a GUI shows it (an ERROR line on stdout) and on stderr.
int fail(const std::string& msg) {
  std::fprintf(stdout, "ERROR pbrain-DojoZero: %s\n", msg.c_str());
  std::fflush(stdout);
  std::fprintf(stderr, "pbrain-DojoZero: %s\n", msg.c_str());
  return 2;
}

// The OS handle behind CRT descriptor fd (0, 1, 2). Taken from the CRT, not GetStdHandle: a parent
// that spawns through the CRT (GLib's gspawn helper in Yixin Board) passes the pipes in the CRT
// descriptor table (STARTUPINFO lpReserved2) and leaves the process standard handles unset.
HANDLE std_handle(int fd) {
  const intptr_t h = _get_osfhandle(fd);
  if (h != -1 && h != -2) return reinterpret_cast<HANDLE>(h);
  return GetStdHandle(fd == 0 ? STD_INPUT_HANDLE : fd == 1 ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);
}

// An inheritable copy of one of our standard handles for the child (closed by the caller).
HANDLE inheritable(int fd) {
  const HANDLE h = std_handle(fd);
  if (h == nullptr || h == INVALID_HANDLE_VALUE) return h;
  HANDLE dup = nullptr;
  if (DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &dup, 0, TRUE, DUPLICATE_SAME_ACCESS)) return dup;
  return nullptr;
}

bool is_console(HANDLE h) { return h != nullptr && h != INVALID_HANDLE_VALUE && GetFileType(h) == FILE_TYPE_CHAR; }

}  // namespace

int wmain() {
  const fs::path dir = exe_dir();
  const auto cfg = read_config(dir / L"pbrain-DojoZero.ini");

  const fs::path repo = fs::path(widen(get(cfg, "repo", DOJOZERO_REPO_DIR)));
  if (repo.empty() || !fs::is_directory(repo / L"python" / L"dojozero"))
    return fail("repository not found (" + narrow(repo.wstring()) + "); set repo= in pbrain-DojoZero.ini next to the exe");
  const fs::path python = resolve(repo, get(cfg, "python", ".venv/Scripts/python.exe"));
  if (!fs::is_regular_file(python)) return fail("python not found: " + narrow(python.wstring()));
  const fs::path model = resolve(repo, get(cfg, "model", "runs/taraguchi_v1/models/gen_001.pt"));
  if (!fs::is_regular_file(model)) return fail("model not found: " + narrow(model.wstring()));
  const std::string device = get(cfg, "device", "xpu");
  const std::string extra = get(cfg, "args", "");
  const std::string err_target = get(cfg, "stderr", "pbrain-DojoZero.log");

  std::wstring cmd = quote_arg(python.wstring()) + L" -m dojozero.protocol.pbrain --think " +
                     quote_arg(L"engine:" + model.wstring()) + L" --device " + quote_arg(widen(device));
  // GUI-only host options (python/dojozero/protocol/pbrain.py); evaluation paths never set them.
  const std::string dialect = get(cfg, "rule_dialect", "gomocup");
  if (dialect != "gomocup") cmd += L" --rule-dialect " + quote_arg(widen(dialect));
  if (get(cfg, "opening_guard", "1") != "0") cmd += L" --opening-guard";
  if (const std::string cap = get(cfg, "max_move_ms", "0"); cap != "0") cmd += L" --max-move-ms " + quote_arg(widen(cap));
  if (!extra.empty()) cmd += L" " + widen(extra);

  // PYTHONPATH=<repo>/python[;previous] for the child (inherited environment).
  const std::wstring pkg = (repo / L"python").wstring();
  std::wstring pythonpath = pkg;
  if (const DWORD n = GetEnvironmentVariableW(L"PYTHONPATH", nullptr, 0); n > 0) {
    std::wstring old(n, L'\0');
    old.resize(GetEnvironmentVariableW(L"PYTHONPATH", old.data(), n));
    if (!old.empty()) pythonpath += L";" + old;
  }
  SetEnvironmentVariableW(L"PYTHONPATH", pythonpath.c_str());

  const HANDLE in = inheritable(0);
  const HANDLE out = inheritable(1);
  HANDLE err = nullptr;
  if (err_target == "inherit") {
    err = inheritable(2);
  } else {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    const fs::path log = resolve(dir, err_target);
    err = CreateFileW(log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_ALWAYS,
                      FILE_ATTRIBUTE_NORMAL, nullptr);
    if (err == INVALID_HANDLE_VALUE) err = inheritable(2);
    else {
      const std::string head = "[pbrain-DojoZero] " + narrow(cmd) + "\r\n";
      DWORD written = 0;
      WriteFile(err, head.data(), static_cast<DWORD>(head.size()), &written, nullptr);
    }
  }

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = in;
  si.hStdOutput = out;
  si.hStdError = err;
  // Under a GUI the launcher has no visible console; keep the Python child windowless too. In a
  // terminal the child shares the console so typed commands reach it.
  const DWORD window = is_console(std_handle(0)) || is_console(std_handle(1))
                           ? 0 : CREATE_NO_WINDOW;
  // Kill-on-close job: if the GUI terminates the launcher, the Python process (and the interpreter
  // the venv redirector starts) goes with it instead of holding the GPU.
  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  if (job) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
  }
  PROCESS_INFORMATION pi{};
  const std::wstring wdir = repo.wstring();
  const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED | window, nullptr,
                                 wdir.c_str(), &si, &pi);
  const DWORD start_error = GetLastError();
  for (HANDLE h : {in, out, err})  // our inheritable copies / the log file
    if (h != nullptr && h != INVALID_HANDLE_VALUE) CloseHandle(h);
  if (!ok) return fail("CreateProcess failed (error " + std::to_string(start_error) + "): " + narrow(cmd));
  if (job) AssignProcessToJobObject(job, pi.hProcess);
  {
    // Same EcoQoS opt-out as protocol/gomocup_process.cpp: a windowless engine may otherwise be put
    // on slow cores (Phase 8 E3b). The Python host also opts its own interpreter out, because the
    // venv python.exe is a redirector that starts the real interpreter as a further child.
    PROCESS_POWER_THROTTLING_STATE pt{};
    pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    pt.StateMask = 0;
    SetProcessInformation(pi.hProcess, ProcessPowerThrottling, &pt, sizeof(pt));
  }
  SetConsoleCtrlHandler(nullptr, TRUE);  // Ctrl+C goes to the child; we wait for its exit code
  ResumeThread(pi.hThread);
  CloseHandle(pi.hThread);
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);
  if (job) CloseHandle(job);
  return static_cast<int>(code);
}
