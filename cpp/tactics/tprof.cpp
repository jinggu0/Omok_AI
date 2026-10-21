#include "tactics/tprof.h"

namespace dz::tactics::tprof {

Counters& counters() {
  thread_local Counters c;
  return c;
}

void reset() { counters() = Counters{}; }

}  // namespace dz::tactics::tprof
