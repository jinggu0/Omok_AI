"""DojoZero Python package: training, evaluation and tooling.

Game rules, features and tactics live in C++ (``dojozero._core``); Python never re-implements them
(ARCHITECTURE.md §1.1).
"""

from . import _core  # noqa: F401  (built by CMake, copied next to this file)

__version__ = "0.4.0"
