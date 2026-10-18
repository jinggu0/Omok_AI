// dojozero._core: Python bindings (ARCHITECTURE.md §4). Python never re-implements rules; it calls
// this module for game state, features, auxiliary targets, symmetry tables and tactics.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>

#include "board/transform.h"
#include "features/features.h"
#include "game/position_spec.h"
#include "search/bots.h"
#include "search/search.h"
#include "tactics/engine.h"

namespace py = pybind11;
using namespace dz;

namespace {

ProfilePtr profile_named(const std::string& name) {
  if (name == "RENJU_RIF") return make_profile(RuleProfile::renju_rif());
  if (name == "TARAGUCHI10") return make_profile(RuleProfile::taraguchi10());
  if (name == "OMOKDOJANG_DAN") return make_profile(RuleProfile::omokdojang_dan());
  throw std::invalid_argument("unknown profile " + name);
}

GameState from_spec(const std::string& json_text, const std::string& profile) {
  std::string error;
  auto spec = parse_position_spec(nlohmann::json::parse(json_text), &error);
  if (!spec) throw std::invalid_argument(error);
  auto g = load_position(*spec, profile_named(profile), &error);
  if (!g) throw std::invalid_argument(error);
  return *g;
}

// NORMAL_RENJU state from a cell array (0 empty, 1 black, 2 white) and the color to move.
GameState from_cells(const int8_t* cells, int side, const ProfilePtr& profile) {
  PositionSpec spec;
  for (int p = 0; p < kNumPoints; ++p) {
    if (cells[p] == 1) spec.black.push_back(static_cast<Point>(p));
    else if (cells[p] == 2) spec.white.push_back(static_cast<Point>(p));
  }
  spec.side_to_move = side == 0 ? Color::BLACK : Color::WHITE;
  std::string error;
  auto g = load_position(spec, profile, &error);
  if (!g) throw std::invalid_argument("invalid board: " + error);
  return *g;
}

template <typename F>
void parallel_for(size_t n, int threads, F&& f) {
  if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  threads = static_cast<int>(std::min<size_t>(threads, std::max<size_t>(1, n)));
  std::vector<std::thread> pool;
  std::exception_ptr err;
  std::mutex mu;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t]() {
      try {
        for (size_t i = t; i < n; i += threads) f(i);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mu);
        if (!err) err = std::current_exception();
      }
    });
  }
  for (auto& th : pool) th.join();
  if (err) std::rethrow_exception(err);
}

py::array_t<int8_t> board_array(const GameState& g) {
  py::array_t<int8_t> out(kNumPoints);
  auto* d = out.mutable_data();
  for (int p = 0; p < kNumPoints; ++p) {
    const Cell c = g.board().at(static_cast<Point>(p));
    d[p] = c == BLACK_STONE ? 1 : c == WHITE_STONE ? 2 : 0;
  }
  return out;
}

// Many search trees driven together: Python collects leaf features from all trees, evaluates them
// in one network batch, and applies the results (ARCHITECTURE.md §3.10, game-parallel batching).
class SearchBatch {
 public:
  explicit SearchBatch(const features::FeatureConfig& fcfg) : fcfg_(fcfg) {}

  int add(const GameState& g, const search::SearchConfig& cfg) {
    trees_.push_back(std::make_unique<search::Tree>(g, cfg, fcfg_));
    return static_cast<int>(trees_.size() - 1);
  }

  // Tree construction runs root tactics (can take ~0.1-1 s each): build many trees in parallel.
  std::vector<int> add_many(const std::vector<GameState>& states, const search::SearchConfig& cfg, int threads) {
    const size_t base = trees_.size();
    trees_.resize(base + states.size());
    {
      py::gil_scoped_release release;
      parallel_for(states.size(), threads, [&](size_t i) {
        search::SearchConfig c = cfg;
        c.seed = cfg.seed * 1000003ULL + base + i;
        trees_[base + i] = std::make_unique<search::Tree>(states[i], c, fcfg_);
      });
    }
    std::vector<int> ids(states.size());
    for (size_t i = 0; i < states.size(); ++i) ids[i] = static_cast<int>(base + i);
    return ids;
  }

  bool done() const {
    for (const auto& t : trees_) {
      if (!t->done() || t->inflight() > 0) return false;
    }
    return true;
  }

  // Up to max_leaves leaves (round-robin over trees). Returns uint8 features [N, C, 15, 15].
  py::array_t<uint8_t> collect(int max_leaves) {
    if (!pending_.empty()) throw std::runtime_error("apply() the previous batch first");
    const size_t plane_bytes = static_cast<size_t>(fcfg_.num_planes()) * kNumPoints;
    buffer_.resize(static_cast<size_t>(max_leaves) * plane_bytes);
    bool progress = true;
    while (static_cast<int>(pending_.size()) < max_leaves && progress) {
      progress = false;
      for (size_t i = 0; i < trees_.size() && static_cast<int>(pending_.size()) < max_leaves; ++i) {
        const int ticket = trees_[i]->select(buffer_.data() + pending_.size() * plane_bytes);
        if (ticket >= 0) {
          pending_.push_back({static_cast<int>(i), ticket});
          progress = true;
        } else if (ticket == -1) {
          progress = true;  // resolved without the network: keep going
        }
      }
    }
    const py::ssize_t n = static_cast<py::ssize_t>(pending_.size());
    py::array_t<uint8_t> out({n, static_cast<py::ssize_t>(fcfg_.num_planes()), static_cast<py::ssize_t>(kBoardSize),
                              static_cast<py::ssize_t>(kBoardSize)});
    std::memcpy(out.mutable_data(), buffer_.data(), static_cast<size_t>(n) * plane_bytes);
    return out;
  }

  void apply(py::array_t<float, py::array::c_style | py::array::forcecast> policy,
             py::array_t<float, py::array::c_style | py::array::forcecast> wdl,
             py::array_t<float, py::array::c_style | py::array::forcecast> swap,
             py::array_t<float, py::array::c_style | py::array::forcecast> branch,
             py::array_t<float, py::array::c_style | py::array::forcecast> propose,
             py::array_t<float, py::array::c_style | py::array::forcecast> select) {
    const size_t n = pending_.size();
    if (static_cast<size_t>(wdl.size()) != n * 3 || static_cast<size_t>(policy.size()) != n * features::kNumPolicyActions ||
        static_cast<size_t>(swap.size()) != n * 2 || static_cast<size_t>(branch.size()) != n * 2 ||
        static_cast<size_t>(propose.size()) != n * kNumPoints || static_cast<size_t>(select.size()) != n * kNumPoints) {
      throw std::invalid_argument("apply: shapes do not match the collected batch");
    }
    for (size_t k = 0; k < n; ++k) {
      search::Evaluation e;
      std::memcpy(e.policy.data(), policy.data() + k * features::kNumPolicyActions, sizeof(float) * features::kNumPolicyActions);
      std::memcpy(e.wdl.data(), wdl.data() + k * 3, sizeof(float) * 3);
      std::memcpy(e.swap.data(), swap.data() + k * 2, sizeof(float) * 2);
      std::memcpy(e.branch.data(), branch.data() + k * 2, sizeof(float) * 2);
      std::memcpy(e.propose.data(), propose.data() + k * kNumPoints, sizeof(float) * kNumPoints);
      std::memcpy(e.select.data(), select.data() + k * kNumPoints, sizeof(float) * kNumPoints);
      trees_[pending_[k].first]->complete(pending_[k].second, e);
    }
    pending_.clear();
  }

  py::dict result(int i) const {
    const search::SearchResult r = trees_.at(static_cast<size_t>(i))->result();
    py::dict d;
    d["action"] = to_string(r.action);
    d["actor"] = player_name(r.actor);
    d["value"] = r.value;
    d["proven"] = r.proven;
    d["proof_source"] = r.proof_source;
    py::list children;
    for (size_t k = 0; k < r.children.size(); ++k) {
      const auto& c = r.children[k];
      py::dict cd;
      cd["action"] = to_string(c.action);
      cd["prior"] = c.prior;
      cd["visits"] = c.visits;
      cd["q"] = c.q;
      cd["proven"] = c.proven;
      cd["target"] = k < r.policy_target.size() ? r.policy_target[k] : 0.0f;
      children.append(cd);
    }
    d["children"] = children;
    std::vector<std::string> pv;
    for (const Action& a : r.pv) pv.push_back(to_string(a));
    d["pv"] = pv;
    d["simulations"] = r.simulations;
    d["nodes"] = r.nodes;
    d["elapsed_ms"] = r.elapsed_ms;
    return d;
  }

  size_t size() const { return trees_.size(); }

 private:
  features::FeatureConfig fcfg_;
  std::vector<std::unique_ptr<search::Tree>> trees_;
  std::vector<std::pair<int, int>> pending_;
  std::vector<uint8_t> buffer_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "DojoZero C++ core: rules, features, tactics";

  m.def("point_to_str", [](int p) { return to_notation(static_cast<Point>(p)); });
  m.def("str_to_point", [](const std::string& s) {
    auto p = parse_point(s);
    if (!p) throw std::invalid_argument("bad point " + s);
    return static_cast<int>(*p);
  });
  m.def("transform_table", []() {
    py::array_t<int32_t> out({kNumTransforms, kNumPoints});
    auto r = out.mutable_unchecked<2>();
    for (int t = 0; t < kNumTransforms; ++t) {
      for (int p = 0; p < kNumPoints; ++p) r(t, p) = transform_point(t, static_cast<Point>(p));
    }
    return out;
  }, "table[t][p] = image of point p under D4 transform t (RULES.md §1.5)");
  m.def("profile_hash", [](const std::string& name) { return profile_named(name)->hash(); });

  py::class_<GameState>(m, "GameState")
      .def(py::init([](const std::string& profile) { return GameState(profile_named(profile)); }),
           py::arg("profile") = "RENJU_RIF")
      .def_static("from_spec", &from_spec, py::arg("json"), py::arg("profile") = "RENJU_RIF")
      .def_static("from_board", [](py::array_t<int8_t, py::array::c_style | py::array::forcecast> cells, int side,
                                   const std::string& profile) {
        if (cells.size() != kNumPoints) throw std::invalid_argument("cells must have 225 entries");
        return from_cells(cells.data(), side, profile_named(profile));
      }, py::arg("cells"), py::arg("side"), py::arg("profile") = "RENJU_RIF")
      .def("apply", [](GameState& g, const std::string& a) {
        auto act = parse_action_in_phase(a, g.phase());
        if (!act) throw std::invalid_argument("cannot parse action " + a);
        return g.apply(*act);
      })
      .def("undo", [](GameState& g) {
        if (!g.can_undo()) throw std::runtime_error("nothing to undo");
        g.undo();
      })
      .def("legal_actions", [](const GameState& g) {
        std::vector<std::string> out;
        for (const Action& a : g.legal_actions()) out.push_back(to_string(a));
        return out;
      })
      .def_property_readonly("phase", [](const GameState& g) { return std::string(phase_name(g.phase())); })
      .def_property_readonly("side_to_move", [](const GameState& g) { return std::string(color_name(g.side_to_move())); })
      .def_property_readonly("actor", [](const GameState& g) { return std::string(player_name(g.actor())); })
      .def_property_readonly("black_holder", [](const GameState& g) { return std::string(player_name(g.assignment().black_holder)); })
      .def_property_readonly("is_terminal", &GameState::is_terminal)
      .def_property_readonly("result", [](const GameState& g) { return std::string(result_name(g.outcome().result)); })
      .def_property_readonly("winner", [](const GameState& g) -> py::object {
        auto w = g.winner();
        if (!w) return py::none();
        return py::str(player_name(*w));
      })
      .def_property_readonly("profile", [](const GameState& g) { return g.profile().id; })
      .def("board", &board_array)
      .def("hash", &GameState::hash)
      .def("canonical_hash", [](const GameState& g) { return g.canonical_hash(); })
      .def("transformed", &GameState::transformed)
      .def("copy", [](const GameState& g) { return GameState(g); })
      .def("__str__", &GameState::to_string);

  py::class_<features::FeatureConfig>(m, "FeatureConfig")
      .def(py::init([](int history, bool tactical, bool forbidden) {
             features::FeatureConfig c;
             c.history_planes = history;
             c.tactical_planes = tactical;
             c.forbidden_plane = forbidden;
             return c;
           }),
           py::arg("history_planes") = 0, py::arg("tactical_planes") = false, py::arg("forbidden_plane") = true)
      .def_readonly("history_planes", &features::FeatureConfig::history_planes)
      .def_readonly("tactical_planes", &features::FeatureConfig::tactical_planes)
      .def_readonly("forbidden_plane", &features::FeatureConfig::forbidden_plane)
      .def_readonly("version", &features::FeatureConfig::version)
      .def_property_readonly("num_planes", &features::FeatureConfig::num_planes)
      .def_property_readonly("plane_names", &features::FeatureConfig::plane_names)
      .def("hash", &features::FeatureConfig::hash);

  m.attr("NUM_POLICY_ACTIONS") = features::kNumPolicyActions;
  m.attr("NUM_AUX_PLANES") = features::kNumAuxPlanes;

  m.def("encode", [](const GameState& g, const features::FeatureConfig& cfg) {
    py::array_t<uint8_t> out({cfg.num_planes(), kBoardSize, kBoardSize});
    features::encode(g, cfg, out.mutable_data());
    return out;
  });
  m.def("legal_mask", [](const GameState& g) {
    py::array_t<uint8_t> out(features::kNumPolicyActions);
    features::legal_policy_mask(g, out.mutable_data());
    return out;
  });
  m.def("aux_targets", [](const GameState& g) {
    py::array_t<uint8_t> out({features::kNumAuxPlanes, kBoardSize, kBoardSize});
    features::aux_targets(g, out.mutable_data());
    return out;
  });

  // Batch path for datasets: boards [N,225] int8, sides [N] uint8, profiles [N] uint8
  // (0 RENJU_RIF, 1 OMOKDOJANG_DAN). Returns (features [N,C,15,15], legal [N,226], aux [N,5,15,15]).
  m.def("encode_boards", [](py::array_t<int8_t, py::array::c_style | py::array::forcecast> boards,
                            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> sides,
                            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> profiles,
                            const features::FeatureConfig& cfg, int threads) {
    const size_t n = static_cast<size_t>(sides.size());
    if (boards.size() != static_cast<py::ssize_t>(n * kNumPoints) || profiles.size() != static_cast<py::ssize_t>(n)) {
      throw std::invalid_argument("shape mismatch");
    }
    const int C = cfg.num_planes();
    py::array_t<uint8_t> feats({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(C),
                                static_cast<py::ssize_t>(kBoardSize), static_cast<py::ssize_t>(kBoardSize)});
    py::array_t<uint8_t> legal({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(features::kNumPolicyActions)});
    py::array_t<uint8_t> aux({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(features::kNumAuxPlanes),
                              static_cast<py::ssize_t>(kBoardSize), static_cast<py::ssize_t>(kBoardSize)});
    const int8_t* b = boards.data();
    const uint8_t* s = sides.data();
    const uint8_t* pr = profiles.data();
    uint8_t* f = feats.mutable_data();
    uint8_t* l = legal.mutable_data();
    uint8_t* x = aux.mutable_data();
    const ProfilePtr kProfiles[2] = {profile_named("RENJU_RIF"), profile_named("OMOKDOJANG_DAN")};
    {
      py::gil_scoped_release release;
      parallel_for(n, threads, [&](size_t i) {
        if (pr[i] > 1) throw std::invalid_argument("profile index must be 0 or 1");
        const GameState g = from_cells(b + i * kNumPoints, s[i], kProfiles[pr[i]]);
        features::encode(g, cfg, f + i * static_cast<size_t>(C) * kNumPoints);
        features::legal_policy_mask(g, l + i * features::kNumPolicyActions);
        features::aux_targets(g, x + i * static_cast<size_t>(features::kNumAuxPlanes) * kNumPoints);
      });
    }
    return py::make_tuple(feats, legal, aux);
  }, py::arg("boards"), py::arg("sides"), py::arg("profiles"), py::arg("config"), py::arg("threads") = 0);

  py::enum_<search::RootMode>(m, "RootMode").value("PUCT", search::RootMode::PUCT).value("GUMBEL", search::RootMode::GUMBEL);
  py::class_<search::SearchConfig>(m, "SearchConfig")
      .def(py::init<>())
      .def_readwrite("simulations", &search::SearchConfig::simulations)
      .def_readwrite("c_puct", &search::SearchConfig::c_puct)
      .def_readwrite("fpu_reduction", &search::SearchConfig::fpu_reduction)
      .def_readwrite("root", &search::SearchConfig::root)
      .def_readwrite("gumbel_m", &search::SearchConfig::gumbel_m)
      .def_readwrite("gumbel_c_visit", &search::SearchConfig::gumbel_c_visit)
      .def_readwrite("gumbel_c_scale", &search::SearchConfig::gumbel_c_scale)
      .def_readwrite("gumbel_noise", &search::SearchConfig::gumbel_noise)
      .def_readwrite("leaf_tactics", &search::SearchConfig::leaf_tactics)
      .def_readwrite("leaf_vcf_nodes", &search::SearchConfig::leaf_vcf_nodes)
      .def_readwrite("root_tactics_nodes", &search::SearchConfig::root_tactics_nodes)
      .def_readwrite("max_inflight", &search::SearchConfig::max_inflight)
      .def_readwrite("virtual_loss", &search::SearchConfig::virtual_loss)
      .def_readwrite("seed", &search::SearchConfig::seed);
  py::class_<SearchBatch>(m, "SearchBatch")
      .def(py::init<const features::FeatureConfig&>())
      .def("add", &SearchBatch::add)
      .def("add_many", &SearchBatch::add_many, py::arg("states"), py::arg("config"), py::arg("threads") = 0)
      .def("done", &SearchBatch::done)
      .def("collect", &SearchBatch::collect, py::arg("max_leaves"))
      .def("apply", &SearchBatch::apply)
      .def("result", &SearchBatch::result)
      .def("__len__", &SearchBatch::size);
  m.def("heuristic_action", [](const GameState& g, uint64_t seed) {
    std::mt19937_64 rng(seed);
    return to_string(search::heuristic_action(g, rng));
  });
  m.def("random_action", [](const GameState& g, uint64_t seed) {
    std::mt19937_64 rng(seed);
    return to_string(search::random_action(g, rng));
  });

  m.def("analyze_tactics", [](const GameState& g, uint64_t vcf_nodes, uint64_t vct_nodes, uint64_t loss_nodes,
                              double max_ms) {
    tactics::TacticalDecision d;
    {
      py::gil_scoped_release release;
      d = tactics::analyze_tactics(g, {vcf_nodes, vct_nodes, loss_nodes, 40, max_ms});
    }
    py::dict out;
    out["result"] = tactics::proof_name(d.result);
    out["reason"] = tactics::tactic_reason_name(d.reason);
    out["move"] = d.move ? py::object(py::int_(static_cast<int>(*d.move))) : py::object(py::none());
    std::vector<int> pv;
    for (Point p : d.pv) pv.push_back(p == kNoPoint ? -1 : static_cast<int>(p));
    out["pv"] = pv;
    out["nodes"] = d.nodes;
    out["elapsed_ms"] = d.elapsed_ms;
    return out;
  }, py::arg("state"), py::arg("vcf_nodes") = 100000, py::arg("vct_nodes") = 50000, py::arg("loss_nodes") = 20000,
     py::arg("max_ms") = 0.0);
}
