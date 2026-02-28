#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "batched_mcts.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS engine for Mandala, Lost Cities, and Dominion";

    py::class_<BatchedMCTS>(m, "BatchedMCTS")
        .def(py::init<const std::string&, int, int, double, double, double, double, int, int>(),
             py::arg("game_type"),
             py::arg("seed"),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 1.0,
             py::arg("dirichlet_alpha") = 0.3,
             py::arg("dirichlet_epsilon") = 0.25,
             py::arg("temperature") = 1.0,
             py::arg("temperature_threshold") = 30,
             py::arg("leaves_per_game") = 1)
        .def("init_games", &BatchedMCTS::init_games)
        .def("begin_move", &BatchedMCTS::begin_move)
        .def("set_root_policies", &BatchedMCTS::set_root_policies)
        .def("simulate_step", &BatchedMCTS::simulate_step)
        .def("apply_nn_results", &BatchedMCTS::apply_nn_results)
        .def("finish_move", &BatchedMCTS::finish_move)
        .def("get_game_data", &BatchedMCTS::get_game_data)
        .def("get_game_summary", &BatchedMCTS::get_game_summary)
        .def("all_done", &BatchedMCTS::all_done)
        .def("active_count", &BatchedMCTS::active_count)
        .def("get_active_players", &BatchedMCTS::get_active_players)
        .def("get_active_game_indices", &BatchedMCTS::get_active_game_indices)
        .def("get_pending_game_indices", &BatchedMCTS::get_pending_game_indices);
}
