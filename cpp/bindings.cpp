#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "batched_mcts.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS engine for Mandala and Lost Cities";

    py::class_<BatchedMCTS>(m, "BatchedMCTS")
        .def(py::init<const std::string&, int, int, double, double, double, double, int>(),
             py::arg("game_type"),
             py::arg("seed"),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 1.0,
             py::arg("dirichlet_alpha") = 0.3,
             py::arg("dirichlet_epsilon") = 0.25,
             py::arg("temperature") = 1.0,
             py::arg("temperature_threshold") = 30)
        .def("init_games", &BatchedMCTS::init_games)
        .def("begin_move", &BatchedMCTS::begin_move)
        .def("set_root_policies", &BatchedMCTS::set_root_policies)
        .def("simulate_step", &BatchedMCTS::simulate_step)
        .def("apply_nn_results", &BatchedMCTS::apply_nn_results)
        .def("finish_move", &BatchedMCTS::finish_move)
        .def("get_game_data", &BatchedMCTS::get_game_data)
        .def("all_done", &BatchedMCTS::all_done)
        .def("active_count", &BatchedMCTS::active_count);
}
