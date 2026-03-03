#pragma once
#include "game_interface.h"
#include "mcts_node.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <random>

namespace py = pybind11;

class BatchedMCTS {
public:
    BatchedMCTS(const std::string& game_type, int seed,
                int num_simulations = 800, double c_puct = 1.0,
                double dirichlet_alpha = 0.3, double dirichlet_epsilon = 0.25,
                double temperature = 1.0, int temperature_threshold = 30,
                int leaves_per_game = 1,
                double action_explore_boost = 0.0,
                double action_buy_force_rate = 0.0,
                double action_play_force_rate = 0.0);

    void init_games(int num_games);

    // Split-phase simulation loop (called from Python)
    py::list begin_move();                           // Returns list of numpy tensors
    void set_root_policies(py::array_t<float> policies); // Expand roots with noise
    py::list simulate_step();                        // Returns leaf tensors needing NN (with virtual loss)
    void apply_nn_results(py::array_t<float> policies, py::array_t<float> values);
    std::vector<int> finish_move();                  // Select actions, advance games
    py::tuple get_game_data(int game_idx);           // Get training data
    py::dict get_game_summary(int game_idx);         // Get game quality stats
    bool all_done() const;
    int active_count() const;

    // Eval support: expose game metadata for two-model evaluation
    std::vector<int> get_active_players() const;       // Current player for each active game
    std::vector<int> get_active_game_indices() const;  // Game indices of active games
    std::vector<int> get_pending_game_indices() const; // Game index for each pending leaf

private:
    struct PerGame {
        std::unique_ptr<GameState> state;
        std::vector<std::vector<float>> recorded_tensors; // state tensors
        std::vector<std::vector<float>> recorded_policies;
        std::vector<int> recorded_players;
        std::vector<std::vector<float>> recorded_belief_labels; // [12]: opp hand (6) + opp cup (6)
        int move_count = 0;
        float outcome = 0.0f;
        int score_p0 = 0;
        int score_p1 = 0;
        bool finished = false;

        // Per-move
        std::unique_ptr<MCTSNode> root;
    };

    struct PendingLeaf {
        int game_index;
        MCTSNode* node;
        std::unique_ptr<GameState> state;
    };

    std::unique_ptr<IGame> game_;
    std::string game_type_;
    int num_simulations_;
    double c_puct_;
    double dirichlet_alpha_;
    double dirichlet_epsilon_;
    double temperature_;
    int temperature_threshold_;
    int leaves_per_game_;
    double action_explore_boost_;
    double action_buy_force_rate_;
    double action_play_force_rate_;
    std::mt19937 rng_;

    int max_moves_ = 0;  // 0 = no limit
    std::vector<PerGame> games_;
    std::vector<int> active_indices_;  // indices of non-finished games
    std::vector<PendingLeaf> pending_leaves_;

    py::array_t<float> tensor_to_numpy(const std::vector<float>& data, int channels) const;
    void add_dirichlet_noise(std::vector<float>& policy);
};
