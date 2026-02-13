#include "batched_mcts.h"
#include "mandala_game.h"
#include "lost_cities_game.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

BatchedMCTS::BatchedMCTS(const std::string& game_type, int seed,
                         int num_simulations, double c_puct,
                         double dirichlet_alpha, double dirichlet_epsilon,
                         double temperature, int temperature_threshold)
    : num_simulations_(num_simulations), c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha), dirichlet_epsilon_(dirichlet_epsilon),
      temperature_(temperature), temperature_threshold_(temperature_threshold),
      rng_(seed)
{
    if (game_type == "mandala") {
        game_ = std::make_unique<MandalaGame>();
    } else if (game_type == "lost_cities") {
        game_ = std::make_unique<LostCitiesGame>();
    } else {
        throw std::runtime_error("Unknown game type: " + game_type);
    }
}

void BatchedMCTS::init_games(int num_games) {
    games_.resize(num_games);
    active_indices_.clear();

    for (int i = 0; i < num_games; i++) {
        games_[i].state = game_->create_initial_state(rng_);
        games_[i].finished = false;
        games_[i].move_count = 0;
        games_[i].outcome = 0.0f;
        games_[i].recorded_tensors.clear();
        games_[i].recorded_policies.clear();
        games_[i].recorded_players.clear();
        active_indices_.push_back(i);
    }
}

py::array_t<float> BatchedMCTS::tensor_to_numpy(const std::vector<float>& data, int channels) const {
    py::array_t<float> arr({channels, 8, 8});
    auto buf = arr.mutable_unchecked<3>();
    int idx = 0;
    for (int c = 0; c < channels; c++)
        for (int h = 0; h < 8; h++)
            for (int w = 0; w < 8; w++)
                buf(c, h, w) = data[idx++];
    return arr;
}

void BatchedMCTS::add_dirichlet_noise(std::vector<float>& policy) {
    int n = static_cast<int>(policy.size());
    std::gamma_distribution<float> gamma(static_cast<float>(dirichlet_alpha_), 1.0f);

    std::vector<float> noise(n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        noise[i] = gamma(rng_);
        sum += noise[i];
    }
    if (sum > 0.0f) {
        float eps = static_cast<float>(dirichlet_epsilon_);
        for (int i = 0; i < n; i++) {
            noise[i] /= sum;
            policy[i] = (1.0f - eps) * policy[i] + eps * noise[i];
        }
    }
}

py::list BatchedMCTS::begin_move() {
    // Rebuild active indices
    active_indices_.clear();
    for (int i = 0; i < static_cast<int>(games_.size()); i++) {
        if (!games_[i].finished && !game_->is_terminal(*games_[i].state)) {
            active_indices_.push_back(i);
        }
    }

    py::list tensors;
    int channels = game_->tensor_channels();

    for (int idx : active_indices_) {
        auto& g = games_[idx];
        g.root = std::make_unique<MCTSNode>(1.0);

        // Get canonical state tensor for NN
        auto canonical = g.state->get_canonical();
        std::vector<float> tensor_data;
        canonical->to_tensor(tensor_data);
        tensors.append(tensor_to_numpy(tensor_data, channels));
    }

    return tensors;
}

void BatchedMCTS::set_root_policies(py::array_t<float> policies) {
    auto pol = policies.unchecked<2>();
    int num_actions = game_->num_actions();

    for (int j = 0; j < static_cast<int>(active_indices_.size()); j++) {
        int idx = active_indices_[j];
        auto& g = games_[idx];

        // Extract policy for this game
        std::vector<float> policy(num_actions);
        for (int a = 0; a < num_actions; a++) {
            policy[a] = pol(j, a);
        }

        // Add Dirichlet noise for exploration
        add_dirichlet_noise(policy);

        // Get valid moves and expand root
        std::vector<float> valid;
        game_->get_valid_moves(*g.state, valid);
        g.root->expand(policy, valid);
    }
}

py::list BatchedMCTS::simulate_step() {
    pending_leaves_.clear();
    py::list leaf_tensors;
    int channels = game_->tensor_channels();

    for (int idx : active_indices_) {
        auto& g = games_[idx];
        if (!g.root) continue;

        // Traverse tree to leaf
        MCTSNode* node = g.root.get();
        auto state = g.state->copy();

        while (!node->is_leaf() && !game_->is_terminal(*state)) {
            auto [action, child] = node->select_child(c_puct_);
            state = game_->get_next_state(*state, action);
            node = child;
        }

        if (game_->is_terminal(*state)) {
            // Terminal: backup immediately with true reward
            float value = game_->get_reward(*state, state->current_player());
            node->backup(value);
        } else {
            // Need NN eval: compute canonical tensor
            auto canonical = state->get_canonical();
            std::vector<float> tensor_data;
            canonical->to_tensor(tensor_data);
            leaf_tensors.append(tensor_to_numpy(tensor_data, channels));

            pending_leaves_.push_back({idx, node, std::move(state)});
        }
    }

    return leaf_tensors;
}

void BatchedMCTS::apply_nn_results(py::array_t<float> policies, py::array_t<float> values) {
    auto pol = policies.unchecked<2>();
    auto val = values.unchecked<1>();

    for (int j = 0; j < static_cast<int>(pending_leaves_.size()); j++) {
        auto& leaf = pending_leaves_[j];

        int num_actions = game_->num_actions();
        std::vector<float> policy(num_actions);
        for (int a = 0; a < num_actions; a++) {
            policy[a] = pol(j, a);
        }

        // Expand with valid moves from actual (non-canonical) state
        std::vector<float> valid;
        game_->get_valid_moves(*leaf.state, valid);
        leaf.node->expand(policy, valid);

        // Backup
        leaf.node->backup(static_cast<double>(val(j)));
    }

    pending_leaves_.clear();
}

std::vector<int> BatchedMCTS::finish_move() {
    std::vector<int> completed;
    int num_actions = game_->num_actions();

    for (int idx : active_indices_) {
        auto& g = games_[idx];
        if (!g.root) continue;

        // Extract visit counts
        std::vector<float> visit_counts(num_actions, 0.0f);
        for (auto& [action, child] : g.root->children) {
            visit_counts[action] = static_cast<float>(child->visit_count);
        }

        // Apply temperature
        double temp = (g.move_count < temperature_threshold_) ? temperature_ : 0.0;
        std::vector<float> action_probs(num_actions, 0.0f);

        if (temp == 0.0) {
            int best = static_cast<int>(std::max_element(visit_counts.begin(),
                                                          visit_counts.end()) - visit_counts.begin());
            action_probs[best] = 1.0f;
        } else {
            float sum = 0.0f;
            for (int a = 0; a < num_actions; a++) {
                action_probs[a] = std::pow(visit_counts[a], 1.0f / static_cast<float>(temp));
                sum += action_probs[a];
            }
            if (sum > 0.0f) {
                for (int a = 0; a < num_actions; a++) action_probs[a] /= sum;
            }
        }

        // Record training data: canonical state tensor, policy, player
        auto canonical = g.state->get_canonical();
        std::vector<float> tensor_data;
        canonical->to_tensor(tensor_data);
        g.recorded_tensors.push_back(tensor_data);
        g.recorded_policies.push_back(action_probs);
        g.recorded_players.push_back(g.state->current_player());

        // Sample action
        std::discrete_distribution<int> dist(action_probs.begin(), action_probs.end());
        int action = dist(rng_);

        // Advance game
        g.state = game_->get_next_state(*g.state, action);
        g.move_count++;
        g.root.reset();

        // Check terminal
        if (game_->is_terminal(*g.state)) {
            g.outcome = game_->get_reward(*g.state, 0);  // From player 0's perspective
            g.finished = true;
            completed.push_back(idx);
        }
    }

    // Update active indices
    active_indices_.erase(
        std::remove_if(active_indices_.begin(), active_indices_.end(),
                       [this](int idx) { return games_[idx].finished; }),
        active_indices_.end()
    );

    return completed;
}

py::tuple BatchedMCTS::get_game_data(int game_idx) {
    auto& g = games_[game_idx];
    int channels = game_->tensor_channels();
    int n = static_cast<int>(g.recorded_tensors.size());

    // Convert to numpy arrays
    py::list state_tensors;
    py::list policies;
    py::list players;

    for (int i = 0; i < n; i++) {
        state_tensors.append(tensor_to_numpy(g.recorded_tensors[i], channels));

        int num_actions = game_->num_actions();
        py::array_t<float> pol(num_actions);
        auto pol_buf = pol.mutable_unchecked<1>();
        for (int a = 0; a < num_actions; a++) {
            pol_buf(a) = g.recorded_policies[i][a];
        }
        policies.append(pol);

        players.append(py::int_(g.recorded_players[i]));
    }

    return py::make_tuple(state_tensors, policies, players, g.outcome);
}

bool BatchedMCTS::all_done() const {
    for (auto& g : games_) {
        if (!g.finished) return false;
    }
    return true;
}

int BatchedMCTS::active_count() const {
    int count = 0;
    for (auto& g : games_) {
        if (!g.finished) count++;
    }
    return count;
}
