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
                         double temperature, int temperature_threshold,
                         int leaves_per_game)
    : num_simulations_(num_simulations), c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha), dirichlet_epsilon_(dirichlet_epsilon),
      temperature_(temperature), temperature_threshold_(temperature_threshold),
      leaves_per_game_(leaves_per_game), rng_(seed)
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
        games_[i].score_p0 = 0;
        games_[i].score_p1 = 0;
        games_[i].recorded_tensors.clear();
        games_[i].recorded_policies.clear();
        games_[i].recorded_players.clear();
        games_[i].recorded_belief_labels.clear();
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
    if (dirichlet_epsilon_ <= 0.0 || dirichlet_alpha_ <= 0.0) return;

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

    // Collect multiple leaves per game using virtual loss
    for (int leaf_round = 0; leaf_round < leaves_per_game_; leaf_round++) {
        for (int idx : active_indices_) {
            auto& g = games_[idx];
            if (!g.root) continue;

            // SO-ISMCTS traversal: determinize, then navigate tree filtering
            // by valid moves at each node. Never waste a simulation on an
            // invalid action. Availability tracking in UCB properly weights
            // actions that are only sometimes legal across determinizations.
            MCTSNode* node = g.root.get();
            auto state = g.state->copy();
            game_->randomize_hidden(*state, rng_);

            while (!node->is_leaf() && !game_->is_terminal(*state)) {
                std::vector<float> valid;
                game_->get_valid_moves(*state, valid);

                // Update availability counts for valid children
                for (auto& [action, child] : node->children) {
                    if (action < static_cast<int>(valid.size()) && valid[action] > 0.0f) {
                        child->availability_count++;
                    }
                }

                // Select among valid children using ISMCTS-PUCT
                auto [action, child] = node->select_child(c_puct_, &valid);
                if (action < 0) break;  // No valid children in this determinization

                state = game_->get_next_state(*state, action);
                node = child;
            }

            if (game_->is_terminal(*state)) {
                // Terminal: backup immediately with true reward
                float value = game_->get_reward(*state, state->current_player());
                node->backup(value);
            } else if (node->is_leaf()) {
                // Leaf: needs NN evaluation
                node->apply_virtual_loss();

                auto canonical = state->get_canonical();
                std::vector<float> tensor_data;
                canonical->to_tensor(tensor_data);
                leaf_tensors.append(tensor_to_numpy(tensor_data, channels));

                pending_leaves_.push_back({idx, node, std::move(state)});
            } else {
                // Internal node with no valid children in this determinization.
                // Backup neutral value — this determinization is uninformative here.
                node->backup(0.0);
            }
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

        // Remove virtual loss, then backup with real value
        leaf.node->remove_virtual_loss();
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

        // Record training data: canonical state tensor, policy, player, belief labels
        auto canonical = g.state->get_canonical();
        std::vector<float> tensor_data;
        canonical->to_tensor(tensor_data);
        g.recorded_tensors.push_back(tensor_data);
        g.recorded_policies.push_back(action_probs);
        g.recorded_players.push_back(g.state->current_player());

        // Extract belief labels: ground-truth opponent hand + cup colors
        // 12 floats: [0-5] = P(opp has color c in hand), [6-11] = P(opp has color c in cup)
        std::vector<float> belief_labels(12, 0.0f);
        if (auto* ms = dynamic_cast<MandalaState*>(g.state.get())) {
            int opp = 1 - ms->current_player_;
            for (int c : ms->hands[opp]) belief_labels[c] = 1.0f;
            for (int c : ms->cups[opp]) belief_labels[6 + c] = 1.0f;
        }
        g.recorded_belief_labels.push_back(belief_labels);

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
            g.score_p0 = game_->get_score(*g.state, 0);
            g.score_p1 = game_->get_score(*g.state, 1);
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
    py::list belief_labels;

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

        // Belief labels: 12 floats (opp hand colors + opp cup colors)
        if (i < static_cast<int>(g.recorded_belief_labels.size())) {
            py::array_t<float> bl(12);
            auto bl_buf = bl.mutable_unchecked<1>();
            for (int j = 0; j < 12; j++) {
                bl_buf(j) = g.recorded_belief_labels[i][j];
            }
            belief_labels.append(bl);
        }
    }

    return py::make_tuple(state_tensors, policies, players, g.outcome,
                          g.score_p0, g.score_p1, belief_labels);
}

py::dict BatchedMCTS::get_game_summary(int game_idx) {
    auto& g = games_[game_idx];
    py::dict summary;

    summary["game_length"] = g.move_count;
    summary["outcome"] = g.outcome;

    if (auto* ms = dynamic_cast<MandalaState*>(g.state.get())) {
        summary["game_type"] = "mandala";

        // Compute scores inline (same as MandalaGame::calculate_score)
        for (int p = 0; p < 2; p++) {
            int score = 0;
            std::unordered_map<int, int> color_values;
            for (int pos = 0; pos < static_cast<int>(ms->rivers[p].size()); pos++) {
                color_values[ms->rivers[p][pos]] = pos + 1;
            }
            for (int c : ms->cups[p]) {
                auto it = color_values.find(c);
                if (it != color_values.end()) score += it->second;
            }
            summary[p == 0 ? "score_p0" : "score_p1"] = score;
        }

        // Behavioral stats
        py::list mt_p0, mt_p1, fld_p0, fld_p1, disc_p0, disc_p1;
        for (int c = 0; c < M_NUM_COLORS; c++) {
            mt_p0.append(ms->mountain_plays[0][c]);
            mt_p1.append(ms->mountain_plays[1][c]);
            fld_p0.append(ms->field_plays[0][c]);
            fld_p1.append(ms->field_plays[1][c]);
            disc_p0.append(ms->discard_plays[0][c]);
            disc_p1.append(ms->discard_plays[1][c]);
        }
        summary["mountain_plays"] = py::make_tuple(mt_p0, mt_p1);
        summary["field_plays"] = py::make_tuple(fld_p0, fld_p1);
        summary["discard_plays"] = py::make_tuple(disc_p0, disc_p1);
        summary["total_moves"] = py::make_tuple(ms->total_moves[0], ms->total_moves[1]);
    }
    else if (auto* ls = dynamic_cast<LostCitiesState*>(g.state.get())) {
        summary["game_type"] = "lost_cities";
        summary["score_p0"] = ls->compute_score(0);
        summary["score_p1"] = ls->compute_score(1);
        summary["turns_played"] = ls->turns_played;

        py::list exp_p0, exp_p1, disc_p0, disc_p1, dpd_p0, dpd_p1;
        for (int c = 0; c < LC_NUM_COLORS; c++) {
            exp_p0.append(ls->expedition_plays[0][c]);
            exp_p1.append(ls->expedition_plays[1][c]);
            disc_p0.append(ls->color_discards[0][c]);
            disc_p1.append(ls->color_discards[1][c]);
            dpd_p0.append(ls->discard_pile_draws[0][c]);
            dpd_p1.append(ls->discard_pile_draws[1][c]);
        }
        summary["expedition_plays"] = py::make_tuple(exp_p0, exp_p1);
        summary["color_discards"] = py::make_tuple(disc_p0, disc_p1);
        summary["discard_pile_draws"] = py::make_tuple(dpd_p0, dpd_p1);
        summary["total_moves"] = py::make_tuple(ls->total_moves[0], ls->total_moves[1]);

        int exps_p0 = 0, exps_p1 = 0;
        for (int c = 0; c < LC_NUM_COLORS; c++) {
            if (!ls->expeditions[0][c].empty()) exps_p0++;
            if (!ls->expeditions[1][c].empty()) exps_p1++;
        }
        summary["num_expeditions"] = py::make_tuple(exps_p0, exps_p1);
    }

    return summary;
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

std::vector<int> BatchedMCTS::get_active_players() const {
    std::vector<int> players;
    players.reserve(active_indices_.size());
    for (int idx : active_indices_) {
        players.push_back(games_[idx].state->current_player());
    }
    return players;
}

std::vector<int> BatchedMCTS::get_active_game_indices() const {
    return active_indices_;
}

std::vector<int> BatchedMCTS::get_pending_game_indices() const {
    std::vector<int> indices;
    indices.reserve(pending_leaves_.size());
    for (auto& leaf : pending_leaves_) {
        indices.push_back(leaf.game_index);
    }
    return indices;
}
