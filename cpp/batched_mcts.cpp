#include "batched_mcts.h"
#include "mandala_game.h"
#include "lost_cities_game.h"
#include "dominion_game.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

BatchedMCTS::BatchedMCTS(const std::string& game_type, int seed,
                         int num_simulations, double c_puct,
                         double dirichlet_alpha, double dirichlet_epsilon,
                         double temperature, int temperature_threshold,
                         double explore_epsilon,
                         int leaves_per_game,
                         double action_explore_boost,
                         double action_buy_force_rate,
                         double action_play_force_rate,
                         int max_action_cards,
                         double big_money_force_rate,
                         std::vector<int> forced_kingdom_cards,
                         std::vector<int> disabled_basic_supply,
                         int province_supply,
                         int max_turns,
                         bool early_terminate_decided)
    : game_type_(game_type), num_simulations_(num_simulations), c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha), dirichlet_epsilon_(dirichlet_epsilon),
      temperature_(temperature), temperature_threshold_(temperature_threshold),
      explore_epsilon_(explore_epsilon),
      leaves_per_game_(leaves_per_game), action_explore_boost_(action_explore_boost),
      action_buy_force_rate_(action_buy_force_rate),
      action_play_force_rate_(action_play_force_rate),
      big_money_force_rate_(big_money_force_rate),
      forced_kingdom_cards_(forced_kingdom_cards),
      disabled_basic_supply_(disabled_basic_supply), province_supply_(province_supply),
      max_turns_(max_turns), early_terminate_decided_(early_terminate_decided), rng_(seed)
{
    if (game_type == "mandala") {
        game_ = std::make_unique<MandalaGame>();
    } else if (game_type == "lost_cities") {
        game_ = std::make_unique<LostCitiesGame>();
    } else if (game_type == "dominion") {
        auto dom = std::make_unique<DominionGame>();
        dom->set_max_action_cards(max_action_cards);
        if (!forced_kingdom_cards.empty()) dom->set_forced_kingdom_cards(forced_kingdom_cards);
        if (!disabled_basic_supply_.empty()) dom->set_disabled_basic_supply(disabled_basic_supply_);
        if (province_supply_ > 0) dom->set_province_supply(province_supply_);
        dom->set_early_terminate_decided(early_terminate_decided_);
        game_ = std::move(dom);
        if (max_turns_ == 0) max_turns_ = 70;  // Default for Dominion if not set via config
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
        g.root->player = g.state->current_player();

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

        // Boost PLAY action priors in Dominion ACTION phase
        if (action_explore_boost_ > 0.0 && game_type_ == "dominion") {
            auto* ds = dynamic_cast<DominionState*>(g.state.get());
            if (ds && ds->phase == DOM_PHASE_ACTION && ds->actions_remaining > 0) {
                // Multiply PLAY action priors by boost factor, then renormalize
                bool any_boosted = false;
                for (int a = DOM_PLAY_OFFSET; a < DOM_BUY_OFFSET; a++) {
                    if (policy[a] > 0.0f) {
                        policy[a] *= static_cast<float>(action_explore_boost_);
                        any_boosted = true;
                    }
                }
                if (any_boosted) {
                    float sum = 0.0f;
                    for (int a = 0; a < num_actions; a++) sum += policy[a];
                    if (sum > 0.0f) {
                        for (int a = 0; a < num_actions; a++) policy[a] /= sum;
                    }
                }
            }
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
            int sim_depth = 0;

            while (!node->is_leaf() && !game_->is_terminal(*state) && sim_depth < 100) {
                sim_depth++;
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
                node->player = state->current_player();
            }

            if (game_->is_terminal(*state)) {
                // Terminal: backup immediately with true reward
                float value = game_->get_reward(*state, state->current_player());
                node->backup(value);
            } else if (node->is_leaf()) {
                // Auto-play forced moves: if only 1 valid action, apply it
                // without NN evaluation. This prevents spurious intermediate
                // evaluations (e.g., forced END_BUYS after buying) from biasing
                // Q-values. Walk forward until we hit a real decision (2+ valid
                // actions) or a terminal state.
                int autoplay_depth = 0;
                while (!game_->is_terminal(*state) && autoplay_depth < 20) {
                    std::vector<float> valid;
                    game_->get_valid_moves(*state, valid);
                    int num_valid = 0;
                    int forced_action = -1;
                    for (int a = 0; a < static_cast<int>(valid.size()); a++) {
                        if (valid[a] > 0.0f) {
                            num_valid++;
                            forced_action = a;
                        }
                    }
                    if (num_valid != 1) break;  // Real decision point or no valid moves
                    // Auto-play the single forced action
                    state = game_->get_next_state(*state, forced_action);
                    autoplay_depth++;
                }

                if (game_->is_terminal(*state)) {
                    // Reached terminal via forced moves — backup true reward
                    float value = game_->get_reward(*state, state->current_player());
                    node->backup(value);
                } else {
                    // Real decision point — queue for NN evaluation
                    node->apply_virtual_loss();

                    auto canonical = state->get_canonical();
                    std::vector<float> tensor_data;
                    canonical->to_tensor(tensor_data);
                    leaf_tensors.append(tensor_to_numpy(tensor_data, channels));

                    pending_leaves_.push_back({idx, node, std::move(state)});
                }
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
        leaf.node->player = leaf.state->current_player();
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

        // Check if we have any visits at all
        float total_visits = 0.0f;
        for (int a = 0; a < num_actions; a++) total_visits += visit_counts[a];

        if (total_visits == 0.0f) {
            // No children were expanded (zero valid moves seen by MCTS).
            // Fall back to uniform over valid moves to avoid hang.
            std::vector<float> valid;
            game_->get_valid_moves(*g.state, valid);
            float vsum = 0.0f;
            for (int a = 0; a < num_actions; a++) vsum += valid[a];
            if (vsum > 0.0f) {
                for (int a = 0; a < num_actions; a++) action_probs[a] = valid[a] / vsum;
            } else {
                // Truly no valid moves — force END_ACTIONS to advance game
                action_probs[0] = 1.0f;
            }
        } else if (temp == 0.0) {
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

        // Track raw MCTS Province visit fraction (before epsilon blend)
        if (game_type_ == "dominion") {
            auto* ds = dynamic_cast<DominionState*>(g.state.get());
            if (ds && ds->phase == DOM_PHASE_BUY && !ds->pending.active()) {
                int prov_action = DOM_BUY_OFFSET + CARD_PROVINCE;
                if (ds->supply[CARD_PROVINCE] > 0 && CARD_DEFS[CARD_PROVINCE].cost <= ds->coins) {
                    // Province is affordable — record raw MCTS visit fraction
                    g.mcts_province_visit_sum += action_probs[prov_action];
                    g.mcts_province_visit_count++;

                    // Track argmax: does Province have the most raw visits?
                    int prov_visits = static_cast<int>(visit_counts[prov_action]);
                    int max_visits = static_cast<int>(*std::max_element(visit_counts.begin(), visit_counts.end()));
                    if (prov_visits == max_visits && prov_visits > 0) {
                        g.mcts_province_argmax_count++;
                    }
                    g.mcts_province_decision_count++;
                }
            }
        }

        // Epsilon-greedy exploration: blend Big Money prior into policy target.
        // Province-biased when affordable, Gold-biased at $6-7, Silver at $3-5.
        // This bootstraps the prior toward correct buy priorities.
        if (explore_epsilon_ > 0.0 && g.move_count >= temperature_threshold_) {
            std::vector<float> valid;
            game_->get_valid_moves(*g.state, valid);
            float eps = static_cast<float>(explore_epsilon_);

            // Build exploration distribution: Big Money priority for Dominion buy phase
            std::vector<float> explore_dist(num_actions, 0.0f);
            bool used_bm_prior = false;

            if (game_type_ == "dominion") {
                auto* ds = dynamic_cast<DominionState*>(g.state.get());
                if (ds && ds->phase == DOM_PHASE_BUY && !ds->pending.active()) {
                    // Province-biased exploration at $8+
                    int prov_action = DOM_BUY_OFFSET + CARD_PROVINCE;
                    int gold_action = DOM_BUY_OFFSET + CARD_GOLD;
                    int silver_action = DOM_BUY_OFFSET + CARD_SILVER;

                    if (valid[prov_action] > 0.0f) {
                        // Can afford Province: 80% Province, 10% Gold, 10% Silver
                        explore_dist[prov_action] = 0.8f;
                        if (valid[gold_action] > 0.0f) explore_dist[gold_action] = 0.1f;
                        if (valid[silver_action] > 0.0f) explore_dist[silver_action] = 0.1f;
                        used_bm_prior = true;
                    } else if (valid[gold_action] > 0.0f) {
                        // Can afford Gold but not Province: 70% Gold, 30% Silver
                        explore_dist[gold_action] = 0.7f;
                        if (valid[silver_action] > 0.0f) explore_dist[silver_action] = 0.3f;
                        used_bm_prior = true;
                    }
                    // Normalize in case some weren't valid
                    if (used_bm_prior) {
                        float esum = 0.0f;
                        for (int a = 0; a < num_actions; a++) esum += explore_dist[a];
                        if (esum > 0.0f) {
                            for (int a = 0; a < num_actions; a++) explore_dist[a] /= esum;
                        }
                    }
                }
            }

            // Fallback: uniform over valid actions (non-Dominion or non-buy phase)
            if (!used_bm_prior) {
                int num_valid = 0;
                for (int a = 0; a < num_actions; a++) {
                    if (valid[a] > 0.0f) num_valid++;
                }
                if (num_valid > 0) {
                    float uniform_prob = 1.0f / num_valid;
                    for (int a = 0; a < num_actions; a++) {
                        explore_dist[a] = (valid[a] > 0.0f) ? uniform_prob : 0.0f;
                    }
                }
            }

            // Blend: (1-eps) * MCTS + eps * explore_dist
            for (int a = 0; a < num_actions; a++) {
                action_probs[a] = action_probs[a] * (1.0f - eps) + explore_dist[a] * eps;
            }
        }

        // Force play action cards in Dominion ACTION phase (epsilon-greedy exploration)
        // Prioritize +action cards (Village, etc.) before terminals (Chapel, Smithy, etc.)
        if (action_play_force_rate_ > 0.0 && game_type_ == "dominion") {
            auto* ds = dynamic_cast<DominionState*>(g.state.get());
            if (ds && ds->phase == DOM_PHASE_ACTION && ds->actions_remaining > 0) {
                std::vector<float> valid;
                game_->get_valid_moves(*g.state, valid);
                std::vector<int> plus_action_plays;   // +action cards (Village, Market, etc.)
                std::vector<int> terminal_plays;      // terminals (Chapel, Smithy, etc.)
                for (int a = DOM_PLAY_OFFSET; a < DOM_BUY_OFFSET; a++) {
                    if (a < static_cast<int>(valid.size()) && valid[a] > 0.0f) {
                        int8_t cid = static_cast<int8_t>(a - DOM_PLAY_OFFSET);
                        if (CARD_DEFS[cid].plus_actions > 0) {
                            plus_action_plays.push_back(a);
                        } else {
                            terminal_plays.push_back(a);
                        }
                    }
                }
                // Pick which set to force from: +action first if available AND terminals exist
                // (if only +action cards, or actions_remaining > 1, terminals are safe too)
                std::vector<int>* force_set = nullptr;
                if (!plus_action_plays.empty() && !terminal_plays.empty() && ds->actions_remaining == 1) {
                    force_set = &plus_action_plays;  // Play village before chapel
                } else if (!plus_action_plays.empty() || !terminal_plays.empty()) {
                    // Combine both — either no conflict, or only one type present
                    plus_action_plays.insert(plus_action_plays.end(),
                                            terminal_plays.begin(), terminal_plays.end());
                    force_set = &plus_action_plays;
                }
                if (force_set && !force_set->empty()) {
                    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
                    if (coin(rng_) < static_cast<float>(action_play_force_rate_)) {
                        std::fill(action_probs.begin(), action_probs.end(), 0.0f);
                        for (int a : *force_set) {
                            action_probs[a] = 1.0f / force_set->size();
                        }
                    }
                }
            }
        }

        // Force buy action cards in Dominion BUY phase (epsilon-greedy exploration)
        if (action_buy_force_rate_ > 0.0 && game_type_ == "dominion") {
            auto* ds = dynamic_cast<DominionState*>(g.state.get());
            if (ds && ds->phase == DOM_PHASE_BUY && !ds->pending.active()) {
                // Collect affordable kingdom action cards in supply
                std::vector<int> buyable_actions;
                for (int k = 0; k < ds->num_kingdom; k++) {
                    int8_t cid = ds->kingdom_cards[k];
                    if (CARD_DEFS[cid].is_action() && ds->supply[cid] > 0
                        && CARD_DEFS[cid].cost <= ds->coins) {
                        buyable_actions.push_back(DOM_BUY_OFFSET + cid);
                    }
                }
                if (!buyable_actions.empty()) {
                    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
                    if (coin(rng_) < static_cast<float>(action_buy_force_rate_)) {
                        // Override action_probs to uniform over buyable action cards
                        std::fill(action_probs.begin(), action_probs.end(), 0.0f);
                        for (int a : buyable_actions) {
                            action_probs[a] = 1.0f / buyable_actions.size();
                        }
                    }
                }
            }
        }

        if (big_money_force_rate_ > 0.0 && game_type_ == "dominion") {
            auto* ds = dynamic_cast<DominionState*>(g.state.get());
            if (ds && ds->phase == DOM_PHASE_BUY && !ds->pending.active()) {
                // Big money priority: Province > Gold > Duchy > Silver
                static const int BM_PRIORITY[] = {CARD_PROVINCE, CARD_GOLD, CARD_DUCHY, CARD_SILVER};
                int forced_action = -1;
                for (int cid : BM_PRIORITY) {
                    if (ds->supply[cid] > 0 && CARD_DEFS[cid].cost <= ds->coins) {
                        forced_action = DOM_BUY_OFFSET + cid;
                        break;
                    }
                }
                if (forced_action >= 0) {
                    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
                    if (coin(rng_) < static_cast<float>(big_money_force_rate_)) {
                        std::fill(action_probs.begin(), action_probs.end(), 0.0f);
                        action_probs[forced_action] = 1.0f;
                    }
                }
            }
        }

        // Record training data: canonical state tensor, policy, player, belief labels
        auto canonical = g.state->get_canonical();
        std::vector<float> tensor_data;
        canonical->to_tensor(tensor_data);
        g.recorded_tensors.push_back(tensor_data);
        g.recorded_policies.push_back(action_probs);
        g.recorded_players.push_back(g.state->current_player());

        // Extract belief labels: ground-truth opponent hand + cup/expedition colors
        // Mandala: 12 floats [0-5] hand, [6-11] cup
        // LC: 12 floats [0-4] hand, [5-9] expedition
        // Dominion: 31 floats — binary presence of each card type in opponent hand
        std::vector<float> belief_labels;
        if (auto* ms = dynamic_cast<MandalaState*>(g.state.get())) {
            belief_labels.assign(12, 0.0f);
            int opp = 1 - ms->current_player_;
            for (int c : ms->hands[opp]) belief_labels[c] = 1.0f;
            for (int c : ms->cups[opp]) belief_labels[6 + c] = 1.0f;
        } else if (auto* ls = dynamic_cast<LostCitiesState*>(g.state.get())) {
            belief_labels.assign(12, 0.0f);
            int opp = 1 - ls->current_player_;
            for (auto& card : ls->hands[opp]) belief_labels[card.color] = 1.0f;
            for (int c = 0; c < LC_NUM_COLORS; c++) {
                if (!ls->expeditions[opp][c].empty()) belief_labels[5 + c] = 1.0f;
            }
        } else if (auto* ds = dynamic_cast<DominionState*>(g.state.get())) {
            belief_labels.assign(DOM_NUM_CARD_TYPES, 0.0f);
            int opp = 1 - ds->current_player_;
            for (auto c : ds->players[opp].hand) {
                belief_labels[static_cast<int>(c)] = 1.0f;
            }
        } else {
            belief_labels.assign(12, 0.0f);
        }
        g.recorded_belief_labels.push_back(belief_labels);

        // Sample action
        std::discrete_distribution<int> dist(action_probs.begin(), action_probs.end());
        int action = dist(rng_);

        // Advance game
        g.state = game_->get_next_state(*g.state, action);
        g.move_count++;
        g.root.reset();

        // Check terminal or turn cap exceeded
        bool terminal = game_->is_terminal(*g.state);
        bool move_cap_hit = false;
        if (!terminal && max_turns_ > 0 && g.state->get_turn_number() >= max_turns_) {
            terminal = true;
            move_cap_hit = true;
            if (auto* ds = dynamic_cast<DominionState*>(g.state.get())) {
                ds->terminated_by = DOM_TERM_TURN_CAP;
            }
        }
        if (terminal) {
            g.score_p0 = game_->get_score(*g.state, 0);
            g.score_p1 = game_->get_score(*g.state, 1);
            if (move_cap_hit) {
                // get_reward guards on game_over flag; bypass it and score by VP margin
                float margin = static_cast<float>(g.score_p0 - g.score_p1);
                g.outcome = std::max(-1.0f, std::min(1.0f, margin / 5.0f));
            } else {
                g.outcome = game_->get_reward(*g.state, 0);  // From player 0's perspective
            }
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

        // Belief labels: variable size (12 for Mandala/LC, 31 for Dominion)
        if (i < static_cast<int>(g.recorded_belief_labels.size())) {
            int bl_size = static_cast<int>(g.recorded_belief_labels[i].size());
            py::array_t<float> bl(bl_size);
            auto bl_buf = bl.mutable_unchecked<1>();
            for (int j = 0; j < bl_size; j++) {
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
    else if (auto* ds = dynamic_cast<DominionState*>(g.state.get())) {
        summary["game_type"] = "dominion";
        DominionGame dom_game;
        summary["score_p0"] = dom_game.get_score(*ds, 0);
        summary["score_p1"] = dom_game.get_score(*ds, 1);
        summary["turn_number"] = static_cast<int>(ds->turn_number);
        // DEVLOG #172: which termination rule ended the game. Maps the DomTerminatedBy enum to a string
        // so Python-side analysis (dashboard, replay scans) can filter without needing the enum.
        const char* term_name;
        switch (ds->terminated_by) {
            case DOM_TERM_PROVINCE_EMPTY:     term_name = "province_empty"; break;
            case DOM_TERM_THREE_PILES:        term_name = "three_piles"; break;
            case DOM_TERM_OUTCOME_DETERMINED: term_name = "outcome_determined"; break;
            case DOM_TERM_TURN_CAP:           term_name = "turn_cap"; break;
            default:                          term_name = "none"; break;
        }
        summary["terminated_by"] = term_name;
        summary["total_buys"] = py::make_tuple(ds->total_buys[0], ds->total_buys[1]);
        summary["province_buys"] = py::make_tuple(ds->province_buys[0], ds->province_buys[1]);
        summary["duchy_buys"] = py::make_tuple(ds->duchy_buys[0], ds->duchy_buys[1]);
        summary["estate_buys"] = py::make_tuple(ds->estate_buys[0], ds->estate_buys[1]);
        summary["copper_buys"] = py::make_tuple(ds->copper_buys[0], ds->copper_buys[1]);
        summary["treasure_buys"] = py::make_tuple(ds->treasure_buys[0], ds->treasure_buys[1]);
        summary["action_buys"] = py::make_tuple(ds->action_buys[0], ds->action_buys[1]);
        summary["curse_buys"] = py::make_tuple(ds->curse_buys[0], ds->curse_buys[1]);
        summary["action_plays"] = py::make_tuple(ds->action_plays[0], ds->action_plays[1]);
        summary["total_moves"] = py::make_tuple(ds->total_moves[0], ds->total_moves[1]);
        summary["total_coins_at_buy"] = py::make_tuple(ds->total_coins_at_buy[0], ds->total_coins_at_buy[1]);
        summary["buy_phase_entries"] = py::make_tuple(ds->buy_phase_entries[0], ds->buy_phase_entries[1]);
        summary["turns_with_action_in_hand"] = py::make_tuple(ds->turns_with_action_in_hand[0], ds->turns_with_action_in_hand[1]);
        summary["turns_action_played"] = py::make_tuple(ds->turns_action_played[0], ds->turns_action_played[1]);
        summary["coins_wasted"] = py::make_tuple(ds->coins_wasted[0], ds->coins_wasted[1]);
        summary["cards_trashed"] = py::make_tuple(ds->cards_trashed[0], ds->cards_trashed[1]);
        summary["cards_discarded_cellar"] = py::make_tuple(ds->cards_discarded_cellar[0], ds->cards_discarded_cellar[1]);
        summary["done_selecting_empty"] = py::make_tuple(ds->done_selecting_empty[0], ds->done_selecting_empty[1]);

        // Per-card buy counts (only include cards with >0 buys)
        py::dict card_buys;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            int total = ds->card_buys[i][0] + ds->card_buys[i][1];
            if (total > 0) {
                card_buys[py::int_(i)] = py::make_tuple(ds->card_buys[i][0], ds->card_buys[i][1]);
            }
        }
        summary["card_buys"] = card_buys;

        // Turn-bucketed buy counts
        py::list bucketed;
        for (int b = 0; b < DominionState::DOM_BUY_BUCKETS; b++) {
            py::dict bucket;
            for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
                int total = ds->bucketed_buys[b][i][0] + ds->bucketed_buys[b][i][1];
                if (total > 0) {
                    bucket[py::int_(i)] = py::make_tuple(ds->bucketed_buys[b][i][0], ds->bucketed_buys[b][i][1]);
                }
            }
            bucketed.append(bucket);
        }
        summary["bucketed_buys"] = bucketed;

        // Buy turn sums (for computing avg turn per card)
        py::dict turn_sums;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            int total = ds->buy_turn_sum[i][0] + ds->buy_turn_sum[i][1];
            if (total > 0) {
                turn_sums[py::int_(i)] = py::make_tuple(ds->buy_turn_sum[i][0], ds->buy_turn_sum[i][1]);
            }
        }
        summary["buy_turn_sum"] = turn_sums;

        // Raw MCTS Province visit fraction (before epsilon blend)
        float mcts_prov_pct = (g.mcts_province_visit_count > 0)
            ? g.mcts_province_visit_sum / g.mcts_province_visit_count
            : 0.0f;
        summary["mcts_province_pct"] = mcts_prov_pct;

        // Province argmax: % of times Province had most visits when affordable
        float mcts_prov_argmax = (g.mcts_province_decision_count > 0)
            ? static_cast<float>(g.mcts_province_argmax_count) / g.mcts_province_decision_count
            : 0.0f;
        summary["mcts_province_argmax_pct"] = mcts_prov_argmax;
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
