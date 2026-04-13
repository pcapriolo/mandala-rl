#include "mcts_node.h"

MCTSNode::MCTSNode(double prior, MCTSNode* parent)
    : parent(parent), prior(prior) {}

std::pair<int, MCTSNode*> MCTSNode::select_child(double c_puct,
    const std::vector<float>* valid_moves) const {
    double best_score = -1e18;
    int best_action = -1;
    MCTSNode* best_child = nullptr;

    if (valid_moves) {
        // ISMCTS mode: only consider children valid in this determinization,
        // use availability_count in exploration term instead of parent visits.
        for (auto& [action, child] : children) {
            if ((*valid_moves)[action] == 0.0f) continue;

            int eff_visits = child->visit_count + child->virtual_losses;
            double eff_value = child->total_value - static_cast<double>(child->virtual_losses);
            // Only negate Q when child is a different player (opponent's node).
            // Same-player children: positive value = good for both parent and child.
            bool same_player = (child->player >= 0 && child->player == this->player);
            double q_value = eff_visits == 0 ? 0.0
                : (same_player ? (eff_value / eff_visits) : -(eff_value / eff_visits));
            double u_value = c_puct * child->prior
                * std::sqrt(static_cast<double>(child->availability_count))
                / (1.0 + eff_visits);
            double score = q_value + u_value;

            if (score > best_score) {
                best_score = score;
                best_action = action;
                best_child = child.get();
            }
        }
    } else {
        // Standard PUCT mode (root evaluation, perfect-info fallback)
        int total_visits = 0;
        for (auto& [action, child] : children) {
            total_visits += child->visit_count + child->virtual_losses;
        }
        double sqrt_total = std::sqrt(static_cast<double>(total_visits));

        for (auto& [action, child] : children) {
            int eff_visits = child->visit_count + child->virtual_losses;
            double eff_value = child->total_value - static_cast<double>(child->virtual_losses);
            bool same_player = (child->player >= 0 && child->player == this->player);
            double q_value = eff_visits == 0 ? 0.0
                : (same_player ? (eff_value / eff_visits) : -(eff_value / eff_visits));
            double u_value = c_puct * child->prior * sqrt_total / (1.0 + eff_visits);
            double score = q_value + u_value;

            if (score > best_score) {
                best_score = score;
                best_action = action;
                best_child = child.get();
            }
        }
    }

    return {best_action, best_child};
}

void MCTSNode::expand(const std::vector<float>& policy, const std::vector<float>& valid_moves) {
    int n = static_cast<int>(policy.size());

    // Mask and renormalize
    std::vector<float> masked(n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        masked[i] = policy[i] * valid_moves[i];
        sum += masked[i];
    }

    if (sum > 0.0f) {
        for (int i = 0; i < n; i++) masked[i] /= sum;
    } else {
        // Uniform over valid
        float valid_sum = 0.0f;
        for (int i = 0; i < n; i++) valid_sum += valid_moves[i];
        if (valid_sum > 0.0f) {
            for (int i = 0; i < n; i++) masked[i] = valid_moves[i] / valid_sum;
        } else {
            return; // No valid moves (terminal)
        }
    }

    // Additive: only create children for valid actions that don't already exist.
    // Supports ISMCTS where different determinizations discover different valid actions.
    for (int i = 0; i < n; i++) {
        if (valid_moves[i] > 0.0f && children.find(i) == children.end()) {
            children[i] = std::make_unique<MCTSNode>(masked[i], const_cast<MCTSNode*>(this));
        }
    }
}

void MCTSNode::backup(double value) {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->visit_count++;
        node->total_value += value;
        // Only negate when player changes between child and parent.
        // Same-player consecutive moves (e.g. Dominion ACTION→BUY→BUY)
        // must NOT negate, or winning looks like losing.
        if (node->parent != nullptr
            && node->parent->player >= 0 && node->player >= 0
            && node->parent->player != node->player) {
            value = -value;
        }
        node = node->parent;
    }
}

void MCTSNode::apply_virtual_loss() {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->virtual_losses++;
        node = node->parent;
    }
}

void MCTSNode::remove_virtual_loss() {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->virtual_losses--;
        node = node->parent;
    }
}
