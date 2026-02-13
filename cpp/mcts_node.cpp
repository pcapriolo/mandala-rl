#include "mcts_node.h"

MCTSNode::MCTSNode(double prior, MCTSNode* parent)
    : parent(parent), prior(prior) {}

std::pair<int, MCTSNode*> MCTSNode::select_child(double c_puct) const {
    double best_score = -1e18;
    int best_action = -1;
    MCTSNode* best_child = nullptr;

    int total_visits = 0;
    for (auto& [action, child] : children) {
        total_visits += child->visit_count;
    }
    double sqrt_total = std::sqrt(static_cast<double>(total_visits));

    for (auto& [action, child] : children) {
        double q_value = -child->get_value();
        double u_value = c_puct * child->prior * sqrt_total / (1.0 + child->visit_count);
        double score = q_value + u_value;

        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child.get();
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

    // Create children
    for (int i = 0; i < n; i++) {
        if (valid_moves[i] > 0.0f) {
            children[i] = std::make_unique<MCTSNode>(masked[i], const_cast<MCTSNode*>(this));
        }
    }
}

void MCTSNode::backup(double value) {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->visit_count++;
        node->total_value += value;
        value = -value;
        node = node->parent;
    }
}
