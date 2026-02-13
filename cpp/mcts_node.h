#pragma once
#include <unordered_map>
#include <memory>
#include <vector>
#include <cmath>
#include <utility>

class MCTSNode {
public:
    MCTSNode* parent;
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    int visit_count = 0;
    double total_value = 0.0;
    double prior;

    MCTSNode(double prior, MCTSNode* parent = nullptr);

    bool is_leaf() const { return children.empty(); }

    double get_value() const {
        return visit_count == 0 ? 0.0 : total_value / visit_count;
    }

    // UCB selection: -child.Q + c_puct * child.prior * sqrt(N) / (1 + child.N)
    std::pair<int, MCTSNode*> select_child(double c_puct) const;

    // Mask policy by valid_moves, renormalize, create children
    void expand(const std::vector<float>& policy, const std::vector<float>& valid_moves);

    // Backup value up tree, negating at each level
    void backup(double value);
};
