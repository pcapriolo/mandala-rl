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
    int virtual_losses = 0;  // In-flight evaluations through this node
    int availability_count = 0;  // ISMCTS: times this action was legal when parent visited

    MCTSNode(double prior, MCTSNode* parent = nullptr);

    bool is_leaf() const { return children.empty(); }

    double get_value() const {
        return visit_count == 0 ? 0.0 : total_value / visit_count;
    }

    // ISMCTS-aware UCB selection. When valid_moves is provided, only considers
    // valid children and uses availability_count in exploration term.
    // When null, uses standard PUCT with parent visit count.
    std::pair<int, MCTSNode*> select_child(double c_puct,
        const std::vector<float>* valid_moves = nullptr) const;

    // Mask policy by valid_moves, renormalize, create children.
    // Additive: won't overwrite existing children (supports ISMCTS incremental expansion).
    void expand(const std::vector<float>& policy, const std::vector<float>& valid_moves);

    // Backup value up tree, negating at each level
    void backup(double value);

    // Virtual loss: penalize in-flight paths to encourage exploration diversity
    void apply_virtual_loss();
    void remove_virtual_loss();
};
