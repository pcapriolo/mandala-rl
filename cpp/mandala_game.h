#pragma once
#include "game_interface.h"
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdint>

static constexpr int M_NUM_COLORS = 6;
static constexpr int M_CARDS_PER_COLOR = 18;
static constexpr int M_TOTAL_CARDS = 108;
static constexpr int M_MAX_HAND = 8;
static constexpr int M_NUM_ACTIONS = 30;
static constexpr int M_TENSOR_CHANNELS = 59;

class MandalaState : public GameState {
public:
    std::vector<int> deck;     // card colors, back() = top
    std::vector<int> discard;
    std::array<std::vector<int>, 2> mountains;       // [mandala]
    std::array<std::array<std::vector<int>, 2>, 2> fields; // [mandala][player]
    std::array<std::vector<int>, 2> hands;            // [player]
    std::array<std::vector<int>, 2> rivers;           // [player], ordered
    std::array<std::vector<int>, 2> cups;             // [player]
    int current_player_ = 0;
    bool game_over = false;
    bool deck_reshuffled = false;
    bool game_ends_next_mandala = false;

    int current_player() const override { return current_player_; }
    int tensor_channels() const override { return M_TENSOR_CHANNELS; }
    std::unique_ptr<GameState> copy() const override;
    void to_tensor(std::vector<float>& out) const override;
    std::unique_ptr<GameState> get_canonical() const override;

    std::set<int> get_colors_in_mandala(int mandala_idx) const;
    bool is_mandala_complete(int mandala_idx) const;
    std::set<int> get_colors_in_river(int player) const;
    size_t compute_hash() const;
};

class MandalaGame : public IGame {
public:
    int num_actions() const override { return M_NUM_ACTIONS; }
    int tensor_channels() const override { return M_TENSOR_CHANNELS; }

    std::unique_ptr<GameState> create_initial_state(std::mt19937& rng) override;
    void get_valid_moves(const GameState& state, std::vector<float>& out) const override;
    std::unique_ptr<GameState> get_next_state(const GameState& state, int action) const override;
    bool is_terminal(const GameState& state) const override;
    float get_reward(const GameState& state, int player) const override;

private:
    bool can_play_to_mountain(const MandalaState& s, int color, int mandala) const;
    bool can_play_to_field(const MandalaState& s, int color, int mandala, int player) const;
    void destroy_mandala(MandalaState& s, int mandala) const;
    void check_and_reshuffle_deck(MandalaState& s) const;
    bool should_game_end(const MandalaState& s) const;
    int calculate_score(const MandalaState& s, int player) const;
};
