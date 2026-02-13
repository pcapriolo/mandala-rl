#pragma once
#include "game_interface.h"
#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>

static constexpr int LC_NUM_COLORS = 5;
static constexpr int LC_CARDS_PER_COLOR = 12;
static constexpr int LC_TOTAL_CARDS = 60;
static constexpr int LC_HAND_SIZE = 8;
static constexpr int LC_INITIAL_DECK = 44;  // 60 - 2*8
static constexpr int LC_MAX_TURNS = 150;
static constexpr int LC_NUM_ACTIONS = 96;
static constexpr int LC_TENSOR_CHANNELS = 66;

struct LCCard {
    int8_t color;  // 0-4
    int8_t value;  // 0=wager, 2-10=numbered
    bool operator<(const LCCard& o) const {
        return (color != o.color) ? (color < o.color) : (value < o.value);
    }
    bool operator==(const LCCard& o) const {
        return color == o.color && value == o.value;
    }
};

class LostCitiesState : public GameState {
public:
    std::vector<LCCard> deck;
    std::array<std::vector<LCCard>, 2> hands;  // sorted
    std::array<std::array<std::vector<LCCard>, LC_NUM_COLORS>, 2> expeditions; // [player][color]
    std::array<std::vector<LCCard>, LC_NUM_COLORS> discard_piles;
    int current_player_ = 0;
    bool game_over = false;
    int turns_played = 0;

    int current_player() const override { return current_player_; }
    int tensor_channels() const override { return LC_TENSOR_CHANNELS; }
    std::unique_ptr<GameState> copy() const override;
    void to_tensor(std::vector<float>& out) const override;
    std::unique_ptr<GameState> get_canonical() const override;

    int compute_score(int player) const;
};

class LostCitiesGame : public IGame {
public:
    int num_actions() const override { return LC_NUM_ACTIONS; }
    int tensor_channels() const override { return LC_TENSOR_CHANNELS; }

    std::unique_ptr<GameState> create_initial_state(std::mt19937& rng) override;
    void get_valid_moves(const GameState& state, std::vector<float>& out) const override;
    std::unique_ptr<GameState> get_next_state(const GameState& state, int action) const override;
    bool is_terminal(const GameState& state) const override;
    float get_reward(const GameState& state, int player) const override;

private:
    static bool can_play_on_expedition(const LCCard& card, const std::vector<LCCard>& expedition);
};
