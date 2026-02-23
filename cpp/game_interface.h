#pragma once
#include <vector>
#include <memory>
#include <random>

// Abstract game state
class GameState {
public:
    virtual ~GameState() = default;
    virtual std::unique_ptr<GameState> copy() const = 0;
    // Fill flat tensor data (C*8*8), each channel broadcast across 8x8
    virtual void to_tensor(std::vector<float>& out) const = 0;
    virtual std::unique_ptr<GameState> get_canonical() const = 0;
    virtual int current_player() const = 0;
    virtual int tensor_channels() const = 0;
};

// Abstract game engine
class IGame {
public:
    virtual ~IGame() = default;
    virtual std::unique_ptr<GameState> create_initial_state(std::mt19937& rng) = 0;
    virtual void get_valid_moves(const GameState& state, std::vector<float>& out) const = 0;
    virtual std::unique_ptr<GameState> get_next_state(const GameState& state, int action) const = 0;
    virtual bool is_terminal(const GameState& state) const = 0;
    virtual float get_reward(const GameState& state, int player) const = 0;
    virtual int get_score(const GameState& state, int player) const = 0;
    virtual int num_actions() const = 0;
    virtual int tensor_channels() const = 0;

    // Determinized MCTS: randomize hidden information from current player's perspective.
    // Default: no-op (perfect information games).
    virtual void randomize_hidden(GameState& state, std::mt19937& rng) const {}
};
