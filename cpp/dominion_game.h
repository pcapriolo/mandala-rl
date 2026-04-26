#pragma once
#include "game_interface.h"
#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>

// --- Constants ---
static constexpr int DOM_NUM_CARD_TYPES = 31;  // IDs 0-30
static constexpr int DOM_NUM_ACTIONS = 131;
static constexpr int DOM_TENSOR_CHANNELS = 404;

// Action offsets
static constexpr int DOM_END_ACTIONS = 0;
static constexpr int DOM_END_BUYS = 1;
static constexpr int DOM_DONE_SELECTING = 2;
static constexpr int DOM_PLAY_OFFSET = 3;      // PLAY[card_type] = 3 + card_id
static constexpr int DOM_BUY_OFFSET = 34;      // BUY[card_type] = 34 + card_id
static constexpr int DOM_SELECT_OFFSET = 65;   // SELECT[card_type] = 65 + card_id
static constexpr int DOM_GAIN_OFFSET = 96;     // GAIN[card_type] = 96 + card_id
static constexpr int DOM_REVEAL_MOAT = 127;
static constexpr int DOM_DECLINE_REACTION = 128;
static constexpr int DOM_ACCEPT = 129;
static constexpr int DOM_DECLINE = 130;

// Card IDs (basic supply)
static constexpr int CARD_COPPER = 0;
static constexpr int CARD_SILVER = 1;
static constexpr int CARD_GOLD = 2;
static constexpr int CARD_ESTATE = 3;
static constexpr int CARD_DUCHY = 4;
static constexpr int CARD_PROVINCE = 5;
static constexpr int CARD_CURSE = 6;

// Card IDs (kingdom)
static constexpr int CARD_CELLAR = 7;
static constexpr int CARD_MOAT = 8;
static constexpr int CARD_CHAPEL = 9;
static constexpr int CARD_HARBINGER = 10;
static constexpr int CARD_MERCHANT = 11;
static constexpr int CARD_VASSAL = 12;
static constexpr int CARD_VILLAGE = 13;
static constexpr int CARD_WORKSHOP = 14;
static constexpr int CARD_BUREAUCRAT = 15;
static constexpr int CARD_GARDENS = 16;
static constexpr int CARD_MILITIA = 17;
static constexpr int CARD_MONEYLENDER = 18;
static constexpr int CARD_POACHER = 19;
static constexpr int CARD_REMODEL = 20;
static constexpr int CARD_SMITHY = 21;
static constexpr int CARD_THRONE_ROOM = 22;
static constexpr int CARD_ARTISAN = 23;
static constexpr int CARD_BANDIT = 24;
static constexpr int CARD_COUNCIL_ROOM = 25;
static constexpr int CARD_LIBRARY = 26;
static constexpr int CARD_MARKET = 27;
static constexpr int CARD_MINE = 28;
static constexpr int CARD_SENTRY = 29;
static constexpr int CARD_WITCH = 30;

// Card type flags
static constexpr uint8_t CF_ACTION   = 0x01;
static constexpr uint8_t CF_TREASURE = 0x02;
static constexpr uint8_t CF_VICTORY  = 0x04;
static constexpr uint8_t CF_CURSE    = 0x08;
static constexpr uint8_t CF_ATTACK   = 0x10;
static constexpr uint8_t CF_REACTION = 0x20;

struct CardDef {
    int8_t id;
    int8_t cost;
    int8_t coins;       // treasure value
    int8_t vp;          // victory points
    int8_t plus_cards;
    int8_t plus_actions;
    int8_t plus_buys;
    int8_t plus_coins;
    uint8_t type_flags;
    bool has_effect;

    bool is_action()   const { return type_flags & CF_ACTION; }
    bool is_treasure() const { return type_flags & CF_TREASURE; }
    bool is_victory()  const { return type_flags & CF_VICTORY; }
    bool is_curse()    const { return type_flags & CF_CURSE; }
    bool is_attack()   const { return type_flags & CF_ATTACK; }
    bool is_reaction() const { return type_flags & CF_REACTION; }
};

// Card definitions table
static constexpr CardDef CARD_DEFS[DOM_NUM_CARD_TYPES] = {
    // id  cost coins vp  +c +a +b +$ flags           has_effect
    {  0,  0,   1,    0,  0, 0, 0, 0, CF_TREASURE,    false }, // Copper
    {  1,  3,   2,    0,  0, 0, 0, 0, CF_TREASURE,    false }, // Silver
    {  2,  6,   3,    0,  0, 0, 0, 0, CF_TREASURE,    false }, // Gold
    {  3,  2,   0,    1,  0, 0, 0, 0, CF_VICTORY,     false }, // Estate
    {  4,  5,   0,    3,  0, 0, 0, 0, CF_VICTORY,     false }, // Duchy
    {  5,  8,   0,    6,  0, 0, 0, 0, CF_VICTORY,     false }, // Province
    {  6,  0,   0,   -1,  0, 0, 0, 0, CF_CURSE,       false }, // Curse
    {  7,  2,   0,    0,  0, 1, 0, 0, CF_ACTION,      true  }, // Cellar
    {  8,  2,   0,    0,  2, 0, 0, 0, CF_ACTION|CF_REACTION, false }, // Moat
    {  9,  2,   0,    0,  0, 0, 0, 0, CF_ACTION,      true  }, // Chapel
    { 10,  3,   0,    0,  1, 1, 0, 0, CF_ACTION,      true  }, // Harbinger
    { 11,  3,   0,    0,  1, 1, 0, 0, CF_ACTION,      true  }, // Merchant
    { 12,  3,   0,    0,  0, 0, 0, 2, CF_ACTION,      true  }, // Vassal
    { 13,  3,   0,    0,  1, 2, 0, 0, CF_ACTION,      false }, // Village
    { 14,  3,   0,    0,  0, 0, 0, 0, CF_ACTION,      true  }, // Workshop
    { 15,  4,   0,    0,  0, 0, 0, 0, CF_ACTION|CF_ATTACK, true }, // Bureaucrat
    { 16,  4,   0,    0,  0, 0, 0, 0, CF_VICTORY,     false }, // Gardens
    { 17,  4,   0,    0,  0, 0, 0, 2, CF_ACTION|CF_ATTACK, true }, // Militia
    { 18,  4,   0,    0,  0, 0, 0, 0, CF_ACTION,      true  }, // Moneylender
    { 19,  4,   0,    0,  1, 1, 0, 1, CF_ACTION,      true  }, // Poacher
    { 20,  4,   0,    0,  0, 0, 0, 0, CF_ACTION,      true  }, // Remodel
    { 21,  4,   0,    0,  3, 0, 0, 0, CF_ACTION,      false }, // Smithy
    { 22,  4,   0,    0,  0, 0, 0, 0, CF_ACTION,      true  }, // Throne Room
    { 23,  6,   0,    0,  0, 0, 0, 0, CF_ACTION,      true  }, // Artisan
    { 24,  5,   0,    0,  0, 0, 0, 0, CF_ACTION|CF_ATTACK, true }, // Bandit
    { 25,  5,   0,    0,  4, 0, 1, 0, CF_ACTION,      true  }, // Council Room
    { 26,  5,   0,    0,  0, 0, 0, 0, CF_ACTION,      true  }, // Library
    { 27,  5,   0,    0,  1, 1, 1, 1, CF_ACTION,      false }, // Market
    { 28,  5,   0,    0,  0, 0, 0, 0, CF_ACTION,      true  }, // Mine
    { 29,  5,   0,    0,  1, 1, 0, 0, CF_ACTION,      true  }, // Sentry
    { 30,  5,   0,    0,  2, 0, 0, 0, CF_ACTION|CF_ATTACK, true }, // Witch
};

// Game phases
enum DomPhase : int8_t {
    DOM_PHASE_ACTION = 0,
    DOM_PHASE_BUY    = 1,
    DOM_PHASE_SELECT = 2,
    DOM_PHASE_REACT  = 3,
};

// How a game ended. 0 = not terminated. Set by check_game_end when game_over.
enum DomTerminatedBy : int8_t {
    DOM_TERM_NONE              = 0,
    DOM_TERM_PROVINCE_EMPTY    = 1,
    DOM_TERM_THREE_PILES       = 2,
    DOM_TERM_OUTCOME_DETERMINED = 3,
    DOM_TERM_TURN_CAP          = 4,  // set by BatchedMCTS when max_turns hit
};

// Pending decision types
enum DomPendingType : int8_t {
    DOM_PEND_NONE = 0,
    DOM_PEND_SELECT_HAND,       // Cellar, Chapel, Militia, Poacher
    DOM_PEND_GAIN,              // Workshop
    DOM_PEND_REMODEL_GAIN,      // Remodel
    DOM_PEND_MINE_GAIN,         // Mine
    DOM_PEND_ARTISAN_TOPDECK,   // Artisan step 2
    DOM_PEND_HARBINGER_TOPDECK, // Harbinger
    DOM_PEND_VASSAL_PLAY,       // Vassal
    DOM_PEND_LIBRARY_DRAW,      // Library
    DOM_PEND_SENTRY_CARD,       // Sentry
    DOM_PEND_BANDIT_TRASH,      // Bandit
    DOM_PEND_BUREAUCRAT_TOPDECK,// Bureaucrat
    DOM_PEND_SELECT_TRASH_HAND, // Remodel/Mine/Moneylender step 1
    DOM_PEND_ARTISAN_GAIN,      // Artisan step 1
    DOM_PEND_THRONE_SELECT,     // Throne Room
    DOM_PEND_REACT,             // Moat reaction
};

struct DomPending {
    DomPendingType type = DOM_PEND_NONE;
    int8_t card_id = -1;          // Which card caused this
    int8_t min_select = 0;
    int8_t max_select = 0;
    int8_t selected_count = 0;
    int8_t target_player = -1;    // For attacks
    int8_t trashed_card_cost = 0; // For gain cost limits
    int8_t revealed[4] = {-1, -1, -1, -1};  // Up to 4 revealed cards
    int8_t revealed_count = 0;
    // Bandit: separate revealed cards from opponent
    int8_t bandit_revealed[2] = {-1, -1};
    int8_t bandit_revealed_count = 0;

    bool active() const { return type != DOM_PEND_NONE; }
};

struct DomPlayerState {
    std::vector<int8_t> deck;
    std::vector<int8_t> hand;
    std::vector<int8_t> discard;
    std::vector<int8_t> in_play;

    int draw(int n, std::mt19937& rng);
    int draw_deterministic(int n, size_t state_hash);
    int draw_to_list(int n, int8_t* out, std::mt19937& rng);
    int draw_to_list_deterministic(int n, int8_t* out, size_t state_hash);

    int total_cards() const {
        return static_cast<int>(deck.size() + hand.size() + discard.size() + in_play.size());
    }

    int count_vp(const int8_t* kingdom_cards, int num_kingdom) const;

    bool has_in_hand(int8_t card_id) const {
        for (auto c : hand) if (c == card_id) return true;
        return false;
    }

    int count_in_hand(int8_t card_id) const {
        int n = 0;
        for (auto c : hand) if (c == card_id) n++;
        return n;
    }

    bool remove_from_hand(int8_t card_id) {
        for (auto it = hand.begin(); it != hand.end(); ++it) {
            if (*it == card_id) { hand.erase(it); return true; }
        }
        return false;
    }

private:
    void reshuffle_discard(std::mt19937& rng);
    void reshuffle_discard_deterministic(size_t state_hash);
};

class DominionState : public GameState {
public:
    int8_t current_player_ = 0;   // Whose decision it is
    int8_t active_player_ = 0;    // Whose turn it is
    DomPlayerState players[2];
    int8_t supply[DOM_NUM_CARD_TYPES] = {};
    bool supply_disabled[DOM_NUM_CARD_TYPES] = {};  // cards disabled by curriculum
    std::vector<int8_t> trash;
    DomPhase phase = DOM_PHASE_ACTION;
    int8_t actions_remaining = 1;
    int8_t buys_remaining = 1;
    int8_t coins = 0;
    DomPending pending;
    int16_t turn_number = 1;
    bool game_over = false;
    int8_t terminated_by = DOM_TERM_NONE;  // Which condition ended the game (set in check_game_end or by BatchedMCTS for turn-cap)
    int8_t kingdom_cards[10] = {};
    int8_t num_kingdom = 0;
    int8_t merchant_silver_bonus = 0;
    bool merchant_silver_triggered = false;
    int8_t throne_card = -1;
    int8_t throne_remaining = 0;
    DomPhase save_phase = DOM_PHASE_ACTION;

    // Behavioral tracking for game quality assessment
    int16_t total_buys[2] = {};           // Total cards bought per player
    int16_t province_buys[2] = {};        // Province buys per player
    int16_t duchy_buys[2] = {};           // Duchy buys per player
    int16_t estate_buys[2] = {};          // Estate buys per player
    int16_t copper_buys[2] = {};          // Copper buys per player
    int16_t treasure_buys[2] = {};        // Silver/Gold buys per player
    int16_t action_buys[2] = {};          // Action cards bought per player
    int16_t curse_buys[2] = {};           // Curses gained (from Witch etc.) per player
    int16_t action_plays[2] = {};         // Action cards played per player
    int16_t total_moves[2] = {};          // Total decisions made per player
    int16_t total_coins_at_buy[2] = {};   // Cumulative coins at buy phase entry
    int16_t buy_phase_entries[2] = {};    // Count of buy phases (for averaging)
    int16_t turns_with_action_in_hand[2] = {};  // Turns where hand had action card(s)
    int16_t turns_action_played[2] = {};        // Turns where player played >= 1 action
    bool action_played_this_turn = false;       // Per-turn flag, reset at turn start
    int16_t card_buys[DOM_NUM_CARD_TYPES][2] = {};  // Per-card buy counts per player
    // Turn-bucketed buy tracking: 10 buckets of 5 turns each (T1-5 through T46-50)
    static constexpr int DOM_BUY_BUCKETS = 10;
    int16_t bucketed_buys[10][DOM_NUM_CARD_TYPES][2] = {};
    int16_t buy_turn_sum[DOM_NUM_CARD_TYPES][2] = {};  // Sum of turn numbers when each card bought
    int buy_bucket() const {
        int idx = (turn_number - 1) / 5;  // 0-based: turns 1-5→0, 6-10→1, ...
        return std::min(idx, DOM_BUY_BUCKETS - 1);
    }
    int16_t coins_wasted[2] = {};               // Coins left unspent at end of buy phase
    int16_t cards_trashed[2] = {};              // Cards trashed (Chapel, Remodel, Mine, Moneylender)
    int16_t cards_discarded_cellar[2] = {};     // Cards discarded via Cellar
    int16_t done_selecting_empty[2] = {};       // Times DONE_SELECTING with 0 cards selected

    // Override GameState
    int current_player() const override { return current_player_; }
    int tensor_channels() const override { return DOM_TENSOR_CHANNELS; }
    int get_turn_number() const override { return turn_number; }
    std::unique_ptr<GameState> copy() const override;
    void to_tensor(std::vector<float>& out) const override;
    std::unique_ptr<GameState> get_canonical() const override;

    size_t compute_hash() const;
};

class DominionGame : public IGame {
public:
    int num_actions() const override { return DOM_NUM_ACTIONS; }
    int tensor_channels() const override { return DOM_TENSOR_CHANNELS; }

    void set_max_action_cards(int n) { max_action_cards_ = n; }
    void set_forced_kingdom_cards(const std::vector<int>& cards) {
        forced_kingdom_cards_.clear();
        for (int c : cards) forced_kingdom_cards_.push_back(static_cast<int8_t>(c));
    }
    void set_province_supply(int n) { province_supply_ = n; }
    void set_disabled_basic_supply(const std::vector<int>& cards) {
        disabled_basic_supply_.clear();
        for (int c : cards) disabled_basic_supply_.push_back(static_cast<int8_t>(c));
    }
    void set_early_terminate_decided(bool v) { early_terminate_decided_ = v; }

    // Whether the VP outcome is mathematically determined: trailing player cannot
    // reach a tie even by claiming every VP card still in supply. Exposed for tests.
    bool is_outcome_determined(const DominionState& s) const;

    std::unique_ptr<GameState> create_initial_state(std::mt19937& rng) override;
    void get_valid_moves(const GameState& state, std::vector<float>& out) const override;
    std::unique_ptr<GameState> get_next_state(const GameState& state, int action) const override;
    bool is_terminal(const GameState& state) const override;
    float get_reward(const GameState& state, int player) const override;
    int get_score(const GameState& state, int player) const override;
    void randomize_hidden(GameState& state, std::mt19937& rng) const override;

private:
    int max_action_cards_ = 10;
    int province_supply_ = 8;
    std::vector<int8_t> forced_kingdom_cards_;  // if non-empty, use exactly these cards every game
    std::vector<int8_t> disabled_basic_supply_;  // basic supply cards to zero out (curriculum)
    bool early_terminate_decided_ = false;      // end game when VP lead > max remaining VP (DEVLOG #172)

    void play_card(DominionState& s, int8_t card_id) const;
    void handle_effect(DominionState& s, int8_t card_id) const;
    void start_attack(DominionState& s, int8_t card_id, DomPendingType dtype,
                      int8_t min_sel = 0, int8_t max_sel = 0) const;
    void resolve_attack(DominionState& s, int8_t card_id, int opp_idx,
                        int8_t min_sel = 0, int8_t max_sel = 0) const;
    void resolve_pending(DominionState& s, int action) const;
    void finish_select_hand(DominionState& s) const;
    void library_step(DominionState& s) const;
    void do_throne_replay(DominionState& s) const;
    void buy_card(DominionState& s, int8_t card_id) const;
    void do_cleanup(DominionState& s) const;
    void check_game_end(DominionState& s) const;
    int compute_vp(const DominionState& s, int player) const;

    // Initial supply counts for 2-player game
    static int initial_supply_count(int8_t card_id, bool is_kingdom);
};
