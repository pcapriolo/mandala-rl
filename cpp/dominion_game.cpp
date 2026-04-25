#include "dominion_game.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <set>

// --- DomPlayerState ---

void DomPlayerState::reshuffle_discard(std::mt19937& rng) {
    deck = discard;
    discard.clear();
    std::shuffle(deck.begin(), deck.end(), rng);
}

void DomPlayerState::reshuffle_discard_deterministic(size_t state_hash) {
    deck = discard;
    discard.clear();
    std::mt19937 rng(static_cast<unsigned>(state_hash & 0x7FFFFFFF));
    std::shuffle(deck.begin(), deck.end(), rng);
}

int DomPlayerState::draw(int n, std::mt19937& rng) {
    int drawn = 0;
    for (int i = 0; i < n; i++) {
        if (deck.empty()) {
            if (discard.empty()) break;
            reshuffle_discard(rng);
        }
        hand.push_back(deck.back());
        deck.pop_back();
        drawn++;
    }
    return drawn;
}

int DomPlayerState::draw_deterministic(int n, size_t state_hash) {
    int drawn = 0;
    for (int i = 0; i < n; i++) {
        if (deck.empty()) {
            if (discard.empty()) break;
            reshuffle_discard_deterministic(state_hash + drawn);
        }
        hand.push_back(deck.back());
        deck.pop_back();
        drawn++;
    }
    return drawn;
}

int DomPlayerState::draw_to_list(int n, int8_t* out, std::mt19937& rng) {
    int drawn = 0;
    for (int i = 0; i < n; i++) {
        if (deck.empty()) {
            if (discard.empty()) break;
            reshuffle_discard(rng);
        }
        out[drawn] = deck.back();
        deck.pop_back();
        drawn++;
    }
    return drawn;
}

int DomPlayerState::draw_to_list_deterministic(int n, int8_t* out, size_t state_hash) {
    int drawn = 0;
    for (int i = 0; i < n; i++) {
        if (deck.empty()) {
            if (discard.empty()) break;
            reshuffle_discard_deterministic(state_hash + drawn);
        }
        out[drawn] = deck.back();
        deck.pop_back();
        drawn++;
    }
    return drawn;
}

int DomPlayerState::count_vp(const int8_t* kingdom_cards, int num_kingdom) const {
    int vp = 0;
    int total = total_cards();
    auto count_card = [&](int8_t cid) {
        if (cid == CARD_GARDENS) {
            vp += total / 10;
        } else {
            vp += CARD_DEFS[cid].vp;
        }
    };
    for (auto c : deck) count_card(c);
    for (auto c : hand) count_card(c);
    for (auto c : discard) count_card(c);
    for (auto c : in_play) count_card(c);
    return vp;
}

// --- DominionState ---

std::unique_ptr<GameState> DominionState::copy() const {
    auto s = std::make_unique<DominionState>();
    s->current_player_ = current_player_;
    s->active_player_ = active_player_;
    s->players[0].deck = players[0].deck;
    s->players[0].hand = players[0].hand;
    s->players[0].discard = players[0].discard;
    s->players[0].in_play = players[0].in_play;
    s->players[1].deck = players[1].deck;
    s->players[1].hand = players[1].hand;
    s->players[1].discard = players[1].discard;
    s->players[1].in_play = players[1].in_play;
    std::memcpy(s->supply, supply, sizeof(supply));
    std::memcpy(s->supply_disabled, supply_disabled, sizeof(supply_disabled));
    s->trash = trash;
    s->phase = phase;
    s->actions_remaining = actions_remaining;
    s->buys_remaining = buys_remaining;
    s->coins = coins;
    s->pending = pending;
    s->turn_number = turn_number;
    s->game_over = game_over;
    std::memcpy(s->kingdom_cards, kingdom_cards, sizeof(kingdom_cards));
    s->num_kingdom = num_kingdom;
    s->merchant_silver_bonus = merchant_silver_bonus;
    s->merchant_silver_triggered = merchant_silver_triggered;
    s->throne_card = throne_card;
    s->throne_remaining = throne_remaining;
    s->save_phase = save_phase;
    std::memcpy(s->total_buys, total_buys, sizeof(total_buys));
    std::memcpy(s->province_buys, province_buys, sizeof(province_buys));
    std::memcpy(s->duchy_buys, duchy_buys, sizeof(duchy_buys));
    std::memcpy(s->estate_buys, estate_buys, sizeof(estate_buys));
    std::memcpy(s->copper_buys, copper_buys, sizeof(copper_buys));
    std::memcpy(s->treasure_buys, treasure_buys, sizeof(treasure_buys));
    std::memcpy(s->action_buys, action_buys, sizeof(action_buys));
    std::memcpy(s->curse_buys, curse_buys, sizeof(curse_buys));
    std::memcpy(s->action_plays, action_plays, sizeof(action_plays));
    std::memcpy(s->total_moves, total_moves, sizeof(total_moves));
    std::memcpy(s->total_coins_at_buy, total_coins_at_buy, sizeof(total_coins_at_buy));
    std::memcpy(s->buy_phase_entries, buy_phase_entries, sizeof(buy_phase_entries));
    std::memcpy(s->turns_with_action_in_hand, turns_with_action_in_hand, sizeof(turns_with_action_in_hand));
    std::memcpy(s->turns_action_played, turns_action_played, sizeof(turns_action_played));
    s->action_played_this_turn = action_played_this_turn;
    std::memcpy(s->card_buys, card_buys, sizeof(card_buys));
    std::memcpy(s->bucketed_buys, bucketed_buys, sizeof(bucketed_buys));
    std::memcpy(s->buy_turn_sum, buy_turn_sum, sizeof(buy_turn_sum));
    std::memcpy(s->coins_wasted, coins_wasted, sizeof(coins_wasted));
    std::memcpy(s->cards_trashed, cards_trashed, sizeof(cards_trashed));
    std::memcpy(s->cards_discarded_cellar, cards_discarded_cellar, sizeof(cards_discarded_cellar));
    std::memcpy(s->done_selecting_empty, done_selecting_empty, sizeof(done_selecting_empty));
    return s;
}

void DominionState::to_tensor(std::vector<float>& out) const {
    int total = DOM_TENSOR_CHANNELS * 64;
    out.assign(total, 0.0f);

    auto set_channel = [&out](int ch, float val) {
        int offset = ch * 64;
        for (int i = 0; i < 64; i++) out[offset + i] = val;
    };

    const auto& me = players[0];
    const auto& opp = players[1];

    // Ch 0-30: My full deck composition (count per card type / 12)
    {
        int counts[DOM_NUM_CARD_TYPES] = {};
        for (auto c : me.deck) counts[c]++;
        for (auto c : me.hand) counts[c]++;
        for (auto c : me.discard) counts[c]++;
        for (auto c : me.in_play) counts[c]++;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            if (counts[i] > 0) set_channel(i, counts[i] / 12.0f);
        }
    }

    // Ch 31-61: My hand (count per card type / 10)
    {
        int counts[DOM_NUM_CARD_TYPES] = {};
        for (auto c : me.hand) counts[c]++;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            if (counts[i] > 0) set_channel(31 + i, counts[i] / 10.0f);
        }
    }

    // Ch 62-92: My in-play area (count per card type / 5)
    {
        int counts[DOM_NUM_CARD_TYPES] = {};
        for (auto c : me.in_play) counts[c]++;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            if (counts[i] > 0) set_channel(62 + i, counts[i] / 5.0f);
        }
    }

    // Ch 93-123: Supply remaining (count per pile / 12)
    for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
        if (supply[i] > 0) set_channel(93 + i, supply[i] / 12.0f);
    }

    // Ch 124: Opponent hand size / 10
    set_channel(124, static_cast<float>(opp.hand.size()) / 10.0f);

    // Ch 125: Opponent total cards / 50
    set_channel(125, opp.total_cards() / 50.0f);

    // Ch 126: My coins this turn / 12
    set_channel(126, coins / 12.0f);

    // Ch 127: Actions remaining / 10
    set_channel(127, actions_remaining / 10.0f);

    // Ch 128: Buys remaining / 5
    set_channel(128, buys_remaining / 5.0f);

    // Ch 129: Phase encoding (ordinal)
    {
        float phase_val = 0.0f;
        switch (phase) {
            case DOM_PHASE_ACTION: phase_val = 0.0f; break;
            case DOM_PHASE_BUY:    phase_val = 0.33f; break;
            case DOM_PHASE_SELECT: phase_val = 0.67f; break;
            case DOM_PHASE_REACT:  phase_val = 1.0f; break;
        }
        set_channel(129, phase_val);
    }

    // Ch 130: Turn number / 40
    set_channel(130, std::min(turn_number / 40.0f, 1.0f));

    // Ch 131: ZEROED (duplicate of Ch 93+5 supply[Province])
    // set_channel(131, supply[CARD_PROVINCE] / 8.0f);

    // Ch 132: Empty supply piles / 3
    {
        int empty = 0;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            if (supply[i] == 0) {
                // Only count piles that are in the game
                bool in_game = (i < 7); // basic supply always in game
                if (!in_game) {
                    for (int k = 0; k < num_kingdom; k++) {
                        if (kingdom_cards[k] == i) { in_game = true; break; }
                    }
                }
                if (in_game) empty++;
            }
        }
        set_channel(132, empty / 3.0f);
    }

    // Ch 133-137: Pending decision context
    if (pending.active()) {
        set_channel(133, pending.card_id / static_cast<float>(DOM_NUM_CARD_TYPES));
        // Encode pending type as fractional
        float type_val = 0.0f;
        switch (pending.type) {
            case DOM_PEND_SELECT_HAND:       type_val = 0.1f; break;
            case DOM_PEND_GAIN:              type_val = 0.3f; break;
            case DOM_PEND_REACT:             type_val = 0.4f; break;
            case DOM_PEND_VASSAL_PLAY:       type_val = 0.5f; break;
            case DOM_PEND_LIBRARY_DRAW:      type_val = 0.6f; break;
            case DOM_PEND_SENTRY_CARD:       type_val = 0.7f; break;
            case DOM_PEND_BANDIT_TRASH:      type_val = 0.8f; break;
            case DOM_PEND_ARTISAN_TOPDECK:   type_val = 0.9f; break;
            case DOM_PEND_MINE_GAIN:         type_val = 0.15f; break;
            case DOM_PEND_HARBINGER_TOPDECK: type_val = 0.25f; break;
            case DOM_PEND_REMODEL_GAIN:      type_val = 0.35f; break;
            case DOM_PEND_BUREAUCRAT_TOPDECK:type_val = 0.45f; break;
            case DOM_PEND_SELECT_TRASH_HAND: type_val = 0.55f; break;
            case DOM_PEND_ARTISAN_GAIN:      type_val = 0.65f; break;
            case DOM_PEND_THRONE_SELECT:     type_val = 0.75f; break;
            default: break;
        }
        set_channel(134, type_val);
        set_channel(135, pending.max_select > 0 ? pending.max_select / 10.0f : 0.0f);
        set_channel(136, pending.selected_count / 10.0f);
        set_channel(137, pending.target_player >= 0 ? 1.0f : 0.0f);
    }

    // Ch 138: My VP / 20 (realistic game range 0-20 VP, amplifies signal)
    {
        int my_vp = me.count_vp(kingdom_cards, num_kingdom);
        set_channel(138, std::max(-0.5f, std::min(1.0f, my_vp / 20.0f)));
    }

    // Ch 139: Opponent VP / 20
    {
        int opp_vp = opp.count_vp(kingdom_cards, num_kingdom);
        set_channel(139, std::max(-0.5f, std::min(1.0f, opp_vp / 20.0f)));
    }

    // Ch 140: VP differential (my_vp - opp_vp) / 20
    {
        int my_vp = me.count_vp(kingdom_cards, num_kingdom);
        int opp_vp = opp.count_vp(kingdom_cards, num_kingdom);
        set_channel(140, std::max(-1.0f, std::min(1.0f, (my_vp - opp_vp) / 20.0f)));
    }

    // Ch 141: My total cards / 50
    set_channel(141, me.total_cards() / 50.0f);

    // Ch 142: ZEROED (derivable from Ch 135 + 136)
    // if (pending.active() && pending.max_select > 0) {
    //     set_channel(142, static_cast<float>(pending.selected_count) / pending.max_select);
    // }

    // Ch 143-150: Derived synergy features
    // These pre-compute relationships the network would otherwise need many
    // layers to discover from raw card counts.
    {
        // Count cards in my full deck
        int my_treasure_coins = 0;  // Total static coin value of all treasures
        int my_total = 0;

        auto scan = [&](int8_t cid) {
            my_total++;
            if (CARD_DEFS[cid].is_treasure()) my_treasure_coins += CARD_DEFS[cid].coins;
        };
        for (auto c : me.deck) scan(c);
        for (auto c : me.hand) scan(c);
        for (auto c : me.discard) scan(c);
        for (auto c : me.in_play) scan(c);

        // Ch 143: My coins wasted per buy phase (buying efficiency)
        if (buy_phase_entries[0] > 0) {
            float wasted_per_buy = static_cast<float>(coins_wasted[0]) / buy_phase_entries[0];
            set_channel(143, std::min(wasted_per_buy / 4.0f, 1.0f));
        }

        // Ch 144: Opponent coins wasted per buy phase
        if (buy_phase_entries[1] > 0) {
            float opp_wasted = static_cast<float>(coins_wasted[1]) / buy_phase_entries[1];
            set_channel(144, std::min(opp_wasted / 4.0f, 1.0f));
        }

        // Ch 145: Payload — treasure-only coins per card, tightened /2
        if (my_total > 0) {
            set_channel(145, static_cast<float>(my_treasure_coins) / my_total / 2.0f);
        }

        // Ch 146: Curse ratio — curse count / total deck size
        // Only curses are unambiguously bad junk (Provinces are good, Estates neutral early)
        if (my_total > 0) {
            int my_curses = 0;
            for (auto c : me.deck) if (c == CARD_CURSE) my_curses++;
            for (auto c : me.hand) if (c == CARD_CURSE) my_curses++;
            for (auto c : me.discard) if (c == CARD_CURSE) my_curses++;
            for (auto c : me.in_play) if (c == CARD_CURSE) my_curses++;
            set_channel(146, static_cast<float>(my_curses) / my_total);
        }

        // Ch 147: ZEROED (terminal collision risk — over-engineered, raw counts suffice)

        // Ch 148: Expected hand value — rough coins per 5-card hand / 8 (tightened)
        if (my_total > 0) {
            float expected = static_cast<float>(my_treasure_coins) * 5.0f / my_total;
            set_channel(148, std::min(1.0f, expected / 8.0f));
        }

        // Ch 149: Opponent deck composition — their expected hand value / 8 (tightened)
        {
            int opp_treasure_coins = 0;
            int opp_total = opp.total_cards();
            for (auto c : opp.deck) opp_treasure_coins += CARD_DEFS[c].coins;
            for (auto c : opp.hand) opp_treasure_coins += CARD_DEFS[c].coins;
            for (auto c : opp.discard) opp_treasure_coins += CARD_DEFS[c].coins;
            for (auto c : opp.in_play) opp_treasure_coins += CARD_DEFS[c].coins;
            if (opp_total > 0) {
                float opp_expected = static_cast<float>(opp_treasure_coins) * 5.0f / opp_total;
                set_channel(149, std::min(1.0f, opp_expected / 8.0f));
            }
        }

        // Ch 150: Attack pressure — do I have attack cards?
        {
            int my_attacks = 0;
            for (auto c : me.deck) if (CARD_DEFS[c].is_attack()) my_attacks++;
            for (auto c : me.hand) if (CARD_DEFS[c].is_attack()) my_attacks++;
            for (auto c : me.discard) if (CARD_DEFS[c].is_attack()) my_attacks++;
            for (auto c : me.in_play) if (CARD_DEFS[c].is_attack()) my_attacks++;
            set_channel(150, std::min(1.0f, my_attacks / 3.0f));
        }
    }

    // Ch 151-155: Action card awareness channels
    // These surface action card bonuses so the model can reason about playing them.
    {
        // Ch 151: Treasure coins currently in hand / 8 (tightened, realistic range 3-8)
        int hand_treasure_coins = 0;
        for (auto c : me.hand) {
            if (CARD_DEFS[c].is_treasure()) hand_treasure_coins += CARD_DEFS[c].coins;
        }
        set_channel(151, std::min(1.0f, hand_treasure_coins / 8.0f));

        // Ch 152-154: Best action card bonuses from playable actions in hand
        // Conditional on actions_remaining > 0 (otherwise you can't play actions)
        int best_cards = 0, best_actions = 0, best_coins = 0;
        int action_count_in_hand = 0;
        if (actions_remaining > 0) {
            for (auto c : me.hand) {
                if (CARD_DEFS[c].is_action()) {
                    action_count_in_hand++;
                    best_cards = std::max(best_cards, static_cast<int>(CARD_DEFS[c].plus_cards));
                    best_actions = std::max(best_actions, static_cast<int>(CARD_DEFS[c].plus_actions));
                    best_coins = std::max(best_coins, static_cast<int>(CARD_DEFS[c].plus_coins));
                }
            }
        } else {
            // Still count actions in hand for ch 155 even if can't play
            for (auto c : me.hand) {
                if (CARD_DEFS[c].is_action()) action_count_in_hand++;
            }
        }

        // Ch 152: Best +cards from playable action in hand / 5
        set_channel(152, std::min(1.0f, best_cards / 5.0f));

        // Ch 153: Best +actions from playable action in hand / 5
        set_channel(153, std::min(1.0f, best_actions / 5.0f));

        // Ch 154: Best +coins from playable action in hand / 5
        set_channel(154, std::min(1.0f, best_coins / 5.0f));

        // Ch 155: Number of action cards in hand / 5
        set_channel(155, std::min(1.0f, action_count_in_hand / 5.0f));
    }

    // Ch 156-186: Opponent full deck composition (count per card type / 12)
    // All buys/gains/trashing are public info in Dominion
    {
        int counts[DOM_NUM_CARD_TYPES] = {};
        for (auto c : opp.deck) counts[c]++;
        for (auto c : opp.hand) counts[c]++;
        for (auto c : opp.discard) counts[c]++;
        for (auto c : opp.in_play) counts[c]++;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            if (counts[i] > 0) set_channel(156 + i, counts[i] / 12.0f);
        }
    }

    // Ch 187-217: Opponent discard pile (count per card type / 12)
    // Discard is public. Combined with total composition, network can compute
    // unknown pool (hand + draw_pile = total - discard - in_play).
    {
        int counts[DOM_NUM_CARD_TYPES] = {};
        for (auto c : opp.discard) counts[c]++;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            if (counts[i] > 0) set_channel(187 + i, counts[i] / 12.0f);
        }
    }

    // Ch 218-248: My cumulative buy counts per card type / 4
    for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
        if (card_buys[i][0] > 0)
            set_channel(218 + i, std::min(1.0f, card_buys[i][0] / 4.0f));
    }

    // Ch 249-279: Opponent cumulative buy counts per card type / 4
    for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
        if (card_buys[i][1] > 0)
            set_channel(249 + i, std::min(1.0f, card_buys[i][1] / 4.0f));
    }
}

std::unique_ptr<GameState> DominionState::get_canonical() const {
    auto s = std::make_unique<DominionState>();
    // Copy all fields
    *s = *this;
    // Deep copy vectors
    s->players[0].deck = players[0].deck;
    s->players[0].hand = players[0].hand;
    s->players[0].discard = players[0].discard;
    s->players[0].in_play = players[0].in_play;
    s->players[1].deck = players[1].deck;
    s->players[1].hand = players[1].hand;
    s->players[1].discard = players[1].discard;
    s->players[1].in_play = players[1].in_play;
    s->trash = trash;

    if (current_player_ == 0) return s;

    // Swap players
    std::swap(s->players[0], s->players[1]);
    std::swap(s->coins_wasted[0], s->coins_wasted[1]);
    std::swap(s->buy_phase_entries[0], s->buy_phase_entries[1]);
    // Swap buy history tracking (used by tensor channels 218-279)
    for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
        std::swap(s->card_buys[i][0], s->card_buys[i][1]);
        std::swap(s->buy_turn_sum[i][0], s->buy_turn_sum[i][1]);
        for (int b = 0; b < DOM_BUY_BUCKETS; b++) {
            std::swap(s->bucketed_buys[b][i][0], s->bucketed_buys[b][i][1]);
        }
    }
    s->active_player_ = 1 - s->active_player_;
    s->current_player_ = 0;
    if (s->pending.target_player >= 0) {
        s->pending.target_player = 1 - s->pending.target_player;
    }
    return s;
}

size_t DominionState::compute_hash() const {
    size_t h = 0;
    auto combine = [&h](size_t v) {
        h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
    };
    for (int p = 0; p < 2; p++) {
        for (auto c : players[p].hand) combine(c + 100);
        combine(players[p].hand.size());
        for (auto c : players[p].deck) combine(c + 200);
        combine(players[p].deck.size());
        for (auto c : players[p].discard) combine(c + 300);
        combine(players[p].discard.size());
        for (auto c : players[p].in_play) combine(c + 400);
    }
    combine(current_player_);
    combine(phase);
    combine(actions_remaining);
    combine(buys_remaining);
    combine(coins);
    combine(turn_number);
    for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) combine(supply[i]);
    return h;
}

// --- DominionGame ---

int DominionGame::initial_supply_count(int8_t card_id, bool is_kingdom) {
    switch (card_id) {
        case CARD_COPPER:   return 46;  // 60 - 7*2
        case CARD_SILVER:   return 40;
        case CARD_GOLD:     return 30;
        case CARD_ESTATE:   return 8;   // 2-player
        case CARD_DUCHY:    return 8;
        case CARD_PROVINCE: return 8;   // Standard 2-player count; curriculum overrides via set_province_supply
        case CARD_CURSE:    return 10;  // (num_players-1)*10
        default:
            if (is_kingdom) {
                if (CARD_DEFS[card_id].is_victory()) return 8;
                return 10;
            }
            return 0;
    }
}

std::unique_ptr<GameState> DominionGame::create_initial_state(std::mt19937& rng) {
    auto s = std::make_unique<DominionState>();

    // Select kingdom cards
    if (!forced_kingdom_cards_.empty()) {
        // Fixed curriculum kingdom — same cards every game
        s->num_kingdom = std::min((int)forced_kingdom_cards_.size(), 10);
        for (int i = 0; i < s->num_kingdom; i++) s->kingdom_cards[i] = forced_kingdom_cards_[i];
        std::sort(s->kingdom_cards, s->kingdom_cards + s->num_kingdom);
    } else {
        // Random selection respecting max_action_cards_ curriculum limit
        std::vector<int8_t> action_pool, non_action_pool;
        for (int i = 7; i <= 30; i++) {
            if (CARD_DEFS[i].is_action() || CARD_DEFS[i].is_reaction())
                action_pool.push_back(static_cast<int8_t>(i));
            else
                non_action_pool.push_back(static_cast<int8_t>(i));
        }
        std::shuffle(action_pool.begin(), action_pool.end(), rng);
        std::shuffle(non_action_pool.begin(), non_action_pool.end(), rng);
        int n_action = std::min(max_action_cards_, static_cast<int>(action_pool.size()));
        int n_fill = std::min(10 - n_action, static_cast<int>(non_action_pool.size()));
        s->num_kingdom = n_action + n_fill;
        int idx = 0;
        for (int i = 0; i < n_action; i++) s->kingdom_cards[idx++] = action_pool[i];
        for (int i = 0; i < n_fill; i++) s->kingdom_cards[idx++] = non_action_pool[i];
        std::sort(s->kingdom_cards, s->kingdom_cards + s->num_kingdom);
    }

    // Setup supply
    for (int i = 0; i < 7; i++) {
        s->supply[i] = initial_supply_count(static_cast<int8_t>(i), false);
    }
    for (int i = 0; i < 10; i++) {
        int8_t cid = s->kingdom_cards[i];
        s->supply[cid] = initial_supply_count(cid, true);
    }

    // Override province supply count to whatever the curriculum sets.
    // BUG FIX: previously gated `if (province_supply_ != 8)` which silently
    // *skipped* the override when curriculum requested 8 — but the default in
    // initial_supply_count was 3, not 8, so setting `province_supply: 8` left
    // the supply at 3. Now unconditional. (DEVLOG #174.)
    s->supply[CARD_PROVINCE] = province_supply_;

    // Disable basic supply cards for curriculum
    for (auto cid : disabled_basic_supply_) {
        s->supply[cid] = 0;
        s->supply_disabled[cid] = true;
    }

    // Setup players: 7 Copper + 3 Estate, shuffled
    for (int p = 0; p < 2; p++) {
        std::vector<int8_t> starting;
        for (int i = 0; i < 7; i++) starting.push_back(CARD_COPPER);
        for (int i = 0; i < 3; i++) starting.push_back(CARD_ESTATE);
        std::shuffle(starting.begin(), starting.end(), rng);
        s->players[p].deck = starting;
        s->players[p].draw(5, rng);
    }

    s->current_player_ = 0;
    s->active_player_ = 0;
    s->phase = DOM_PHASE_ACTION;
    s->actions_remaining = 1;
    s->buys_remaining = 1;
    s->coins = 0;
    s->turn_number = 1;
    return s;
}

void DominionGame::get_valid_moves(const GameState& state_base, std::vector<float>& out) const {
    out.assign(DOM_NUM_ACTIONS, 0.0f);
    auto& s = static_cast<const DominionState&>(state_base);

    if (s.game_over) return;

    const auto& player = s.players[s.current_player_];

    // Handle pending decisions
    if (s.pending.active()) {
        const auto& p = s.pending;

        switch (p.type) {
        case DOM_PEND_REACT:
            if (player.has_in_hand(CARD_MOAT))
                out[DOM_REVEAL_MOAT] = 1.0f;
            out[DOM_DECLINE_REACTION] = 1.0f;
            break;

        case DOM_PEND_SELECT_HAND: {
            // Select cards from hand
            std::set<int8_t> unique_hand(player.hand.begin(), player.hand.end());
            for (auto cid : unique_hand)
                out[DOM_SELECT_OFFSET + cid] = 1.0f;
            if (p.selected_count >= p.min_select)
                out[DOM_DONE_SELECTING] = 1.0f;
            break;
        }

        case DOM_PEND_GAIN: {
            int cost_limit = p.trashed_card_cost;
            bool any_valid = false;
            for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
                if (s.supply[i] > 0 && CARD_DEFS[i].cost <= cost_limit) {
                    out[DOM_GAIN_OFFSET + i] = 1.0f;
                    any_valid = true;
                }
            }
            if (!any_valid) out[DOM_DONE_SELECTING] = 1.0f;
            break;
        }

        case DOM_PEND_REMODEL_GAIN: {
            int cost_limit = p.trashed_card_cost + 2;
            bool any_valid = false;
            for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
                if (s.supply[i] > 0 && CARD_DEFS[i].cost <= cost_limit) {
                    out[DOM_GAIN_OFFSET + i] = 1.0f;
                    any_valid = true;
                }
            }
            if (!any_valid) out[DOM_DONE_SELECTING] = 1.0f;
            break;
        }

        case DOM_PEND_MINE_GAIN: {
            int cost_limit = p.trashed_card_cost + 3;
            bool any_valid = false;
            for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
                if (s.supply[i] > 0 && CARD_DEFS[i].is_treasure() && CARD_DEFS[i].cost <= cost_limit) {
                    out[DOM_GAIN_OFFSET + i] = 1.0f;
                    any_valid = true;
                }
            }
            if (!any_valid) out[DOM_DONE_SELECTING] = 1.0f;
            break;
        }

        case DOM_PEND_ARTISAN_TOPDECK: {
            std::set<int8_t> unique_hand(player.hand.begin(), player.hand.end());
            for (auto cid : unique_hand)
                out[DOM_SELECT_OFFSET + cid] = 1.0f;
            break;
        }

        case DOM_PEND_HARBINGER_TOPDECK: {
            std::set<int8_t> unique_discard(player.discard.begin(), player.discard.end());
            for (auto cid : unique_discard)
                out[DOM_SELECT_OFFSET + cid] = 1.0f;
            out[DOM_DONE_SELECTING] = 1.0f;  // Can skip
            break;
        }

        case DOM_PEND_VASSAL_PLAY:
            out[DOM_ACCEPT] = 1.0f;
            out[DOM_DECLINE] = 1.0f;
            break;

        case DOM_PEND_LIBRARY_DRAW:
            out[DOM_ACCEPT] = 1.0f;
            out[DOM_DECLINE] = 1.0f;
            break;

        case DOM_PEND_SENTRY_CARD:
            out[DOM_SELECT_OFFSET + p.revealed[0]] = 1.0f;  // Trash
            out[DOM_ACCEPT] = 1.0f;   // Discard
            out[DOM_DECLINE] = 1.0f;  // Keep on deck
            break;

        case DOM_PEND_BANDIT_TRASH: {
            bool any_valid = false;
            for (int i = 0; i < p.bandit_revealed_count; i++) {
                int8_t cid = p.bandit_revealed[i];
                if (cid != CARD_COPPER && CARD_DEFS[cid].is_treasure()) {
                    out[DOM_SELECT_OFFSET + cid] = 1.0f;
                    any_valid = true;
                }
            }
            if (!any_valid) out[DOM_DONE_SELECTING] = 1.0f;
            break;
        }

        case DOM_PEND_BUREAUCRAT_TOPDECK: {
            bool has_victory = false;
            std::set<int8_t> unique_hand(player.hand.begin(), player.hand.end());
            for (auto cid : unique_hand) {
                if (CARD_DEFS[cid].is_victory()) {
                    out[DOM_SELECT_OFFSET + cid] = 1.0f;
                    has_victory = true;
                }
            }
            if (!has_victory) out[DOM_DONE_SELECTING] = 1.0f;
            break;
        }

        case DOM_PEND_SELECT_TRASH_HAND: {
            if (p.card_id == CARD_MONEYLENDER) {
                if (player.has_in_hand(CARD_COPPER))
                    out[DOM_SELECT_OFFSET + CARD_COPPER] = 1.0f;
                else
                    out[DOM_DONE_SELECTING] = 1.0f;
            } else if (p.card_id == CARD_MINE) {
                bool has_treasure = false;
                std::set<int8_t> unique_hand(player.hand.begin(), player.hand.end());
                for (auto cid : unique_hand) {
                    if (CARD_DEFS[cid].is_treasure()) {
                        out[DOM_SELECT_OFFSET + cid] = 1.0f;
                        has_treasure = true;
                    }
                }
                if (!has_treasure) out[DOM_DONE_SELECTING] = 1.0f;
            } else {
                // Remodel: any card from hand
                std::set<int8_t> unique_hand(player.hand.begin(), player.hand.end());
                for (auto cid : unique_hand)
                    out[DOM_SELECT_OFFSET + cid] = 1.0f;
                if (player.hand.empty()) out[DOM_DONE_SELECTING] = 1.0f;
            }
            break;
        }

        case DOM_PEND_ARTISAN_GAIN: {
            bool any_valid = false;
            for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
                if (s.supply[i] > 0 && CARD_DEFS[i].cost <= 5) {
                    out[DOM_GAIN_OFFSET + i] = 1.0f;
                    any_valid = true;
                }
            }
            if (!any_valid) out[DOM_DONE_SELECTING] = 1.0f;
            break;
        }

        case DOM_PEND_THRONE_SELECT: {
            bool has_action = false;
            std::set<int8_t> unique_hand(player.hand.begin(), player.hand.end());
            for (auto cid : unique_hand) {
                if (CARD_DEFS[cid].is_action()) {
                    out[DOM_SELECT_OFFSET + cid] = 1.0f;
                    has_action = true;
                }
            }
            if (!has_action) out[DOM_DONE_SELECTING] = 1.0f;
            break;
        }

        default:
            break;
        }
        return;
    }

    // Non-pending phases
    if (s.phase == DOM_PHASE_ACTION) {
        out[DOM_END_ACTIONS] = 1.0f;
        if (s.actions_remaining > 0) {
            std::set<int8_t> unique_hand(player.hand.begin(), player.hand.end());
            for (auto cid : unique_hand) {
                if (CARD_DEFS[cid].is_action())
                    out[DOM_PLAY_OFFSET + cid] = 1.0f;
            }
        }
    } else if (s.phase == DOM_PHASE_BUY) {
        // Treasures auto-played on phase transition — buy phase is purely "what to buy?"
        bool can_buy_something = false;
        if (s.buys_remaining > 0) {
            for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
                if (i == CARD_CURSE) continue;  // Never allow voluntary Curse purchase
                if (s.supply[i] > 0 && CARD_DEFS[i].cost <= s.coins) {
                    out[DOM_BUY_OFFSET + i] = 1.0f;
                    can_buy_something = true;
                }
            }
        }
        out[DOM_END_BUYS] = 1.0f;  // Always allow pass — bot must learn when NOT to buy
    }
}

std::unique_ptr<GameState> DominionGame::get_next_state(const GameState& state_base, int action) const {
    auto new_state = state_base.copy();
    auto& s = static_cast<DominionState&>(*new_state);
    s.total_moves[s.current_player_]++;

    if (s.pending.active()) {
        resolve_pending(s, action);
    } else if (action == DOM_END_ACTIONS) {
        s.phase = DOM_PHASE_BUY;
        // Auto-play all treasures from hand
        auto& ap = s.players[s.current_player_];
        for (int i = (int)ap.hand.size() - 1; i >= 0; i--) {
            int8_t cid = ap.hand[i];
            if (CARD_DEFS[cid].is_treasure()) {
                ap.hand.erase(ap.hand.begin() + i);
                ap.in_play.push_back(cid);
                s.coins += CARD_DEFS[cid].coins;
                if (cid == CARD_SILVER && !s.merchant_silver_triggered) {
                    s.coins += s.merchant_silver_bonus;
                    s.merchant_silver_triggered = true;
                }
            }
        }
        // Track purchasing power at buy phase entry
        s.total_coins_at_buy[s.current_player_] += s.coins;
        s.buy_phase_entries[s.current_player_]++;
    } else if (action == DOM_END_BUYS) {
        s.coins_wasted[s.current_player_] += s.coins;
        do_cleanup(s);
    } else if (action >= DOM_PLAY_OFFSET && action < DOM_BUY_OFFSET) {
        int8_t card_id = static_cast<int8_t>(action - DOM_PLAY_OFFSET);
        play_card(s, card_id);
    } else if (action >= DOM_BUY_OFFSET && action < DOM_SELECT_OFFSET) {
        int8_t card_id = static_cast<int8_t>(action - DOM_BUY_OFFSET);
        buy_card(s, card_id);
    }

    // After any resolution, check Throne Room replay
    if (s.throne_remaining > 0 && !s.pending.active()
            && s.phase == DOM_PHASE_ACTION) {
        do_throne_replay(s);
    }

    return new_state;
}

void DominionGame::play_card(DominionState& s, int8_t card_id) const {
    auto& player = s.players[s.current_player_];
    const auto& card = CARD_DEFS[card_id];

    player.remove_from_hand(card_id);
    player.in_play.push_back(card_id);

    if (card.is_treasure()) {
        s.coins += card.coins;
        // Merchant bonus: +1 coin first Silver this turn
        if (card_id == CARD_SILVER && !s.merchant_silver_triggered) {
            s.coins += s.merchant_silver_bonus;
            s.merchant_silver_triggered = true;
        }
        return;
    }

    // Action card
    s.action_plays[s.current_player_]++;
    if (!s.action_played_this_turn) {
        s.turns_action_played[s.current_player_]++;
        s.action_played_this_turn = true;
    }
    if (s.phase == DOM_PHASE_ACTION) {
        s.actions_remaining -= 1;
    }

    s.actions_remaining += card.plus_actions;
    s.buys_remaining += card.plus_buys;
    s.coins += card.plus_coins;
    if (card.plus_cards > 0) {
        size_t hash = s.compute_hash();
        player.draw_deterministic(card.plus_cards, hash);
    }

    if (card.has_effect) {
        handle_effect(s, card_id);
    }
}

void DominionGame::handle_effect(DominionState& s, int8_t card_id) const {
    auto& player = s.players[s.active_player_];
    auto& opp = s.players[1 - s.active_player_];

    switch (card_id) {
    case CARD_CELLAR:
        if (!player.hand.empty()) {
            s.pending.type = DOM_PEND_SELECT_HAND;
            s.pending.card_id = CARD_CELLAR;
            s.pending.min_select = 0;
            s.pending.max_select = static_cast<int8_t>(player.hand.size());
            s.pending.selected_count = 0;
            s.phase = DOM_PHASE_SELECT;
        }
        break;

    case CARD_CHAPEL:
        if (!player.hand.empty()) {
            s.pending.type = DOM_PEND_SELECT_HAND;
            s.pending.card_id = CARD_CHAPEL;
            s.pending.min_select = 0;
            s.pending.max_select = static_cast<int8_t>(std::min(4, static_cast<int>(player.hand.size())));
            s.pending.selected_count = 0;
            s.phase = DOM_PHASE_SELECT;
        }
        break;

    case CARD_HARBINGER:
        if (!player.discard.empty()) {
            s.pending.type = DOM_PEND_HARBINGER_TOPDECK;
            s.pending.card_id = CARD_HARBINGER;
            s.phase = DOM_PHASE_SELECT;
        }
        break;

    case CARD_MERCHANT:
        s.merchant_silver_bonus += 1;
        break;

    case CARD_VASSAL: {
        int8_t revealed[1];
        size_t hash = s.compute_hash();
        int n = player.draw_to_list_deterministic(1, revealed, hash);
        if (n > 0) {
            player.discard.push_back(revealed[0]);
            if (CARD_DEFS[revealed[0]].is_action()) {
                s.pending.type = DOM_PEND_VASSAL_PLAY;
                s.pending.card_id = CARD_VASSAL;
                s.pending.revealed[0] = revealed[0];
                s.pending.revealed_count = 1;
                s.phase = DOM_PHASE_SELECT;
            }
        }
        break;
    }

    case CARD_WORKSHOP:
        s.pending.type = DOM_PEND_GAIN;
        s.pending.card_id = CARD_WORKSHOP;
        s.pending.trashed_card_cost = 4;
        s.phase = DOM_PHASE_SELECT;
        break;

    case CARD_BUREAUCRAT:
        if (s.supply[CARD_SILVER] > 0) {
            s.supply[CARD_SILVER]--;
            player.deck.push_back(CARD_SILVER);
        }
        start_attack(s, CARD_BUREAUCRAT, DOM_PEND_BUREAUCRAT_TOPDECK);
        break;

    case CARD_MILITIA:
        start_attack(s, CARD_MILITIA, DOM_PEND_SELECT_HAND, 0, 0);
        break;

    case CARD_MONEYLENDER:
        if (player.has_in_hand(CARD_COPPER)) {
            s.pending.type = DOM_PEND_SELECT_TRASH_HAND;
            s.pending.card_id = CARD_MONEYLENDER;
            s.phase = DOM_PHASE_SELECT;
        }
        break;

    case CARD_POACHER: {
        int empty_piles = 0;
        for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
            if (s.supply[i] == 0) {
                bool in_game = (i < 7);
                if (!in_game) {
                    for (int k = 0; k < s.num_kingdom; k++) {
                        if (s.kingdom_cards[k] == i) { in_game = true; break; }
                    }
                }
                if (in_game) empty_piles++;
            }
        }
        if (empty_piles > 0 && !player.hand.empty()) {
            int8_t discard_count = static_cast<int8_t>(std::min(empty_piles, static_cast<int>(player.hand.size())));
            s.pending.type = DOM_PEND_SELECT_HAND;
            s.pending.card_id = CARD_POACHER;
            s.pending.min_select = discard_count;
            s.pending.max_select = discard_count;
            s.pending.selected_count = 0;
            s.phase = DOM_PHASE_SELECT;
        }
        break;
    }

    case CARD_REMODEL:
        if (!player.hand.empty()) {
            s.pending.type = DOM_PEND_SELECT_TRASH_HAND;
            s.pending.card_id = CARD_REMODEL;
            s.phase = DOM_PHASE_SELECT;
        }
        break;

    case CARD_THRONE_ROOM: {
        bool has_action = false;
        for (auto c : player.hand) {
            if (CARD_DEFS[c].is_action()) { has_action = true; break; }
        }
        if (has_action) {
            s.pending.type = DOM_PEND_THRONE_SELECT;
            s.pending.card_id = CARD_THRONE_ROOM;
            s.phase = DOM_PHASE_SELECT;
        }
        break;
    }

    case CARD_ARTISAN:
        s.pending.type = DOM_PEND_ARTISAN_GAIN;
        s.pending.card_id = CARD_ARTISAN;
        s.phase = DOM_PHASE_SELECT;
        break;

    case CARD_BANDIT:
        if (s.supply[CARD_GOLD] > 0) {
            s.supply[CARD_GOLD]--;
            player.discard.push_back(CARD_GOLD);
        }
        start_attack(s, CARD_BANDIT, DOM_PEND_BANDIT_TRASH);
        break;

    case CARD_COUNCIL_ROOM: {
        size_t hash = s.compute_hash();
        opp.draw_deterministic(1, hash);
        break;
    }

    case CARD_LIBRARY:
        library_step(s);
        break;

    case CARD_MINE: {
        bool has_treasure = false;
        for (auto c : player.hand) {
            if (CARD_DEFS[c].is_treasure()) { has_treasure = true; break; }
        }
        if (has_treasure) {
            s.pending.type = DOM_PEND_SELECT_TRASH_HAND;
            s.pending.card_id = CARD_MINE;
            s.phase = DOM_PHASE_SELECT;
        }
        break;
    }

    case CARD_SENTRY: {
        int8_t revealed[2];
        size_t hash = s.compute_hash();
        int n = player.draw_to_list_deterministic(2, revealed, hash);
        if (n > 0) {
            s.pending.type = DOM_PEND_SENTRY_CARD;
            s.pending.card_id = CARD_SENTRY;
            s.pending.revealed_count = static_cast<int8_t>(n);
            for (int i = 0; i < n; i++) s.pending.revealed[i] = revealed[i];
            s.phase = DOM_PHASE_SELECT;
        }
        break;
    }

    case CARD_WITCH:
        start_attack(s, CARD_WITCH, DOM_PEND_NONE);
        break;

    default:
        break;
    }
}

void DominionGame::start_attack(DominionState& s, int8_t card_id, DomPendingType dtype,
                                 int8_t min_sel, int8_t max_sel) const {
    int opp_idx = 1 - s.active_player_;
    auto& opp = s.players[opp_idx];

    if (opp.has_in_hand(CARD_MOAT)) {
        s.save_phase = s.phase;
        s.phase = DOM_PHASE_REACT;
        s.current_player_ = static_cast<int8_t>(opp_idx);
        s.pending.type = DOM_PEND_REACT;
        s.pending.card_id = card_id;
        s.pending.target_player = static_cast<int8_t>(opp_idx);
        s.pending.min_select = min_sel;
        s.pending.max_select = max_sel;
    } else {
        resolve_attack(s, card_id, opp_idx, min_sel, max_sel);
    }
}

void DominionGame::resolve_attack(DominionState& s, int8_t card_id, int opp_idx,
                                    int8_t min_sel, int8_t max_sel) const {
    auto& opp = s.players[opp_idx];

    switch (card_id) {
    case CARD_MILITIA:
        if (static_cast<int>(opp.hand.size()) > 3) {
            int8_t discard_count = static_cast<int8_t>(opp.hand.size() - 3);
            s.current_player_ = static_cast<int8_t>(opp_idx);
            s.pending.type = DOM_PEND_SELECT_HAND;
            s.pending.card_id = CARD_MILITIA;
            s.pending.min_select = discard_count;
            s.pending.max_select = discard_count;
            s.pending.selected_count = 0;
            s.pending.target_player = static_cast<int8_t>(opp_idx);
            s.phase = DOM_PHASE_SELECT;
        }
        break;

    case CARD_BUREAUCRAT: {
        bool has_victory = false;
        for (auto c : opp.hand) {
            if (CARD_DEFS[c].is_victory()) { has_victory = true; break; }
        }
        if (has_victory) {
            s.current_player_ = static_cast<int8_t>(opp_idx);
            s.pending.type = DOM_PEND_BUREAUCRAT_TOPDECK;
            s.pending.card_id = CARD_BUREAUCRAT;
            s.pending.target_player = static_cast<int8_t>(opp_idx);
            s.phase = DOM_PHASE_SELECT;
        }
        break;
    }

    case CARD_BANDIT: {
        int8_t revealed[2];
        size_t hash = s.compute_hash();
        int n = opp.draw_to_list_deterministic(2, revealed, hash);

        int trashable_count = 0;
        int8_t trashable[2] = {-1, -1};
        int non_trashable_count = 0;
        int8_t non_trashable[2] = {-1, -1};

        for (int i = 0; i < n; i++) {
            if (revealed[i] != CARD_COPPER && CARD_DEFS[revealed[i]].is_treasure()) {
                trashable[trashable_count++] = revealed[i];
            } else {
                non_trashable[non_trashable_count++] = revealed[i];
            }
        }

        if (trashable_count == 1) {
            s.trash.push_back(trashable[0]);
            for (int i = 0; i < non_trashable_count; i++)
                opp.discard.push_back(non_trashable[i]);
        } else if (trashable_count > 1) {
            s.current_player_ = static_cast<int8_t>(opp_idx);
            s.pending.type = DOM_PEND_BANDIT_TRASH;
            s.pending.card_id = CARD_BANDIT;
            s.pending.bandit_revealed_count = static_cast<int8_t>(n);
            for (int i = 0; i < n; i++) s.pending.bandit_revealed[i] = revealed[i];
            s.pending.target_player = static_cast<int8_t>(opp_idx);
            s.phase = DOM_PHASE_SELECT;
        } else {
            for (int i = 0; i < n; i++)
                opp.discard.push_back(revealed[i]);
        }
        break;
    }

    case CARD_WITCH:
        if (s.supply[CARD_CURSE] > 0) {
            s.supply[CARD_CURSE]--;
            opp.discard.push_back(CARD_CURSE);
            s.curse_buys[opp_idx]++;  // Track curses gained (from Witch)
        }
        break;

    default:
        break;
    }
}

void DominionGame::resolve_pending(DominionState& s, int action) const {
    auto& p = s.pending;
    auto& player = s.players[s.current_player_];

    switch (p.type) {
    case DOM_PEND_REACT:
        if (action == DOM_REVEAL_MOAT) {
            s.current_player_ = s.active_player_;
            s.pending = DomPending{};
            s.phase = s.save_phase;
        } else if (action == DOM_DECLINE_REACTION) {
            int opp_idx = p.target_player;
            int8_t card_id = p.card_id;
            int8_t min_sel = p.min_select;
            int8_t max_sel = p.max_select;
            s.pending = DomPending{};
            s.phase = s.save_phase;
            resolve_attack(s, card_id, opp_idx, min_sel, max_sel);
        }
        break;

    case DOM_PEND_SELECT_HAND:
        if (action == DOM_DONE_SELECTING) {
            if (p.selected_count == 0) s.done_selecting_empty[s.current_player_]++;
            finish_select_hand(s);
        } else if (action >= DOM_SELECT_OFFSET && action < DOM_GAIN_OFFSET) {
            int8_t card_id = static_cast<int8_t>(action - DOM_SELECT_OFFSET);
            if (p.card_id == CARD_CELLAR) {
                player.remove_from_hand(card_id);
                player.discard.push_back(card_id);
                p.selected_count++;
                s.cards_discarded_cellar[s.current_player_]++;
                if (p.selected_count >= p.max_select)
                    finish_select_hand(s);
            } else if (p.card_id == CARD_CHAPEL) {
                player.remove_from_hand(card_id);
                s.trash.push_back(card_id);
                p.selected_count++;
                s.cards_trashed[s.current_player_]++;
                if (p.selected_count >= p.max_select)
                    finish_select_hand(s);
            } else if (p.card_id == CARD_MILITIA) {
                player.remove_from_hand(card_id);
                player.discard.push_back(card_id);
                p.selected_count++;
                if (p.selected_count >= p.min_select)
                    finish_select_hand(s);
            } else if (p.card_id == CARD_POACHER) {
                player.remove_from_hand(card_id);
                player.discard.push_back(card_id);
                p.selected_count++;
                if (p.selected_count >= p.min_select)
                    finish_select_hand(s);
            }
        }
        break;

    case DOM_PEND_SELECT_TRASH_HAND:
        if (action == DOM_DONE_SELECTING) {
            if (p.selected_count == 0) s.done_selecting_empty[s.current_player_]++;
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        } else if (action >= DOM_SELECT_OFFSET && action < DOM_GAIN_OFFSET) {
            int8_t card_id = static_cast<int8_t>(action - DOM_SELECT_OFFSET);
            player.remove_from_hand(card_id);
            s.trash.push_back(card_id);
            s.cards_trashed[s.current_player_]++;
            int8_t trashed_cost = CARD_DEFS[card_id].cost;

            if (p.card_id == CARD_MONEYLENDER) {
                s.coins += 3;
                s.pending = DomPending{};
                s.phase = DOM_PHASE_ACTION;
            } else if (p.card_id == CARD_REMODEL) {
                s.pending = DomPending{};
                s.pending.type = DOM_PEND_REMODEL_GAIN;
                s.pending.card_id = CARD_REMODEL;
                s.pending.trashed_card_cost = trashed_cost;
            } else if (p.card_id == CARD_MINE) {
                s.pending = DomPending{};
                s.pending.type = DOM_PEND_MINE_GAIN;
                s.pending.card_id = CARD_MINE;
                s.pending.trashed_card_cost = trashed_cost;
            }
        }
        break;

    case DOM_PEND_GAIN:
    case DOM_PEND_REMODEL_GAIN:
        if (action == DOM_DONE_SELECTING) {
            // No qualifying cards in supply — gain fizzles
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        } else if (action >= DOM_GAIN_OFFSET && action < DOM_REVEAL_MOAT) {
            int8_t card_id = static_cast<int8_t>(action - DOM_GAIN_OFFSET);
            s.supply[card_id]--;
            player.discard.push_back(card_id);
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
            check_game_end(s);
        }
        break;

    case DOM_PEND_MINE_GAIN:
        if (action == DOM_DONE_SELECTING) {
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        } else if (action >= DOM_GAIN_OFFSET && action < DOM_REVEAL_MOAT) {
            int8_t card_id = static_cast<int8_t>(action - DOM_GAIN_OFFSET);
            s.supply[card_id]--;
            player.hand.push_back(card_id);  // Mine gains to hand
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
            check_game_end(s);
        }
        break;

    case DOM_PEND_ARTISAN_GAIN:
        if (action == DOM_DONE_SELECTING) {
            // No qualifying cards — skip gain AND topdeck
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        } else if (action >= DOM_GAIN_OFFSET && action < DOM_REVEAL_MOAT) {
            int8_t card_id = static_cast<int8_t>(action - DOM_GAIN_OFFSET);
            s.supply[card_id]--;
            player.hand.push_back(card_id);  // Artisan gains to hand
            s.pending = DomPending{};
            s.pending.type = DOM_PEND_ARTISAN_TOPDECK;
            s.pending.card_id = CARD_ARTISAN;
            check_game_end(s);
        }
        break;

    case DOM_PEND_ARTISAN_TOPDECK:
        if (action >= DOM_SELECT_OFFSET && action < DOM_GAIN_OFFSET) {
            int8_t card_id = static_cast<int8_t>(action - DOM_SELECT_OFFSET);
            player.remove_from_hand(card_id);
            player.deck.push_back(card_id);
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        }
        break;

    case DOM_PEND_HARBINGER_TOPDECK:
        if (action == DOM_DONE_SELECTING) {
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        } else if (action >= DOM_SELECT_OFFSET && action < DOM_GAIN_OFFSET) {
            int8_t card_id = static_cast<int8_t>(action - DOM_SELECT_OFFSET);
            // Remove from discard
            for (auto it = player.discard.begin(); it != player.discard.end(); ++it) {
                if (*it == card_id) { player.discard.erase(it); break; }
            }
            player.deck.push_back(card_id);
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        }
        break;

    case DOM_PEND_VASSAL_PLAY: {
        int8_t rcid = p.revealed[0];
        if (action == DOM_ACCEPT) {
            // Play from discard
            for (auto it = player.discard.begin(); it != player.discard.end(); ++it) {
                if (*it == rcid) { player.discard.erase(it); break; }
            }
            player.in_play.push_back(rcid);
            const auto& card = CARD_DEFS[rcid];
            s.actions_remaining += card.plus_actions;
            s.buys_remaining += card.plus_buys;
            s.coins += card.plus_coins;
            if (card.plus_cards > 0) {
                size_t hash = s.compute_hash();
                player.draw_deterministic(card.plus_cards, hash);
            }
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
            if (card.has_effect) {
                handle_effect(s, rcid);
            }
        } else if (action == DOM_DECLINE) {
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        }
        break;
    }

    case DOM_PEND_LIBRARY_DRAW:
        if (action == DOM_ACCEPT) {
            // Keep card (already in hand)
            s.pending = DomPending{};
            library_step(s);
        } else if (action == DOM_DECLINE) {
            // Set aside action
            if (p.revealed_count > 0 && p.revealed[0] >= 0) {
                int8_t cid = p.revealed[0];
                player.remove_from_hand(cid);
                player.discard.push_back(cid);
            }
            s.pending = DomPending{};
            library_step(s);
        }
        break;

    case DOM_PEND_SENTRY_CARD: {
        int8_t rcid = p.revealed[0];
        // Shift remaining revealed cards
        int8_t remaining[4] = {-1, -1, -1, -1};
        int8_t remaining_count = 0;
        for (int i = 1; i < p.revealed_count; i++) {
            remaining[remaining_count++] = p.revealed[i];
        }

        if (action >= DOM_SELECT_OFFSET && action < DOM_GAIN_OFFSET) {
            // Trash
            s.trash.push_back(rcid);
        } else if (action == DOM_ACCEPT) {
            // Discard
            player.discard.push_back(rcid);
        } else if (action == DOM_DECLINE) {
            // Put back on deck
            player.deck.push_back(rcid);
        }

        if (remaining_count > 0) {
            s.pending.revealed_count = remaining_count;
            for (int i = 0; i < 4; i++) s.pending.revealed[i] = remaining[i];
        } else {
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        }
        break;
    }

    case DOM_PEND_BANDIT_TRASH:
        if (action >= DOM_SELECT_OFFSET && action < DOM_GAIN_OFFSET) {
            int8_t card_id = static_cast<int8_t>(action - DOM_SELECT_OFFSET);
            s.trash.push_back(card_id);
            // Discard the other revealed cards
            bool removed_first = false;
            for (int i = 0; i < p.bandit_revealed_count; i++) {
                if (p.bandit_revealed[i] == card_id && !removed_first) {
                    removed_first = true;
                    continue;
                }
                player.discard.push_back(p.bandit_revealed[i]);
            }
            s.current_player_ = s.active_player_;
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        }
        break;

    case DOM_PEND_BUREAUCRAT_TOPDECK:
        if (action == DOM_DONE_SELECTING) {
            s.current_player_ = s.active_player_;
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        } else if (action >= DOM_SELECT_OFFSET && action < DOM_GAIN_OFFSET) {
            int8_t card_id = static_cast<int8_t>(action - DOM_SELECT_OFFSET);
            player.remove_from_hand(card_id);
            player.deck.push_back(card_id);
            s.current_player_ = s.active_player_;
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        }
        break;

    case DOM_PEND_THRONE_SELECT:
        if (action == DOM_DONE_SELECTING) {
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
        } else if (action >= DOM_SELECT_OFFSET && action < DOM_GAIN_OFFSET) {
            int8_t card_id = static_cast<int8_t>(action - DOM_SELECT_OFFSET);
            player.remove_from_hand(card_id);
            player.in_play.push_back(card_id);
            const auto& card = CARD_DEFS[card_id];

            s.throne_card = card_id;
            s.throne_remaining = 1;

            // First play
            s.actions_remaining += card.plus_actions;
            s.buys_remaining += card.plus_buys;
            s.coins += card.plus_coins;
            if (card.plus_cards > 0) {
                size_t hash = s.compute_hash();
                player.draw_deterministic(card.plus_cards, hash);
            }
            s.pending = DomPending{};
            s.phase = DOM_PHASE_ACTION;
            if (card.has_effect) {
                handle_effect(s, card_id);
            }
            // If no pending from effect, trigger second play
            if (!s.pending.active()) {
                do_throne_replay(s);
            }
        }
        break;

    default:
        break;
    }
}

void DominionGame::finish_select_hand(DominionState& s) const {
    auto& p = s.pending;
    auto& player = s.players[s.current_player_];

    if (p.card_id == CARD_CELLAR) {
        size_t hash = s.compute_hash();
        player.draw_deterministic(p.selected_count, hash);
        s.pending = DomPending{};
        s.phase = DOM_PHASE_ACTION;
    } else if (p.card_id == CARD_CHAPEL) {
        s.pending = DomPending{};
        s.phase = DOM_PHASE_ACTION;
    } else if (p.card_id == CARD_MILITIA) {
        s.current_player_ = s.active_player_;
        s.pending = DomPending{};
        s.phase = DOM_PHASE_ACTION;
    } else if (p.card_id == CARD_POACHER) {
        s.pending = DomPending{};
        s.phase = DOM_PHASE_ACTION;
    }
}

void DominionGame::library_step(DominionState& s) const {
    auto& player = s.players[s.active_player_];
    if (static_cast<int>(player.hand.size()) >= 7) {
        s.pending = DomPending{};
        s.phase = DOM_PHASE_ACTION;
        return;
    }

    int8_t drawn[1];
    size_t hash = s.compute_hash();
    int n = player.draw_to_list_deterministic(1, drawn, hash);
    if (n == 0) {
        s.pending = DomPending{};
        s.phase = DOM_PHASE_ACTION;
        return;
    }

    int8_t cid = drawn[0];
    if (CARD_DEFS[cid].is_action()) {
        player.hand.push_back(cid);
        s.pending.type = DOM_PEND_LIBRARY_DRAW;
        s.pending.card_id = CARD_LIBRARY;
        s.pending.revealed[0] = cid;
        s.pending.revealed_count = 1;
        s.phase = DOM_PHASE_SELECT;
    } else {
        player.hand.push_back(cid);
        library_step(s);
    }
}

void DominionGame::do_throne_replay(DominionState& s) const {
    if (s.throne_remaining <= 0) return;
    int8_t card_id = s.throne_card;
    const auto& card = CARD_DEFS[card_id];
    auto& player = s.players[s.active_player_];

    s.throne_remaining--;
    if (s.throne_remaining == 0) {
        s.throne_card = -1;
    }

    s.actions_remaining += card.plus_actions;
    s.buys_remaining += card.plus_buys;
    s.coins += card.plus_coins;
    if (card.plus_cards > 0) {
        size_t hash = s.compute_hash();
        player.draw_deterministic(card.plus_cards, hash);
    }

    if (card.has_effect) {
        handle_effect(s, card_id);
    }
}

void DominionGame::buy_card(DominionState& s, int8_t card_id) const {
    int p = s.current_player_;
    s.coins -= CARD_DEFS[card_id].cost;
    s.buys_remaining--;
    s.supply[card_id]--;
    s.players[p].discard.push_back(card_id);

    // Track buys
    s.total_buys[p]++;
    s.card_buys[card_id][p]++;
    s.bucketed_buys[s.buy_bucket()][card_id][p]++;
    s.buy_turn_sum[card_id][p] += s.turn_number;
    if (card_id == CARD_PROVINCE) s.province_buys[p]++;
    if (card_id == CARD_DUCHY) s.duchy_buys[p]++;
    if (card_id == CARD_ESTATE) s.estate_buys[p]++;
    if (card_id == CARD_COPPER) s.copper_buys[p]++;
    if (card_id == CARD_SILVER || card_id == CARD_GOLD) s.treasure_buys[p]++;
    if (CARD_DEFS[card_id].is_action()) s.action_buys[p]++;
    if (card_id == CARD_CURSE) s.curse_buys[p]++;

    check_game_end(s);
}

void DominionGame::do_cleanup(DominionState& s) const {
    auto& player = s.players[s.active_player_];

    player.discard.insert(player.discard.end(), player.hand.begin(), player.hand.end());
    player.hand.clear();
    player.discard.insert(player.discard.end(), player.in_play.begin(), player.in_play.end());
    player.in_play.clear();

    size_t hash = s.compute_hash();
    player.draw_deterministic(5, hash);

    int8_t next = 1 - s.active_player_;
    s.active_player_ = next;
    s.current_player_ = next;
    s.phase = DOM_PHASE_ACTION;
    s.actions_remaining = 1;
    s.buys_remaining = 1;
    s.coins = 0;
    s.merchant_silver_bonus = 0;
    s.merchant_silver_triggered = false;
    s.throne_card = -1;
    s.throne_remaining = 0;
    s.pending = DomPending{};
    s.action_played_this_turn = false;

    // Track if next player's hand has any action cards
    for (int8_t card_id : s.players[next].hand) {
        if (CARD_DEFS[card_id].is_action()) {
            s.turns_with_action_in_hand[next]++;
            break;
        }
    }

    if (s.active_player_ == 0) {
        s.turn_number++;
    }

    check_game_end(s);
}

void DominionGame::check_game_end(DominionState& s) const {
    if (s.supply[CARD_PROVINCE] == 0) {
        s.game_over = true;
        s.terminated_by = DOM_TERM_PROVINCE_EMPTY;
        return;
    }
    int empty = 0;
    for (int i = 0; i < DOM_NUM_CARD_TYPES; i++) {
        if (s.supply[i] == 0 && !s.supply_disabled[i]) {
            bool in_game = (i < 7);
            if (!in_game) {
                for (int k = 0; k < s.num_kingdom; k++) {
                    if (s.kingdom_cards[k] == i) { in_game = true; break; }
                }
            }
            if (in_game) empty++;
        }
    }
    if (empty >= 3) {
        s.game_over = true;
        s.terminated_by = DOM_TERM_THREE_PILES;
        return;
    }
    if (early_terminate_decided_ && is_outcome_determined(s)) {
        s.game_over = true;
        s.terminated_by = DOM_TERM_OUTCOME_DETERMINED;
    }
}

// DEVLOG #172: A game's outcome is mathematically determined when the current
// VP lead exceeds every VP point still obtainable from the supply.
//
//   remaining_max = Province × 6 + Duchy × 3 + Estate × 1 + Gardens × 10
//
// Gardens is bounded at 10 VP per copy (would require 100-card deck to realize,
// not possible without Gardens-in-supply curriculum — so the *10 is a defensive
// upper bound when Gardens is enabled; with current disabled_basic_supply=[…,16]
// it adds 0). Trailing player cannot catch up once lead > remaining_max, so the
// game is over from a training-signal perspective; further turns only generate
// pathological replay samples (DEVLOG #172 stuck-at-50 analysis).
bool DominionGame::is_outcome_determined(const DominionState& s) const {
    int vp0 = compute_vp(s, 0);
    int vp1 = compute_vp(s, 1);
    int lead = std::abs(vp0 - vp1);
    int remaining = s.supply[CARD_PROVINCE] * 6
                  + s.supply[CARD_DUCHY]    * 3
                  + s.supply[CARD_ESTATE]   * 1
                  + s.supply[CARD_GARDENS]  * 10;
    return lead > remaining;
}

int DominionGame::compute_vp(const DominionState& s, int player) const {
    return s.players[player].count_vp(s.kingdom_cards, s.num_kingdom);
}

bool DominionGame::is_terminal(const GameState& state_base) const {
    return static_cast<const DominionState&>(state_base).game_over;
}

float DominionGame::get_reward(const GameState& state_base, int player) const {
    auto& s = static_cast<const DominionState&>(state_base);
    if (!s.game_over) return 0.0f;

    int vp0 = compute_vp(s, 0);
    int vp1 = compute_vp(s, 1);
    int margin = (player == 0) ? (vp0 - vp1) : (vp1 - vp0);

    // Score-margin reward: /5 for stronger gradient during early training
    // (typical margins are 1-3 VP at current training stage, /30 was too weak)
    float scaled = margin / 5.0f;
    float reward = std::max(-1.0f, std::min(1.0f, scaled));

    // Per-turn discount: penalizes slow Gardens accumulation vs fast Province wins.
    // At turn 40 (typical Province game): 0.82x. At turn 120 (Gardens equilibrium): 0.55x.
    // A +4VP Province win in 40 turns (0.80 * 0.82 = 0.66) beats a +6VP Gardens win
    // in 120 turns (1.00 * 0.55 = 0.55), incentivizing the bot to end games sooner.
    float discount = std::pow(0.995f, static_cast<float>(s.turn_number));
    return reward * discount;
}

int DominionGame::get_score(const GameState& state_base, int player) const {
    auto& s = static_cast<const DominionState&>(state_base);
    return compute_vp(s, player);
}


void DominionGame::randomize_hidden(GameState& state_base, std::mt19937& rng) const {
    auto& s = static_cast<DominionState&>(state_base);
    int opp = 1 - s.current_player_;

    // Pool opponent's hand + deck (both unknown to current player)
    // Discard pile and in_play are public information
    std::vector<int8_t> pool;
    pool.reserve(s.players[opp].hand.size() + s.players[opp].deck.size());
    pool.insert(pool.end(), s.players[opp].hand.begin(), s.players[opp].hand.end());
    pool.insert(pool.end(), s.players[opp].deck.begin(), s.players[opp].deck.end());

    size_t hand_sz = s.players[opp].hand.size();

    std::shuffle(pool.begin(), pool.end(), rng);

    s.players[opp].hand.assign(pool.begin(), pool.begin() + hand_sz);
    s.players[opp].deck.assign(pool.begin() + hand_sz, pool.end());
}
