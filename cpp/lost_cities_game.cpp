#include "lost_cities_game.h"
#include <cstring>
#include <algorithm>
#include <cmath>

// --- LostCitiesState ---

std::unique_ptr<GameState> LostCitiesState::copy() const {
    auto s = std::make_unique<LostCitiesState>();
    s->deck = deck;
    s->hands = hands;
    s->expeditions = expeditions;
    s->discard_piles = discard_piles;
    s->current_player_ = current_player_;
    s->game_over = game_over;
    s->turns_played = turns_played;
    s->expedition_plays = expedition_plays;
    s->color_discards = color_discards;
    s->discard_pile_draws = discard_pile_draws;
    s->total_moves = total_moves;
    return s;
}

void LostCitiesState::to_tensor(std::vector<float>& out) const {
    int total = LC_TENSOR_CHANNELS * 8 * 8;
    out.assign(total, 0.0f);

    auto set_channel = [&out](int ch, float val) {
        int offset = ch * 64;
        for (int i = 0; i < 64; i++) out[offset + i] = val;
    };

    // Ch 0-4: My hand count per color (/8)
    // Ch 5-9: My hand wager count per color (/3)
    for (auto& card : hands[0]) {
        set_channel(card.color, out[card.color * 64] + 1.0f / LC_HAND_SIZE);
        if (card.value == 0) {
            set_channel(5 + card.color, out[(5 + card.color) * 64] + 1.0f / 3.0f);
        }
    }

    // Ch 10-14: My expedition top value per color (/10)
    // Ch 15-19: My expedition length per color (/12)
    // Ch 20-24: My expedition wager count per color (/3)
    for (int c = 0; c < LC_NUM_COLORS; c++) {
        auto& exp = expeditions[0][c];
        if (!exp.empty()) {
            int max_val = 0;
            int wagers = 0;
            for (auto& card : exp) {
                if (card.value > max_val) max_val = card.value;
                if (card.value == 0) wagers++;
            }
            set_channel(10 + c, max_val / 10.0f);
            set_channel(15 + c, static_cast<float>(exp.size()) / LC_CARDS_PER_COLOR);
            set_channel(20 + c, wagers / 3.0f);
        }
    }

    // Ch 25-29: Opp expedition top value per color (/10)
    // Ch 30-34: Opp expedition length per color (/12)
    // Ch 35-39: Opp expedition wager count per color (/3)
    for (int c = 0; c < LC_NUM_COLORS; c++) {
        auto& exp = expeditions[1][c];
        if (!exp.empty()) {
            int max_val = 0;
            int wagers = 0;
            for (auto& card : exp) {
                if (card.value > max_val) max_val = card.value;
                if (card.value == 0) wagers++;
            }
            set_channel(25 + c, max_val / 10.0f);
            set_channel(30 + c, static_cast<float>(exp.size()) / LC_CARDS_PER_COLOR);
            set_channel(35 + c, wagers / 3.0f);
        }
    }

    // Ch 40-44: Discard pile top value per color (/10)
    for (int c = 0; c < LC_NUM_COLORS; c++) {
        if (!discard_piles[c].empty()) {
            set_channel(40 + c, discard_piles[c].back().value / 10.0f);
        }
    }

    // Ch 45: Deck size / 44
    set_channel(45, static_cast<float>(deck.size()) / LC_INITIAL_DECK);
    // Ch 46: Opp hand size / 8
    set_channel(46, static_cast<float>(hands[1].size()) / LC_HAND_SIZE);
    // Ch 47: My score / 200 + 0.5
    float my_score = std::max(0.0f, std::min(1.0f, compute_score(0) / 200.0f + 0.5f));
    set_channel(47, my_score);
    // Ch 48: Opp score / 200 + 0.5
    float opp_score = std::max(0.0f, std::min(1.0f, compute_score(1) / 200.0f + 0.5f));
    set_channel(48, opp_score);
    // Ch 49: Game progress
    set_channel(49, static_cast<float>(turns_played) / LC_MAX_TURNS);

    // Ch 50-57: Hand position 0-7 card color ((color+1)/6, 0=empty)
    // Ch 58-65: Hand position 0-7 card value ((value+1)/11, 0=empty)
    for (int i = 0; i < static_cast<int>(hands[0].size()); i++) {
        set_channel(50 + i, (hands[0][i].color + 1) / 6.0f);
        set_channel(58 + i, (hands[0][i].value + 1) / 11.0f);
    }

    // Count known cards per color (shared by belief + unseen channels)
    int known[LC_NUM_COLORS] = {};
    for (auto& card : hands[0]) known[card.color]++;
    for (int p = 0; p < 2; p++)
        for (int c = 0; c < LC_NUM_COLORS; c++)
            known[c] += static_cast<int>(expeditions[p][c].size());
    for (int c = 0; c < LC_NUM_COLORS; c++)
        known[c] += static_cast<int>(discard_piles[c].size());

    int total_unseen = 0;
    int remaining[LC_NUM_COLORS];
    for (int c = 0; c < LC_NUM_COLORS; c++) {
        remaining[c] = LC_CARDS_PER_COLOR - known[c];
        if (remaining[c] < 0) remaining[c] = 0;
        total_unseen += remaining[c];
    }

    // Ch 66-70: Belief channels — P(opp has ≥1 of color c)
    {
        int opp_hand_sz = static_cast<int>(hands[1].size());
        for (int c = 0; c < LC_NUM_COLORS; c++) {
            float belief = 0.0f;
            if (remaining[c] > 0 && total_unseen > 0 && opp_hand_sz > 0) {
                float p_none = 1.0f;
                int N = total_unseen;
                int k = remaining[c];
                for (int i = 0; i < opp_hand_sz && (N - i) > 0; i++) {
                    p_none *= static_cast<float>(N - k - i) / static_cast<float>(N - i);
                }
                belief = 1.0f - p_none;
            }
            set_channel(66 + c, belief);
        }
    }

    // Ch 71-80: Expedition card value sum per color (my 71-75, opp 76-80)
    // Ch 81-90: Expedition net score per color (my 81-85, opp 86-90)
    for (int p = 0; p < 2; p++) {
        for (int c = 0; c < LC_NUM_COLORS; c++) {
            auto& exp = expeditions[p][c];
            if (exp.empty()) continue;
            int card_sum = 0;
            int wagers = 0;
            for (auto& card : exp) {
                if (card.value == 0) wagers++;
                else card_sum += card.value;
            }
            set_channel(71 + p * 5 + c, card_sum / 54.0f);
            int net = (card_sum - 20) * (1 + wagers);
            if (static_cast<int>(exp.size()) >= 8) net += 20;
            set_channel(81 + p * 5 + c, std::max(0.0f, std::min(1.0f, net / 100.0f + 0.5f)));
        }
    }

    // Ch 91-95: Unseen cards per color (remaining / 12)
    for (int c = 0; c < LC_NUM_COLORS; c++) {
        set_channel(91 + c, remaining[c] / static_cast<float>(LC_CARDS_PER_COLOR));
    }
}

std::unique_ptr<GameState> LostCitiesState::get_canonical() const {
    auto s = std::make_unique<LostCitiesState>();
    *s = *this;

    if (current_player_ == 0) return s;

    std::swap(s->hands[0], s->hands[1]);
    std::swap(s->expeditions[0], s->expeditions[1]);
    std::swap(s->expedition_plays[0], s->expedition_plays[1]);
    std::swap(s->color_discards[0], s->color_discards[1]);
    std::swap(s->discard_pile_draws[0], s->discard_pile_draws[1]);
    std::swap(s->total_moves[0], s->total_moves[1]);
    s->current_player_ = 0;
    return s;
}

int LostCitiesState::compute_score(int player) const {
    int total = 0;
    for (int c = 0; c < LC_NUM_COLORS; c++) {
        auto& exp = expeditions[player][c];
        if (exp.empty()) continue;
        int wagers = 0;
        int card_sum = 0;
        for (auto& card : exp) {
            if (card.value == 0) wagers++;
            else card_sum += card.value;
        }
        total += (card_sum - 20) * (1 + wagers);
        if (static_cast<int>(exp.size()) >= 8) total += 20;
    }
    return total;
}

// --- LostCitiesGame ---

std::unique_ptr<GameState> LostCitiesGame::create_initial_state(std::mt19937& rng) {
    auto s = std::make_unique<LostCitiesState>();

    // Create deck: 3 wagers + cards 2-10 per color x 5 colors = 60
    std::vector<LCCard> cards;
    cards.reserve(LC_TOTAL_CARDS);
    for (int color = 0; color < LC_NUM_COLORS; color++) {
        for (int w = 0; w < 3; w++)
            cards.push_back({static_cast<int8_t>(color), 0});
        for (int v = 2; v <= 10; v++)
            cards.push_back({static_cast<int8_t>(color), static_cast<int8_t>(v)});
    }

    std::shuffle(cards.begin(), cards.end(), rng);

    s->hands[0] = std::vector<LCCard>(cards.begin(), cards.begin() + LC_HAND_SIZE);
    s->hands[1] = std::vector<LCCard>(cards.begin() + LC_HAND_SIZE, cards.begin() + 2 * LC_HAND_SIZE);
    s->deck = std::vector<LCCard>(cards.begin() + 2 * LC_HAND_SIZE, cards.end());

    std::sort(s->hands[0].begin(), s->hands[0].end());
    std::sort(s->hands[1].begin(), s->hands[1].end());

    s->current_player_ = 0;
    return s;
}

bool LostCitiesGame::can_play_on_expedition(const LCCard& card, const std::vector<LCCard>& expedition) {
    if (expedition.empty()) return true;
    int last_value = expedition.back().value;
    if (last_value == 0) return true;  // Any card OK after wagers
    return card.value > last_value;     // Must be strictly ascending
}

void LostCitiesGame::get_valid_moves(const GameState& state_base, std::vector<float>& out) const {
    out.assign(LC_NUM_ACTIONS, 0.0f);
    auto& s = static_cast<const LostCitiesState&>(state_base);

    if (s.game_over) return;

    auto& hand = s.hands[s.current_player_];

    for (int hand_pos = 0; hand_pos < static_cast<int>(hand.size()); hand_pos++) {
        auto& card = hand[hand_pos];

        for (int dest = 0; dest < 2; dest++) {
            // Check play legality
            if (dest == 0) {
                if (!can_play_on_expedition(card, s.expeditions[s.current_player_][card.color]))
                    continue;
            }

            for (int draw_src = 0; draw_src < 6; draw_src++) {
                // Can't draw from pile you just discarded to
                if (dest == 1 && draw_src == card.color + 1) continue;

                // Check draw source availability
                if (draw_src == 0) {
                    if (s.deck.empty()) continue;
                } else {
                    int pile_color = draw_src - 1;
                    if (s.discard_piles[pile_color].empty()) continue;
                }

                int action = hand_pos * 12 + dest * 6 + draw_src;
                out[action] = 1.0f;
            }
        }
    }
}

std::unique_ptr<GameState> LostCitiesGame::get_next_state(const GameState& state_base, int action) const {
    auto new_state_ptr = state_base.copy();
    auto& s = static_cast<LostCitiesState&>(*new_state_ptr);
    int player = s.current_player_;

    // Decode action
    int hand_pos = action / 12;
    int dest = (action % 12) / 6;
    int draw_src = action % 6;

    LCCard card = s.hands[player][hand_pos];
    s.hands[player].erase(s.hands[player].begin() + hand_pos);

    // Track behavioral accumulators
    s.total_moves[player]++;
    if (dest == 0) {
        s.expedition_plays[player][card.color]++;
        s.expeditions[player][card.color].push_back(card);
    } else {
        s.color_discards[player][card.color]++;
        s.discard_piles[card.color].push_back(card);
    }

    // Draw
    if (draw_src == 0) {
        LCCard drawn = s.deck.back();
        s.deck.pop_back();
        s.hands[player].push_back(drawn);
        if (s.deck.empty()) s.game_over = true;
    } else {
        int pile_color = draw_src - 1;
        s.discard_pile_draws[player][pile_color]++;
        LCCard drawn = s.discard_piles[pile_color].back();
        s.discard_piles[pile_color].pop_back();
        s.hands[player].push_back(drawn);
    }

    // Re-sort hand
    std::sort(s.hands[player].begin(), s.hands[player].end());

    // Advance turn
    s.turns_played++;
    if (s.turns_played >= LC_MAX_TURNS) s.game_over = true;

    s.current_player_ = 1 - player;
    return new_state_ptr;
}

void LostCitiesGame::randomize_hidden(GameState& state_base, std::mt19937& rng) const {
    auto& s = static_cast<LostCitiesState&>(state_base);
    int opp = 1 - s.current_player_;

    // Unseen from current player's perspective: opponent hand + deck
    std::vector<LCCard> unseen;
    unseen.reserve(s.hands[opp].size() + s.deck.size());
    unseen.insert(unseen.end(), s.hands[opp].begin(), s.hands[opp].end());
    unseen.insert(unseen.end(), s.deck.begin(), s.deck.end());

    size_t opp_hand_sz = s.hands[opp].size();

    // Shuffle and re-deal
    std::shuffle(unseen.begin(), unseen.end(), rng);
    s.hands[opp].assign(unseen.begin(), unseen.begin() + opp_hand_sz);
    s.deck.assign(unseen.begin() + opp_hand_sz, unseen.end());

    // Maintain sorted hand invariant
    std::sort(s.hands[opp].begin(), s.hands[opp].end());
}

bool LostCitiesGame::is_terminal(const GameState& state_base) const {
    return static_cast<const LostCitiesState&>(state_base).game_over;
}

int LostCitiesGame::get_score(const GameState& state_base, int player) const {
    auto& s = static_cast<const LostCitiesState&>(state_base);
    return s.compute_score(player);
}

float LostCitiesGame::get_reward(const GameState& state_base, int player) const {
    auto& s = static_cast<const LostCitiesState&>(state_base);
    if (!s.game_over) return 0.0f;

    int score_p = s.compute_score(player);
    int score_opp = s.compute_score(1 - player);
    int margin = score_p - score_opp;

    // Timeout = draw (0.0). In zero-sum training, -1.0 from current player's
    // perspective becomes +1.0 for the opponent — "both lose" is impossible.
    // A draw gives both players 0.0, making natural wins (±0.8-1.0) strictly
    // preferable and giving the value head actual signal to learn from.
    if (s.turns_played >= LC_MAX_TURNS) {
        return 0.0f;
    }

    // Binary win/loss + small margin tiebreaker
    // Win: [0.8, 1.0], Loss: [-1.0, -0.8], Draw: 0.0
    if (margin > 0) return 0.8f + std::min(0.2f, margin / 200.0f);
    if (margin < 0) return -0.8f + std::max(-0.2f, margin / 200.0f);
    return 0.0f;
}
