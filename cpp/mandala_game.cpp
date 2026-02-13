#include "mandala_game.h"
#include <numeric>
#include <unordered_map>
#include <cstring>

// --- MandalaState ---

std::unique_ptr<GameState> MandalaState::copy() const {
    auto s = std::make_unique<MandalaState>();
    s->deck = deck;
    s->discard = discard;
    s->mountains = mountains;
    s->fields = fields;
    s->hands = hands;
    s->rivers = rivers;
    s->cups = cups;
    s->current_player_ = current_player_;
    s->game_over = game_over;
    s->deck_reshuffled = deck_reshuffled;
    s->game_ends_next_mandala = game_ends_next_mandala;
    return s;
}

void MandalaState::to_tensor(std::vector<float>& out) const {
    int total = M_TENSOR_CHANNELS * 8 * 8;
    out.assign(total, 0.0f);

    // Helper: count cards per color, normalized by 18
    auto color_counts = [](const std::vector<int>& cards, float* dest) {
        float counts[M_NUM_COLORS] = {};
        for (int c : cards) counts[c] += 1.0f;
        for (int i = 0; i < M_NUM_COLORS; i++) dest[i] = counts[i] / 18.0f;
    };

    // Broadcast a value across 8x8 for channel ch
    auto set_channel = [&out](int ch, float val) {
        int offset = ch * 64;
        for (int i = 0; i < 64; i++) out[offset + i] = val;
    };

    // Ch 0-5: My hand
    {
        float counts[M_NUM_COLORS];
        color_counts(hands[0], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(c, counts[c]);
    }
    // Ch 6-11: Mountain 0
    {
        float counts[M_NUM_COLORS];
        color_counts(mountains[0], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(6 + c, counts[c]);
    }
    // Ch 12-17: Mountain 1
    {
        float counts[M_NUM_COLORS];
        color_counts(mountains[1], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(12 + c, counts[c]);
    }
    // Ch 18-23: My Field Mandala 0
    {
        float counts[M_NUM_COLORS];
        color_counts(fields[0][0], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(18 + c, counts[c]);
    }
    // Ch 24-29: My Field Mandala 1
    {
        float counts[M_NUM_COLORS];
        color_counts(fields[1][0], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(24 + c, counts[c]);
    }
    // Ch 30-35: Opp Field Mandala 0
    {
        float counts[M_NUM_COLORS];
        color_counts(fields[0][1], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(30 + c, counts[c]);
    }
    // Ch 36-41: Opp Field Mandala 1
    {
        float counts[M_NUM_COLORS];
        color_counts(fields[1][1], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(36 + c, counts[c]);
    }
    // Ch 42-47: My River value per color
    for (int pos = 0; pos < static_cast<int>(rivers[0].size()); pos++) {
        int color = rivers[0][pos];
        set_channel(42 + color, (pos + 1) / 6.0f);
    }
    // Ch 48-53: Opp River value per color
    for (int pos = 0; pos < static_cast<int>(rivers[1].size()); pos++) {
        int color = rivers[1][pos];
        set_channel(48 + color, (pos + 1) / 6.0f);
    }
    // Ch 54: My cup size / 20
    set_channel(54, static_cast<float>(cups[0].size()) / 20.0f);
    // Ch 55: Opp cup size / 20
    set_channel(55, static_cast<float>(cups[1].size()) / 20.0f);
    // Ch 56: Opp hand size / 8
    set_channel(56, static_cast<float>(hands[1].size()) / 8.0f);
    // Ch 57: Deck size / 108
    set_channel(57, static_cast<float>(deck.size()) / 108.0f);
    // Ch 58: Game ends next mandala flag
    set_channel(58, game_ends_next_mandala ? 1.0f : 0.0f);
}

std::unique_ptr<GameState> MandalaState::get_canonical() const {
    auto s = std::make_unique<MandalaState>();
    *s = *this;  // shallow copy (vectors are value types)

    if (current_player_ == 0) return s;

    // Swap player perspectives
    std::swap(s->hands[0], s->hands[1]);
    std::swap(s->rivers[0], s->rivers[1]);
    std::swap(s->cups[0], s->cups[1]);
    // Swap fields: for each mandala, swap player 0 and 1
    for (int m = 0; m < 2; m++) {
        std::swap(s->fields[m][0], s->fields[m][1]);
    }
    s->current_player_ = 0;
    return s;
}

std::set<int> MandalaState::get_colors_in_mandala(int mandala_idx) const {
    std::set<int> colors;
    for (int c : mountains[mandala_idx]) colors.insert(c);
    for (int c : fields[mandala_idx][0]) colors.insert(c);
    for (int c : fields[mandala_idx][1]) colors.insert(c);
    return colors;
}

bool MandalaState::is_mandala_complete(int mandala_idx) const {
    return get_colors_in_mandala(mandala_idx).size() == 6;
}

std::set<int> MandalaState::get_colors_in_river(int player) const {
    return std::set<int>(rivers[player].begin(), rivers[player].end());
}

size_t MandalaState::compute_hash() const {
    size_t h = 0;
    auto combine = [&h](size_t v) {
        h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
    };
    for (auto c : deck) combine(c);
    combine(deck.size());
    for (auto c : discard) combine(c);
    combine(discard.size());
    for (int m = 0; m < 2; m++) {
        for (auto c : mountains[m]) combine(c);
        combine(mountains[m].size() + 100);
    }
    for (int m = 0; m < 2; m++)
        for (int p = 0; p < 2; p++) {
            for (auto c : fields[m][p]) combine(c);
            combine(fields[m][p].size() + 200);
        }
    for (int p = 0; p < 2; p++) {
        for (auto c : hands[p]) combine(c);
        combine(hands[p].size() + 300);
    }
    combine(current_player_);
    combine(deck_reshuffled ? 1 : 0);
    combine(game_ends_next_mandala ? 1 : 0);
    return h;
}

// --- MandalaGame ---

std::unique_ptr<GameState> MandalaGame::create_initial_state(std::mt19937& rng) {
    auto s = std::make_unique<MandalaState>();

    // Create deck: 18 cards per color x 6 colors = 108
    s->deck.reserve(M_TOTAL_CARDS);
    for (int color = 0; color < M_NUM_COLORS; color++)
        for (int i = 0; i < M_CARDS_PER_COLOR; i++)
            s->deck.push_back(color);

    std::shuffle(s->deck.begin(), s->deck.end(), rng);

    // Deal 2 cards to each Mountain
    for (int m = 0; m < 2; m++) {
        s->mountains[m].push_back(s->deck.back()); s->deck.pop_back();
        s->mountains[m].push_back(s->deck.back()); s->deck.pop_back();
    }

    // Each player: 6 cards in hand, 2 cards in Cup
    for (int p = 0; p < 2; p++) {
        for (int i = 0; i < 6; i++) {
            s->hands[p].push_back(s->deck.back()); s->deck.pop_back();
        }
        for (int i = 0; i < 2; i++) {
            s->cups[p].push_back(s->deck.back()); s->deck.pop_back();
        }
    }

    s->current_player_ = 0;
    return s;
}

bool MandalaGame::can_play_to_mountain(const MandalaState& s, int color, int mandala) const {
    // Color cannot already be in either Field of this Mandala
    for (int c : s.fields[mandala][0]) if (c == color) return false;
    for (int c : s.fields[mandala][1]) if (c == color) return false;
    return true;
}

bool MandalaGame::can_play_to_field(const MandalaState& s, int color, int mandala, int player) const {
    // Color cannot be in Mountain or opponent's Field
    for (int c : s.mountains[mandala]) if (c == color) return false;
    int opp = 1 - player;
    for (int c : s.fields[mandala][opp]) if (c == color) return false;
    return true;
}

void MandalaGame::get_valid_moves(const GameState& state_base, std::vector<float>& out) const {
    out.assign(M_NUM_ACTIONS, 0.0f);
    auto& s = static_cast<const MandalaState&>(state_base);

    if (s.game_over) return;

    auto& hand = s.hands[s.current_player_];
    if (hand.empty()) return;

    // Count colors in hand
    std::unordered_map<int, int> hand_color_count;
    for (int c : hand) hand_color_count[c]++;

    // BUILD_MOUNTAIN: action = mandala*6 + color
    for (auto& [color, count] : hand_color_count) {
        for (int mandala = 0; mandala < 2; mandala++) {
            if (can_play_to_mountain(s, color, mandala)) {
                out[mandala * 6 + color] = 1.0f;
            }
        }
    }

    // GROW_FIELD: action = 12 + mandala*6 + color
    for (auto& [color, count] : hand_color_count) {
        // Must keep at least 1 card: can only play if count < hand size
        if (count < static_cast<int>(hand.size())) {
            for (int mandala = 0; mandala < 2; mandala++) {
                if (can_play_to_field(s, color, mandala, s.current_player_)) {
                    out[12 + mandala * 6 + color] = 1.0f;
                }
            }
        }
    }

    // DISCARD: action = 24 + color
    for (auto& [color, count] : hand_color_count) {
        out[24 + color] = 1.0f;
    }
}

std::unique_ptr<GameState> MandalaGame::get_next_state(const GameState& state_base, int action_id) const {
    auto new_state_ptr = state_base.copy();
    auto& s = static_cast<MandalaState&>(*new_state_ptr);
    int player = s.current_player_;
    int completed_mandala = -1;

    if (action_id < 12) {
        // BUILD_MOUNTAIN
        int mandala = action_id / 6;
        int color = action_id % 6;

        // Remove one card of this color from hand
        auto& hand = s.hands[player];
        for (auto it = hand.begin(); it != hand.end(); ++it) {
            if (*it == color) { hand.erase(it); break; }
        }

        s.mountains[mandala].push_back(color);

        // Draw up to 3 cards (max hand size 8)
        int cards_to_draw = std::min(3, M_MAX_HAND - static_cast<int>(s.hands[player].size()));
        for (int i = 0; i < cards_to_draw; i++) {
            check_and_reshuffle_deck(s);
            if (!s.deck.empty()) {
                s.hands[player].push_back(s.deck.back());
                s.deck.pop_back();
            }
        }

        if (s.is_mandala_complete(mandala)) completed_mandala = mandala;

    } else if (action_id < 24) {
        // GROW_FIELD
        int a = action_id - 12;
        int mandala = a / 6;
        int color = a % 6;

        // Remove all cards of this color (but keep at least 1 card total)
        auto& hand = s.hands[player];
        std::vector<int> to_play, remaining;
        for (int c : hand) {
            if (c == color) to_play.push_back(c);
            else remaining.push_back(c);
        }
        if (remaining.empty()) {
            remaining.push_back(to_play.back());
            to_play.pop_back();
        }
        s.hands[player] = remaining;
        s.fields[mandala][player].insert(s.fields[mandala][player].end(),
                                          to_play.begin(), to_play.end());

        // No drawing for GROW_FIELD
        if (s.is_mandala_complete(mandala)) completed_mandala = mandala;

    } else {
        // DISCARD: action = 24 + color
        int color = action_id - 24;

        auto& hand = s.hands[player];
        std::vector<int> to_discard, remaining;
        for (int c : hand) {
            if (c == color) to_discard.push_back(c);
            else remaining.push_back(c);
        }
        s.hands[player] = remaining;
        s.discard.insert(s.discard.end(), to_discard.begin(), to_discard.end());

        // Draw equal number
        for (int i = 0; i < static_cast<int>(to_discard.size()); i++) {
            check_and_reshuffle_deck(s);
            if (!s.deck.empty()) {
                s.hands[player].push_back(s.deck.back());
                s.deck.pop_back();
            }
        }
    }

    // Handle completed Mandala
    if (completed_mandala >= 0) {
        destroy_mandala(s, completed_mandala);

        if (should_game_end(s)) {
            s.game_over = true;
            s.current_player_ = 1 - player;
            return new_state_ptr;
        }

        // Refill mountain with 2 cards
        check_and_reshuffle_deck(s);
        if (!s.deck.empty()) {
            s.mountains[completed_mandala].push_back(s.deck.back());
            s.deck.pop_back();
        }
        check_and_reshuffle_deck(s);
        if (!s.deck.empty()) {
            s.mountains[completed_mandala].push_back(s.deck.back());
            s.deck.pop_back();
        }
    }

    // Switch player
    s.current_player_ = 1 - player;

    // End game if next player has no cards and deck is exhausted
    if (s.hands[s.current_player_].empty()) {
        check_and_reshuffle_deck(s);
        if (s.deck.empty()) {
            s.game_over = true;
        }
    }

    return new_state_ptr;
}

void MandalaGame::destroy_mandala(MandalaState& s, int mandala) const {
    // Determine who chooses first
    int fc0 = static_cast<int>(s.fields[mandala][0].size());
    int fc1 = static_cast<int>(s.fields[mandala][1].size());

    int first_player;
    if (fc0 > fc1) first_player = 0;
    else if (fc1 > fc0) first_player = 1;
    else first_player = 1 - s.current_player_;  // tie: non-current player

    // Group mountain cards by color
    std::unordered_map<int, std::vector<int>> mountain_colors;
    for (int c : s.mountains[mandala]) {
        mountain_colors[c].push_back(c);
    }

    // Sorted claiming order
    std::vector<int> colors_to_claim;
    for (auto& [color, cards] : mountain_colors) colors_to_claim.push_back(color);
    std::sort(colors_to_claim.begin(), colors_to_claim.end());

    int field_counts[2] = {fc0, fc1};
    int current_claimer = first_player;

    for (int color : colors_to_claim) {
        auto& cards = mountain_colors[color];

        if (field_counts[current_claimer] == 0) {
            // No field contribution -> discard
            s.discard.insert(s.discard.end(), cards.begin(), cards.end());
        } else {
            auto river_colors = s.get_colors_in_river(current_claimer);
            if (river_colors.count(color)) {
                // Already in river -> all to cup
                s.cups[current_claimer].insert(s.cups[current_claimer].end(),
                                               cards.begin(), cards.end());
            } else {
                // New: 1 to river, rest to cup
                if (!cards.empty()) {
                    s.rivers[current_claimer].push_back(cards[0]);
                    for (size_t i = 1; i < cards.size(); i++) {
                        s.cups[current_claimer].push_back(cards[i]);
                    }
                }
            }
        }
        current_claimer = 1 - current_claimer;
    }

    // Return field cards to bottom of deck
    auto& f0 = s.fields[mandala][0];
    auto& f1 = s.fields[mandala][1];
    std::vector<int> field_cards;
    field_cards.insert(field_cards.end(), f0.begin(), f0.end());
    field_cards.insert(field_cards.end(), f1.begin(), f1.end());
    // Insert at bottom: field_cards + deck (deck.back() = top)
    field_cards.insert(field_cards.end(), s.deck.begin(), s.deck.end());
    s.deck = field_cards;

    f0.clear();
    f1.clear();
    s.mountains[mandala].clear();
}

void MandalaGame::check_and_reshuffle_deck(MandalaState& s) const {
    if (s.deck.empty() && !s.deck_reshuffled && !s.discard.empty()) {
        s.deck = s.discard;
        s.discard.clear();
        // Deterministic shuffle based on state hash
        size_t seed = s.compute_hash() & 0x7FFFFFFF;
        std::mt19937 rng(static_cast<unsigned>(seed));
        std::shuffle(s.deck.begin(), s.deck.end(), rng);
        s.deck_reshuffled = true;
        s.game_ends_next_mandala = true;
    }
}

bool MandalaGame::should_game_end(const MandalaState& s) const {
    if (s.rivers[0].size() >= 6 || s.rivers[1].size() >= 6) return true;
    if (s.game_ends_next_mandala) return true;
    return false;
}

bool MandalaGame::is_terminal(const GameState& state_base) const {
    return static_cast<const MandalaState&>(state_base).game_over;
}

float MandalaGame::get_reward(const GameState& state_base, int player) const {
    auto& s = static_cast<const MandalaState&>(state_base);
    if (!s.game_over) return 0.0f;

    int score_p = calculate_score(s, player);
    int score_opp = calculate_score(s, 1 - player);

    if (score_p > score_opp) return 1.0f;
    if (score_p < score_opp) return -1.0f;

    // Tiebreaker: more cards in Cup wins
    int cups_p = static_cast<int>(s.cups[player].size());
    int cups_opp = static_cast<int>(s.cups[1 - player].size());
    if (cups_p > cups_opp) return 1.0f;
    if (cups_p < cups_opp) return -1.0f;
    return 0.0f;
}

int MandalaGame::calculate_score(const MandalaState& s, int player) const {
    int score = 0;
    // River: color -> position value (1-indexed)
    std::unordered_map<int, int> color_values;
    for (int pos = 0; pos < static_cast<int>(s.rivers[player].size()); pos++) {
        color_values[s.rivers[player][pos]] = pos + 1;
    }
    // Cup cards score by their river position value
    for (int c : s.cups[player]) {
        auto it = color_values.find(c);
        if (it != color_values.end()) score += it->second;
    }
    return score;
}
