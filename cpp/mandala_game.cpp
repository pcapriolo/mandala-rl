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
    s->phase = phase;
    s->claiming_mandala = claiming_mandala;
    s->claiming_colors_available = claiming_colors_available;
    s->triggering_player = triggering_player;
    s->mountain_plays = mountain_plays;
    s->field_plays = field_plays;
    s->discard_plays = discard_plays;
    s->total_moves = total_moves;
    return s;
}

void MandalaState::to_tensor(std::vector<float>& out) const {
    int total = M_TENSOR_CHANNELS * 8 * 8;
    out.assign(total, 0.0f);

    auto color_counts = [](const std::vector<int>& cards, float* dest) {
        float counts[M_NUM_COLORS] = {};
        for (int c : cards) counts[c] += 1.0f;
        for (int i = 0; i < M_NUM_COLORS; i++) dest[i] = counts[i] / 18.0f;
    };

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

    // Ch 59-64: Belief channels
    // Count all visible cards (hand, mountains, fields, rivers, discard — discard is face-up)
    int known[M_NUM_COLORS] = {};
    {
        for (int c : hands[0]) known[c]++;
        for (int m = 0; m < 2; m++) {
            for (int c : mountains[m]) known[c]++;
            for (int c : fields[m][0]) known[c]++;
            for (int c : fields[m][1]) known[c]++;
        }
        for (int c : rivers[0]) known[c]++;
        for (int c : rivers[1]) known[c]++;
        for (int c : discard) known[c]++;  // Discard is face-up public info

        int total_unseen = 0;
        int remaining[M_NUM_COLORS];
        for (int c = 0; c < M_NUM_COLORS; c++) {
            remaining[c] = M_CARDS_PER_COLOR - known[c];
            if (remaining[c] < 0) remaining[c] = 0;
            total_unseen += remaining[c];
        }

        int opp_hand_sz = static_cast<int>(hands[1].size());

        for (int c = 0; c < M_NUM_COLORS; c++) {
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
            set_channel(59 + c, belief);
        }
    }

    // Ch 65-70: Opp mountain play frequency per color
    // Ch 71-76: Opp field play frequency per color
    // Ch 77-82: Opp discard frequency per color
    {
        float opp_total = static_cast<float>(std::max(total_moves[1], 1));
        for (int c = 0; c < M_NUM_COLORS; c++) {
            set_channel(65 + c, mountain_plays[1][c] / opp_total);
            set_channel(71 + c, field_plays[1][c] / opp_total);
            set_channel(77 + c, discard_plays[1][c] / opp_total);
        }
    }

    // Ch 83: Phase flag
    set_channel(83, phase == M_PHASE_CLAIM ? 1.0f : 0.0f);

    // Ch 84-89: Claiming colors available
    for (int c : claiming_colors_available) {
        set_channel(84 + c, 1.0f);
    }

    // Ch 90-95: My cup color counts
    {
        float counts[M_NUM_COLORS];
        color_counts(cups[0], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(90 + c, counts[c]);
    }

    // Ch 96-101: Discard pile color counts (/18) — face-up public info
    {
        float counts[M_NUM_COLORS];
        color_counts(discard, counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(96 + c, counts[c]);
    }

    // Ch 102-107: Opponent cup color counts (/18) — observable from watching claims
    {
        float counts[M_NUM_COLORS];
        color_counts(cups[1], counts);
        for (int c = 0; c < M_NUM_COLORS; c++) set_channel(102 + c, counts[c]);
    }

    // Ch 108-113: Global visible card count per color (/18)
    // Uses known[] computed above (hand + mountains + fields + rivers + discard)
    for (int c = 0; c < M_NUM_COLORS; c++) {
        set_channel(108 + c, static_cast<float>(known[c]) / 18.0f);
    }

    // Ch 114-119: Dead-color binary flags (1.0 if all 18 of color c are accounted for)
    for (int c = 0; c < M_NUM_COLORS; c++) {
        set_channel(114 + c, known[c] >= M_CARDS_PER_COLOR ? 1.0f : 0.0f);
    }

    // Ch 120: Game progress = (total river cards) / 12.0
    set_channel(120, static_cast<float>(rivers[0].size() + rivers[1].size()) / 12.0f);

    // Ch 121: Field advantage Mandala 0 (my total - opp total), raw count
    set_channel(121, static_cast<float>(
        static_cast<int>(fields[0][0].size()) - static_cast<int>(fields[0][1].size())));

    // Ch 122: Field advantage Mandala 1 (my total - opp total), raw count
    set_channel(122, static_cast<float>(
        static_cast<int>(fields[1][0].size()) - static_cast<int>(fields[1][1].size())));

    // Ch 123-136: Pre-computed scoring channels
    // p=0 (me): score from ALL cup cards. p=1 (opp): known score from claimed cups only (skip 2 hidden starting cards)
    for (int p = 0; p < 2; p++) {
        int river_val[M_NUM_COLORS] = {};
        for (int pos = 0; pos < static_cast<int>(rivers[p].size()); pos++)
            river_val[rivers[p][pos]] = pos + 1;

        int cup_cnt[M_NUM_COLORS] = {};
        int start = (p == 0) ? 0 : 2;  // Skip first 2 for opponent (hidden starting cards)
        for (int i = start; i < static_cast<int>(cups[p].size()); i++)
            cup_cnt[cups[p][i]]++;

        int total = 0;
        for (int c = 0; c < M_NUM_COLORS; c++) {
            int contrib = cup_cnt[c] * river_val[c];
            total += contrib;
            set_channel(125 + p * 6 + c, static_cast<float>(contrib) / 18.0f);
        }
        set_channel(123 + p, static_cast<float>(total) / 100.0f);
    }
}

std::unique_ptr<GameState> MandalaState::get_canonical() const {
    auto s = std::make_unique<MandalaState>();
    *s = *this;

    if (current_player_ == 0) return s;

    // Swap player perspectives
    std::swap(s->hands[0], s->hands[1]);
    std::swap(s->rivers[0], s->rivers[1]);
    std::swap(s->cups[0], s->cups[1]);
    for (int m = 0; m < 2; m++) {
        std::swap(s->fields[m][0], s->fields[m][1]);
    }
    std::swap(s->mountain_plays[0], s->mountain_plays[1]);
    std::swap(s->field_plays[0], s->field_plays[1]);
    std::swap(s->discard_plays[0], s->discard_plays[1]);
    std::swap(s->total_moves[0], s->total_moves[1]);
    // CLAIM phase fields are global — no swap needed
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
    combine(phase);
    combine(claiming_mandala + 10);
    for (auto c : claiming_colors_available) combine(c + 400);
    return h;
}

// --- MandalaGame ---

std::unique_ptr<GameState> MandalaGame::create_initial_state(std::mt19937& rng) {
    auto s = std::make_unique<MandalaState>();

    s->deck.reserve(M_TOTAL_CARDS);
    for (int color = 0; color < M_NUM_COLORS; color++)
        for (int i = 0; i < M_CARDS_PER_COLOR; i++)
            s->deck.push_back(color);

    std::shuffle(s->deck.begin(), s->deck.end(), rng);

    for (int m = 0; m < 2; m++) {
        s->mountains[m].push_back(s->deck.back()); s->deck.pop_back();
        s->mountains[m].push_back(s->deck.back()); s->deck.pop_back();
    }

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
    // Rule of Color: color must not appear in either field
    // But you CAN add more cards of a color already in the mountain
    for (int c : s.fields[mandala][0]) if (c == color) return false;
    for (int c : s.fields[mandala][1]) if (c == color) return false;
    return true;
}

bool MandalaGame::can_play_to_field(const MandalaState& s, int color, int mandala, int player) const {
    for (int c : s.mountains[mandala]) if (c == color) return false;
    int opp = 1 - player;
    for (int c : s.fields[mandala][opp]) if (c == color) return false;
    return true;
}

void MandalaGame::get_valid_moves(const GameState& state_base, std::vector<float>& out) const {
    out.assign(M_NUM_ACTIONS, 0.0f);
    auto& s = static_cast<const MandalaState&>(state_base);

    if (s.game_over) return;

    // CLAIM phase: only CLAIM_COLOR actions (144-149)
    if (s.phase == M_PHASE_CLAIM) {
        for (int color : s.claiming_colors_available) {
            out[144 + color] = 1.0f;
        }
        return;
    }

    // PLAY phase
    auto& hand = s.hands[s.current_player_];
    if (hand.empty()) return;

    std::unordered_map<int, int> hand_color_count;
    for (int c : hand) hand_color_count[c]++;
    int hand_size = static_cast<int>(hand.size());

    // BUILD_MOUNTAIN: action = mandala*6 + color (0-11)
    for (auto& [color, count] : hand_color_count) {
        for (int mandala = 0; mandala < 2; mandala++) {
            if (can_play_to_mountain(s, color, mandala)) {
                out[mandala * 6 + color] = 1.0f;
            }
        }
    }

    // GROW_FIELD: action = 12 + mandala*42 + color*7 + (count-1) (12-95)
    for (auto& [color, count] : hand_color_count) {
        int max_count = std::min(count, hand_size - 1);  // Must keep >=1 card
        if (max_count < 1) continue;
        for (int mandala = 0; mandala < 2; mandala++) {
            if (can_play_to_field(s, color, mandala, s.current_player_)) {
                for (int c = 1; c <= max_count; c++) {
                    out[12 + mandala * 42 + color * 7 + (c - 1)] = 1.0f;
                }
            }
        }
    }

    // DISCARD: action = 96 + color*8 + (count-1) (96-143)
    // Only available when no constructive play (mountain/field) exists
    bool has_constructive = false;
    for (int i = 0; i < 96; i++) {
        if (out[i] > 0.5f) { has_constructive = true; break; }
    }
    if (!has_constructive) {
        for (auto& [color, count] : hand_color_count) {
            for (int c = 1; c <= count; c++) {
                out[96 + color * 8 + (c - 1)] = 1.0f;
            }
        }
    }
}

std::unique_ptr<GameState> MandalaGame::get_next_state(const GameState& state_base, int action_id) const {
    auto new_state_ptr = state_base.copy();
    auto& s = static_cast<MandalaState&>(*new_state_ptr);

    // Handle CLAIM phase actions (144-149)
    if (action_id >= 144) {
        process_claim(s, action_id - 144);
        return new_state_ptr;
    }

    int player = s.current_player_;
    int completed_mandala = -1;

    // Track behavioral accumulators
    s.total_moves[player]++;

    if (action_id < 12) {
        // BUILD_MOUNTAIN
        int mandala = action_id / 6;
        int color = action_id % 6;
        s.mountain_plays[player][color]++;

        auto& hand = s.hands[player];
        for (auto it = hand.begin(); it != hand.end(); ++it) {
            if (*it == color) { hand.erase(it); break; }
        }

        s.mountains[mandala].push_back(color);

        int cards_to_draw = std::min(3, M_MAX_HAND - static_cast<int>(s.hands[player].size()));
        for (int i = 0; i < cards_to_draw; i++) {
            check_and_reshuffle_deck(s);
            if (!s.deck.empty()) {
                s.hands[player].push_back(s.deck.back());
                s.deck.pop_back();
            }
        }

        if (s.is_mandala_complete(mandala)) completed_mandala = mandala;

    } else if (action_id < 96) {
        // GROW_FIELD: decode mandala, color, count
        int a = action_id - 12;
        int mandala = a / 42;
        int r = a % 42;
        int color = r / 7;
        int count = r % 7 + 1;
        s.field_plays[player][color]++;

        // Remove exactly `count` cards of this color
        auto& hand = s.hands[player];
        std::vector<int> to_play, remaining;
        for (int c : hand) {
            if (c == color && static_cast<int>(to_play.size()) < count)
                to_play.push_back(c);
            else
                remaining.push_back(c);
        }
        s.hands[player] = remaining;
        s.fields[mandala][player].insert(s.fields[mandala][player].end(),
                                          to_play.begin(), to_play.end());

        if (s.is_mandala_complete(mandala)) completed_mandala = mandala;

    } else {
        // DISCARD: action = 96 + color*8 + (count-1)
        int a = action_id - 96;
        int color = a / 8;
        int count = a % 8 + 1;
        s.discard_plays[player][color]++;

        auto& hand = s.hands[player];
        std::vector<int> to_discard, remaining;
        for (int c : hand) {
            if (c == color && static_cast<int>(to_discard.size()) < count)
                to_discard.push_back(c);
            else
                remaining.push_back(c);
        }
        s.hands[player] = remaining;
        s.discard.insert(s.discard.end(), to_discard.begin(), to_discard.end());

        for (int i = 0; i < static_cast<int>(to_discard.size()); i++) {
            check_and_reshuffle_deck(s);
            if (!s.deck.empty()) {
                s.hands[player].push_back(s.deck.back());
                s.deck.pop_back();
            }
        }
    }

    // Handle completed Mandala — enter CLAIM phase
    if (completed_mandala >= 0) {
        start_claiming(s, completed_mandala, player);
        return new_state_ptr;
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

void MandalaGame::start_claiming(MandalaState& s, int mandala, int triggering) const {
    int fc0 = static_cast<int>(s.fields[mandala][0].size());
    int fc1 = static_cast<int>(s.fields[mandala][1].size());

    int first_player;
    if (fc0 > fc1) first_player = 0;
    else if (fc1 > fc0) first_player = 1;
    else first_player = 1 - triggering;

    // Get unique colors in mountain
    std::set<int> color_set;
    for (int c : s.mountains[mandala]) color_set.insert(c);
    std::vector<int> colors(color_set.begin(), color_set.end());
    std::sort(colors.begin(), colors.end());

    s.phase = M_PHASE_CLAIM;
    s.claiming_mandala = mandala;
    s.claiming_colors_available = colors;
    s.triggering_player = triggering;
    s.current_player_ = first_player;
}

void MandalaGame::process_claim(MandalaState& s, int color) const {
    int mandala = s.claiming_mandala;
    int claimer = s.current_player_;

    // Gather mountain cards of this color
    std::vector<int> cards;
    for (int c : s.mountains[mandala]) {
        if (c == color) cards.push_back(c);
    }

    // Award cards
    int fc = static_cast<int>(s.fields[mandala][claimer].size());
    if (fc == 0) {
        s.discard.insert(s.discard.end(), cards.begin(), cards.end());
    } else {
        auto river_colors = s.get_colors_in_river(claimer);
        if (river_colors.count(color)) {
            s.cups[claimer].insert(s.cups[claimer].end(), cards.begin(), cards.end());
        } else {
            if (!cards.empty()) {
                s.rivers[claimer].push_back(cards[0]);
                for (size_t i = 1; i < cards.size(); i++) {
                    s.cups[claimer].push_back(cards[i]);
                }
            }
        }
    }

    // Remove color from available
    auto& avail = s.claiming_colors_available;
    avail.erase(std::remove(avail.begin(), avail.end(), color), avail.end());

    if (avail.empty()) {
        finish_claiming(s);
    } else {
        s.current_player_ = 1 - claimer;
    }
}

void MandalaGame::finish_claiming(MandalaState& s) const {
    int mandala = s.claiming_mandala;
    int triggering = s.triggering_player;

    // Discard field cards (per rules: "place all field cards in the discard pile")
    auto& f0 = s.fields[mandala][0];
    auto& f1 = s.fields[mandala][1];
    s.discard.insert(s.discard.end(), f0.begin(), f0.end());
    s.discard.insert(s.discard.end(), f1.begin(), f1.end());
    f0.clear();
    f1.clear();

    // Clear mountain
    s.mountains[mandala].clear();

    // Reset CLAIM phase
    s.phase = M_PHASE_PLAY;
    s.claiming_mandala = -1;
    s.claiming_colors_available.clear();
    s.triggering_player = -1;

    if (should_game_end(s)) {
        s.game_over = true;
        s.current_player_ = 1 - triggering;
        return;
    }

    // Refill mountain with 2 cards
    check_and_reshuffle_deck(s);
    if (!s.deck.empty()) {
        s.mountains[mandala].push_back(s.deck.back());
        s.deck.pop_back();
    }
    check_and_reshuffle_deck(s);
    if (!s.deck.empty()) {
        s.mountains[mandala].push_back(s.deck.back());
        s.deck.pop_back();
    }

    // Switch to other player
    s.current_player_ = 1 - triggering;

    // End game if next player has no cards and deck is exhausted
    if (s.hands[s.current_player_].empty()) {
        check_and_reshuffle_deck(s);
        if (s.deck.empty()) {
            s.game_over = true;
        }
    }
}

void MandalaGame::check_and_reshuffle_deck(MandalaState& s) const {
    if (s.deck.empty() && !s.deck_reshuffled && !s.discard.empty()) {
        s.deck = s.discard;
        s.discard.clear();
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

void MandalaGame::randomize_hidden(GameState& state_base, std::mt19937& rng) const {
    auto& s = static_cast<MandalaState&>(state_base);
    int me = s.current_player_;
    int opp = 1 - me;

    // Only shuffle truly hidden information:
    // - Opponent's hand (unknown)
    // - Deck (unknown)
    // - Opponent's cups (2 initial face-down cards unknown; claimed additions known but we
    //   don't track provenance, so shuffle all opp cup cards — minor imprecision)
    // NOT shuffled:
    // - Discard pile (face-up, public info)
    // - Current player's cups (player can look at their own cup)
    std::vector<int> unseen;
    unseen.reserve(s.hands[opp].size() + s.deck.size() + s.cups[opp].size());
    unseen.insert(unseen.end(), s.hands[opp].begin(), s.hands[opp].end());
    unseen.insert(unseen.end(), s.deck.begin(), s.deck.end());
    unseen.insert(unseen.end(), s.cups[opp].begin(), s.cups[opp].end());

    size_t opp_hand_sz = s.hands[opp].size();
    size_t deck_sz = s.deck.size();
    size_t opp_cup_sz = s.cups[opp].size();

    std::shuffle(unseen.begin(), unseen.end(), rng);
    size_t pos = 0;
    s.hands[opp].assign(unseen.begin() + pos, unseen.begin() + pos + opp_hand_sz);
    pos += opp_hand_sz;
    s.deck.assign(unseen.begin() + pos, unseen.begin() + pos + deck_sz);
    pos += deck_sz;
    s.cups[opp].assign(unseen.begin() + pos, unseen.begin() + pos + opp_cup_sz);
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

    int cups_p = static_cast<int>(s.cups[player].size());
    int cups_opp = static_cast<int>(s.cups[1 - player].size());
    if (cups_p > cups_opp) return 1.0f;
    if (cups_p < cups_opp) return -1.0f;
    return 0.0f;
}

int MandalaGame::get_score(const GameState& state_base, int player) const {
    auto& s = static_cast<const MandalaState&>(state_base);
    return calculate_score(s, player);
}

int MandalaGame::calculate_score(const MandalaState& s, int player) const {
    int score = 0;
    std::unordered_map<int, int> color_values;
    for (int pos = 0; pos < static_cast<int>(s.rivers[player].size()); pos++) {
        color_values[s.rivers[player][pos]] = pos + 1;
    }
    for (int c : s.cups[player]) {
        auto it = color_values.find(c);
        if (it != color_values.end()) score += it->second;
    }
    return score;
}
