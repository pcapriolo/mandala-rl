"""
Benchmark bots for evaluation: Random and Basic Strategy.

Both bots implement the same interface as a neural network model:
  - eval() method (no-op, returns self)
  - to(device) method (no-op, returns self)
  - __call__(batch_tensor) -> (logits, values, scores)

The MCTS engine doesn't care what produces the policy — it just needs
tensors of the right shape. Invalid moves are masked in C++.
"""
import torch


class RandomBot:
    """Returns uniform policy (equal weight on all actions) + neutral value.

    With uniform logits, softmax gives equal probability over all valid actions
    (C++ masks invalids). Zero values mean MCTS Q-values stay at 0, so action
    selection is driven purely by priors = effectively random play.
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, batch):
        N = batch.shape[0]
        logits = torch.zeros(N, self.num_actions, device=batch.device)
        values = torch.zeros(N, 1, device=batch.device)
        return logits, values, torch.zeros(N, 1, device=logits.device), torch.zeros(N, 12, device=logits.device)


class MandalaStrategyBot:
    """Expert-level heuristic Mandala bot.

    Implements key strategic concepts from expert play:
    - River timing: delay claiming high-volume colors for high-value river slots
    - Hate-drafting: deny opponent valuable claims even at own cost
    - Mountain blocking: avoid completing mandalas the opponent will win
    - Discard suppression: experts almost never discard
    - Field momentum: prefer building on existing field presence

    Action space (150 actions):
      0-11:    BUILD_MOUNTAIN  = mandala * 6 + color
      12-95:   GROW_FIELD      = 12 + mandala * 42 + color * 7 + (count - 1)
      96-143:  DISCARD         = 96 + color * 8 + (count - 1)
      144-149: CLAIM_COLOR     = 144 + color

    Tensor encoding (96 channels, broadcast across 8x8):
      Ch 0-5:   My hand count per color (/18)
      Ch 6-11:  Mountain 0 per color (/18)
      Ch 12-17: Mountain 1 per color (/18)
      Ch 18-23: My field, mandala 0, per color (/18)
      Ch 24-29: My field, mandala 1, per color (/18)
      Ch 30-35: Opp field, mandala 0, per color (/18)
      Ch 36-41: Opp field, mandala 1, per color (/18)
      Ch 42-47: My river value per color (/6, 0=not claimed)
      Ch 48-53: Opp river value per color (/6)
      Ch 54:    My cup size (/20)
      Ch 55:    Opp cup size (/20)
      Ch 56:    Opp hand size (/8)
      Ch 57:    Deck size (/108)
      Ch 58:    game_ends_next_mandala flag
      Ch 83:    Phase flag (1=CLAIM, 0=PLAY)
      Ch 84-89: Claiming colors available
      Ch 90-95: My cup color counts (/18)
    """

    NUM_ACTIONS = 150

    def __init__(self, num_actions=150):
        self.num_actions = num_actions

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, batch):
        N = batch.shape[0]
        dev = batch.device
        logits = torch.full((N, self.num_actions), -10.0, device=dev)

        # Decode state tensor (all channels are broadcast, value at [ch, 0, 0])
        hand = batch[:, 0:6, 0, 0] * 18           # (N, 6) card counts per color
        mtn0 = batch[:, 6:12, 0, 0] * 18          # mountain 0 occupancy
        mtn1 = batch[:, 12:18, 0, 0] * 18         # mountain 1 occupancy
        my_f0 = batch[:, 18:24, 0, 0] * 18        # my field, mandala 0
        my_f1 = batch[:, 24:30, 0, 0] * 18        # my field, mandala 1
        opp_f0 = batch[:, 30:36, 0, 0] * 18       # opp field, mandala 0
        opp_f1 = batch[:, 36:42, 0, 0] * 18       # opp field, mandala 1
        my_river = batch[:, 42:48, 0, 0] * 6      # river positions (1-6, 0=unclaimed)
        opp_river = batch[:, 48:54, 0, 0] * 6     # opp river positions
        phase = batch[:, 83, 0, 0]                 # 1.0 = CLAIM phase
        claim_avail = batch[:, 84:90, 0, 0]        # claiming colors available
        my_cup = batch[:, 90:96, 0, 0] * 18       # my cup color counts
        deck_size = batch[:, 57, 0, 0] * 108      # cards remaining in deck

        is_claim = (phase > 0.5)  # (N,) bool

        # Count colors on each mountain (how close to completion)
        mtn0_colors = (mtn0 > 0).float().sum(dim=1)  # (N,)
        mtn1_colors = (mtn1 > 0).float().sum(dim=1)

        # My field advantage per mandala
        my_f0_total = my_f0.sum(dim=1)
        opp_f0_total = opp_f0.sum(dim=1)
        my_f1_total = my_f1.sum(dim=1)
        opp_f1_total = opp_f1.sum(dim=1)

        # River slots filled (how many colors claimed so far)
        my_river_filled = (my_river > 0.1).float().sum(dim=1)    # (N,) 0-6
        opp_river_filled = (opp_river > 0.1).float().sum(dim=1)

        # Next river position value (higher = more valuable)
        my_next_river_pos = my_river_filled + 1  # (N,) next slot is worth this many pts/card

        # Hand size for count validation
        hand_size = hand.sum(dim=1)  # (N,)

        # Game progress (0=early, 1=late)
        progress = 1.0 - deck_size / 108.0  # (N,)

        for c in range(6):
            h = hand[:, c]  # (N,) count of this color in hand

            # --- CLAIM_COLOR (actions 144-149) ---
            # Expert strategy: optimize river order + hate-draft
            action = 144 + c
            in_claim = is_claim & (claim_avail[:, c] > 0.5)

            # Score for claiming: river value × expected cup cards
            # New river color: next position value × (cup cards we'd score)
            # Already in river: extra cup cards × existing river value
            new_river_value = my_next_river_pos * (my_cup[:, c] + 1)  # +1 for the claim itself
            existing_value = my_river[:, c] * my_cup[:, c] * 0.5

            my_claim_value = torch.where(
                my_river[:, c] < 0.1,  # not in river yet
                new_river_value,
                existing_value,
            )

            # Hate-drafting: estimate what opponent would gain if they claim this
            # Opponent gains: their next river pos × their cup count for this color
            opp_next_pos = opp_river_filled + 1
            opp_claim_value = torch.where(
                opp_river[:, c] < 0.1,
                opp_next_pos * 2.0,  # estimate opponent cup cards (unknown, use 2)
                torch.tensor(1.0, device=dev),
            )

            # Total claim score: own gain + deny value
            claim_weight = my_claim_value + opp_claim_value * 0.5

            claim_weight = torch.where(in_claim, claim_weight, torch.tensor(-10.0, device=dev))
            logits[:, action] = claim_weight

            # --- GROW_FIELD (actions 12-95) ---
            for m_idx in range(2):
                if m_idx == 0:
                    my_field_c = my_f0[:, c]
                    opp_field_c = opp_f0[:, c]
                    my_field_total = my_f0_total
                    opp_field_total = opp_f0_total
                    mtn_colors = mtn0_colors
                else:
                    my_field_c = my_f1[:, c]
                    opp_field_c = opp_f1[:, c]
                    my_field_total = my_f1_total
                    opp_field_total = opp_f1_total
                    mtn_colors = mtn1_colors

                for count in range(1, 8):
                    action = 12 + m_idx * 42 + c * 7 + (count - 1)
                    weight = torch.full((N,), -10.0, device=dev)

                    valid = (~is_claim) & (h >= count) & (hand_size > count)

                    # Base: prefer playing more cards
                    base = torch.tensor(count * 2.0, device=dev)

                    # Bonus for existing field presence (momentum)
                    base = base + (my_field_c > 0).float() * 3.0

                    # Bonus for building toward field advantage before completion
                    near_complete = (mtn_colors >= 4)
                    building_lead = near_complete & (my_field_total + count > opp_field_total)
                    base = base + building_lead.float() * 2.0

                    # Late game bonus: bigger field plays are more decisive
                    base = base + progress * count * 0.5

                    weight = torch.where(valid, base, weight)
                    logits[:, action] = weight

            # --- BUILD_MOUNTAIN (actions 0-11) ---
            for m_idx in range(2):
                action = m_idx * 6 + c
                if m_idx == 0:
                    mtn_colors = mtn0_colors
                    my_field_total = my_f0_total
                    opp_field_total = opp_f0_total
                    mtn_c = mtn0[:, c]
                else:
                    mtn_colors = mtn1_colors
                    my_field_total = my_f1_total
                    opp_field_total = opp_f1_total
                    mtn_c = mtn1[:, c]

                # Base: moderate desire to build mountains (sets up mandalas)
                weight = torch.tensor(2.0, device=dev).expand(N).clone()

                # Mountain stacking: slight bonus if color already in mountain (builds claim value)
                weight = weight + (mtn_c > 0).float() * 1.0

                # Mountain blocking: don't complete if opponent has field advantage
                completing = (mtn_colors >= 5)
                opp_advantage = (opp_field_total > my_field_total)
                weight = weight - (completing & opp_advantage).float() * 8.0

                # Push completion if we have field advantage
                our_advantage = (my_field_total > opp_field_total) & completing
                weight = weight + our_advantage.float() * 5.0

                # Early game: mountain building is good (sets up future claims)
                weight = weight + (1.0 - progress) * 1.0

                # Suppress during CLAIM phase
                weight = torch.where(is_claim, torch.tensor(-10.0, device=dev), weight)

                # Need card in hand
                weight = torch.where(h > 0, weight, torch.tensor(-10.0, device=dev))

                logits[:, action] = weight

            # --- DISCARD (actions 96-143) ---
            # Discard is a strategic tool: dump unwanted colors, draw fresh cards.
            for count in range(1, 9):
                action = 96 + c * 8 + (count - 1)
                weight = torch.tensor(-3.0, device=dev).expand(N).clone()
                valid = (~is_claim) & (h >= count)
                # Prefer discarding more of a clogged color
                weight = weight + torch.clamp(h - 3, min=0.0) * 0.5
                weight = weight + torch.tensor(count * 0.2, device=dev)
                weight = torch.where(valid, weight, torch.tensor(-10.0, device=dev))
                logits[:, action] = weight

        values = torch.zeros(N, 1, device=dev)
        return logits, values, torch.zeros(N, 1, device=logits.device), torch.zeros(N, 12, device=logits.device)


class LostCitiesStrategyBot:
    """Expert-level heuristic Lost Cities bot.

    Implements key strategic concepts from expert play:
    - Wager awareness: wagers multiply both gains AND losses
    - Expedition profitability: estimate ROI before committing
    - Dynamic starting threshold: need more cards with more wagers
    - Score gap awareness: play safe when ahead, risky when behind
    - Opponent feeding prevention: never discard what opponent needs
    - 8-card bonus pursuit: push for 20pt bonus when close

    Action space (96 actions):
      action = hand_pos * 12 + dest * 6 + draw_src
      hand_pos: 0-7 (which card in hand)
      dest: 0 = play to expedition, 1 = discard
      draw_src: 0 = deck, 1-5 = discard pile for color (draw_src - 1)

    Tensor encoding (86 channels, broadcast across 8x8):
      Ch 0-4:   My hand count per color (/8)
      Ch 5-9:   My hand wager count per color (/3)
      Ch 10-14: My expedition top value per color (/10)
      Ch 15-19: My expedition length per color (/12)
      Ch 20-24: My expedition wager count per color (/3)
      Ch 25-29: Opp expedition top value per color (/10)
      Ch 30-34: Opp expedition length per color (/12)
      Ch 35-39: Opp expedition wager count per color (/3)
      Ch 40-44: Discard pile top value per color (/10)
      Ch 45:    Deck size (/44)
      Ch 46:    Opp hand size (/8)
      Ch 47:    My score (/200 + 0.5)
      Ch 48:    Opp score (/200 + 0.5)
      Ch 49:    Game progress (turns / max_turns)
      Ch 50-57: Hand pos 0-7 card color ((color+1)/6, 0=empty)
      Ch 58-65: Hand pos 0-7 card value ((value+1)/11, 0=empty)
    """

    NUM_ACTIONS = 96

    def __init__(self, num_actions=96):
        self.num_actions = num_actions

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, batch):
        N = batch.shape[0]
        dev = batch.device
        logits = torch.full((N, self.num_actions), -10.0, device=dev)

        # Decode tensor
        hand_count = batch[:, 0:5, 0, 0] * 8        # (N, 5) cards per color
        hand_wagers = batch[:, 5:10, 0, 0] * 3      # (N, 5) wager cards in hand
        my_exp_top = batch[:, 10:15, 0, 0] * 10     # my expedition top value
        my_exp_len = batch[:, 15:20, 0, 0] * 12     # my expedition length
        my_exp_wagers = batch[:, 20:25, 0, 0] * 3   # my expedition wager count
        opp_exp_top = batch[:, 25:30, 0, 0] * 10    # opp expedition top value
        opp_exp_len = batch[:, 30:35, 0, 0] * 12    # opp expedition length
        opp_exp_wagers = batch[:, 35:40, 0, 0] * 3  # opp expedition wagers
        discard_top = batch[:, 40:45, 0, 0] * 10    # discard pile top value
        deck_size = batch[:, 45, 0, 0] * 44         # cards left in deck
        my_score = (batch[:, 47, 0, 0] - 0.5) * 200   # my current score
        opp_score = (batch[:, 48, 0, 0] - 0.5) * 200  # opp current score
        progress = batch[:, 49, 0, 0]               # game progress 0-1

        # Per-hand-position card info
        hand_colors = batch[:, 50:58, 0, 0] * 6 - 1    # (N, 8) color, -1 = empty
        hand_values = batch[:, 58:66, 0, 0] * 11 - 1   # (N, 8) value, -1 = empty

        # Score gap: positive = we're ahead
        score_gap = my_score - opp_score  # (N,)
        # Risk adjustment: play safe when ahead, aggressive when behind
        risk_factor = torch.clamp(-score_gap / 50.0, min=-1.0, max=1.0)  # -1=safe, +1=risky

        # Count active expeditions
        my_active_exps = (my_exp_len > 0).float().sum(dim=1)  # (N,)

        for hp in range(8):
            card_color = hand_colors[:, hp]     # (N,) float, -1 = empty
            card_value = hand_values[:, hp]     # (N,) float, 0 = wager, 1-10 = numbered
            has_card = (card_color >= 0)         # (N,) bool
            is_wager = has_card & (card_value < 0.5)  # wager cards have value 0

            for dest in range(2):
                for draw_src in range(6):
                    action = hp * 12 + dest * 6 + draw_src
                    weight = torch.full((N,), -10.0, device=dev)

                    if dest == 0:
                        # PLAY TO EXPEDITION
                        for c in range(5):
                            is_this_color = has_card & (card_color.round().long() == c)
                            if not is_this_color.any():
                                continue

                            exp_len_c = my_exp_len[:, c]
                            hcount_c = hand_count[:, c]
                            wagers_c = my_exp_wagers[:, c]
                            hand_wagers_c = hand_wagers[:, c]

                            # Multiplier from wagers: (1 + wager_count)
                            multiplier = 1.0 + wagers_c

                            # --- CONTINUING an existing expedition ---
                            # Strong base score: continuing is almost always good
                            continue_weight = torch.tensor(4.0, device=dev)
                            # Higher value cards are more valuable
                            continue_weight = continue_weight + card_value * 0.3
                            # 8-card bonus pursuit: strong push when close
                            close_to_bonus = (exp_len_c >= 5) & (exp_len_c < 8)
                            continue_weight = continue_weight + close_to_bonus.float() * 3.0
                            # Wager multiplier makes each card more valuable
                            continue_weight = continue_weight + wagers_c * 0.5

                            # --- STARTING a new expedition ---
                            # Lower threshold: with deck draws, game ends in ~22 turns
                            # so we need to commit early. 2 cards = ok to start.
                            min_cards_needed = 2.0 + hand_wagers_c
                            has_enough = hcount_c >= min_cards_needed

                            # Don't start too many expeditions (3-4 max)
                            too_many = my_active_exps >= 4

                            # Risk-adjusted starting: easier to start when behind
                            start_weight = torch.where(
                                has_enough & (~too_many),
                                torch.tensor(2.0, device=dev) + risk_factor * 1.0,
                                torch.tensor(-4.0, device=dev),
                            )

                            # Wager card to new expedition: very risky
                            wager_start_penalty = is_wager.float() * (
                                torch.where(exp_len_c > 0,
                                    torch.tensor(0.0, device=dev),  # wager on existing: ok early
                                    torch.tensor(-3.0, device=dev))  # wager on new: very bad
                            )

                            # Late game penalty: don't start expeditions late
                            late_start_penalty = torch.where(
                                (exp_len_c < 0.5) & (progress > 0.5),
                                torch.tensor(-3.0, device=dev),
                                torch.tensor(0.0, device=dev),
                            )

                            play_weight = torch.where(
                                exp_len_c > 0,
                                continue_weight,
                                start_weight,
                            ) + wager_start_penalty + late_start_penalty

                            weight = torch.where(is_this_color, play_weight, weight)

                    else:
                        # DISCARD
                        for c in range(5):
                            is_this_color = has_card & (card_color.round().long() == c)
                            if not is_this_color.any():
                                continue

                            exp_len_c = my_exp_len[:, c]
                            opp_len_c = opp_exp_len[:, c]
                            opp_wagers_c = opp_exp_wagers[:, c]

                            # Discard colors we're not collecting
                            discard_weight = torch.where(
                                exp_len_c > 0,
                                torch.tensor(-6.0, device=dev),  # we're collecting — never
                                torch.tensor(1.5, device=dev),   # not collecting — safe
                            )

                            # Opponent feeding: penalize based on their investment
                            # Opponent with wagers gets multiplied benefit
                            opp_multiplier = 1.0 + opp_wagers_c
                            feeding_penalty = (opp_len_c > 0).float() * opp_multiplier * 1.5
                            discard_weight = discard_weight - feeding_penalty

                            # Prefer discarding low-value cards (less useful to opponent)
                            discard_weight = discard_weight - card_value * 0.15

                            # Wager discard: safe if not collecting (can't feed wager)
                            discard_weight = discard_weight + is_wager.float() * (
                                torch.where(exp_len_c > 0,
                                    torch.tensor(-2.0, device=dev),  # our wager: terrible
                                    torch.tensor(0.5, device=dev))   # not our color: fine
                            )

                            weight = torch.where(is_this_color, discard_weight, weight)

                    # Draw source preference
                    # KEY INSIGHT: deck draws deplete deck → end game naturally.
                    # In real LC, games ALWAYS end by deck depletion (~44 turns).
                    # Discard draws extend games → 60-turn timeout = degenerate play.
                    # Almost always draw from deck; discard draw only for exceptional pickups.
                    if draw_src == 0:
                        # Draw from deck — DOMINANT default, depletes deck toward natural end
                        weight = weight + 3.0
                    else:
                        # Draw from discard pile (draw_src - 1 = color)
                        dc = draw_src - 1
                        has_discard = discard_top[:, dc] > 0
                        collecting = my_exp_len[:, dc] > 0
                        discard_val = discard_top[:, dc]
                        exp_wagers_dc = my_exp_wagers[:, dc]

                        # EXCEPTIONAL: collecting with wagers AND high-value card (8+)
                        # Only case where extending game is worth it
                        exceptional = collecting & (discard_val >= 8) & (exp_wagers_dc >= 1)
                        weight = torch.where(
                            has_discard & exceptional,
                            weight + 0.0,  # Neutral — still loses to deck draw's +3.0
                            weight - 10.0, # Hard penalty: almost never draw from discard
                        )

                    logits[:, action] = weight

        values = torch.zeros(N, 1, device=dev)
        return logits, values, torch.zeros(N, 1, device=logits.device), torch.zeros(N, 12, device=logits.device)


class DominionStrategyBot:
    """Big Money heuristic bot for Dominion.

    Buy priority: Province > Gold > Duchy (late) > Silver > Estate (very late).
    Plays all treasures, ends action phase immediately (no action play).
    Uses tensor channels for context-aware buying decisions.

    Tensor encoding (151 channels, broadcast 8x8):
      Ch 31-61:  My hand (count per card type / 10)
      Ch 93-123: Supply remaining (count per pile / 12)
      Ch 126:    My coins / 12
      Ch 128:    My buys remaining / 5
      Ch 129:    Phase (0=action, 0.33=buy, 0.67=select, 1.0=react)
      Ch 131:    Provinces remaining / 8
      Ch 138:    My VP / 40
      Ch 139:    Opp VP / 40

    Action space (131):
      0: END_ACTIONS, 1: END_BUYS, 2: DONE_SELECTING
      3-33: PLAY, 34-64: BUY, 65-95: SELECT, 96-126: GAIN
      127: REVEAL_MOAT, 128: DECLINE_REACTION, 129: ACCEPT, 130: DECLINE
    """

    NUM_ACTIONS = 131
    COPPER, SILVER, GOLD = 0, 1, 2
    ESTATE, DUCHY, PROVINCE, CURSE = 3, 4, 5, 6

    def __init__(self, num_actions=131):
        self.num_actions = num_actions

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, batch):
        N = batch.shape[0]
        dev = batch.device
        logits = torch.full((N, self.num_actions), -10.0, device=dev)

        my_hand = batch[:, 31:62, 0, 0] * 10
        supply = batch[:, 93:124, 0, 0] * 12
        my_coins = batch[:, 126, 0, 0] * 12
        my_buys = batch[:, 128, 0, 0] * 5
        phase = batch[:, 129, 0, 0]
        provinces_left = batch[:, 131, 0, 0] * 8
        my_vp = batch[:, 138, 0, 0] * 40
        opp_vp = batch[:, 139, 0, 0] * 40

        is_action = phase < 0.1
        is_buy = (phase > 0.2) & (phase < 0.5)
        is_select = (phase > 0.5) & (phase < 0.9)
        is_react = phase > 0.9
        has_buys = my_buys > 0.5

        # Action phase: always end immediately
        logits[:, 0] = torch.where(is_action, torch.tensor(5.0, device=dev), logits[:, 0])

        # Buy phase: play all treasures first
        for tid in [self.COPPER, self.SILVER, self.GOLD]:
            has_t = my_hand[:, tid] > 0.1
            logits[:, 3 + tid] = torch.where(is_buy & has_t, torch.tensor(8.0, device=dev), logits[:, 3 + tid])

        # Buy Province (cost 8)
        can = is_buy & has_buys & (my_coins >= 8) & (supply[:, self.PROVINCE] > 0.5)
        logits[:, 34 + self.PROVINCE] = torch.where(can, torch.tensor(10.0, device=dev), logits[:, 34 + self.PROVINCE])

        # Buy Gold (cost 6)
        can = is_buy & has_buys & (my_coins >= 6) & (supply[:, self.GOLD] > 0.5)
        logits[:, 34 + self.GOLD] = torch.where(can, torch.tensor(7.0, device=dev), logits[:, 34 + self.GOLD])

        # Buy Duchy (cost 5) — only when greening (<=4 provinces left)
        can = is_buy & has_buys & (my_coins >= 5) & (supply[:, self.DUCHY] > 0.5) & (provinces_left < 4.5)
        logits[:, 34 + self.DUCHY] = torch.where(can, torch.tensor(6.0, device=dev), logits[:, 34 + self.DUCHY])

        # Buy Silver (cost 3)
        can = is_buy & has_buys & (my_coins >= 3) & (supply[:, self.SILVER] > 0.5)
        logits[:, 34 + self.SILVER] = torch.where(can, torch.tensor(4.0, device=dev), logits[:, 34 + self.SILVER])

        # Buy Estate (cost 2) — very late game only (<=2 provinces)
        can = is_buy & has_buys & (my_coins >= 2) & (supply[:, self.ESTATE] > 0.5) & (provinces_left < 2.5)
        logits[:, 34 + self.ESTATE] = torch.where(can, torch.tensor(3.0, device=dev), logits[:, 34 + self.ESTATE])

        # End buys fallback
        logits[:, 1] = torch.where(is_buy, torch.tensor(1.0, device=dev), logits[:, 1])

        # React: prefer Moat reveal
        logits[:, 127] = torch.where(is_react, torch.tensor(5.0, device=dev), logits[:, 127])
        logits[:, 128] = torch.where(is_react, torch.tensor(1.0, device=dev), logits[:, 128])

        # Select: uniform fallback (C++ masks invalids)
        logits[:, 129] = torch.where(is_select, torch.tensor(2.0, device=dev), logits[:, 129])
        logits[:, 130] = torch.where(is_select, torch.tensor(1.0, device=dev), logits[:, 130])
        logits[:, 2] = torch.where(is_select, torch.tensor(0.5, device=dev), logits[:, 2])
        for i in range(31):
            logits[:, 65 + i] = torch.where(is_select, torch.tensor(0.0, device=dev), logits[:, 65 + i])
            logits[:, 96 + i] = torch.where(is_select, torch.tensor(0.0, device=dev), logits[:, 96 + i])

        values = torch.zeros(N, 1, device=dev)
        return logits, values, torch.zeros(N, 1, device=logits.device), torch.zeros(N, 12, device=logits.device)
