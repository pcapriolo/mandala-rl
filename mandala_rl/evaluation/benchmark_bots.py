"""
Benchmark bots for evaluation: Random and Basic Strategy.

Both bots implement the same interface as a neural network model:
  - eval() method (no-op, returns self)
  - to(device) method (no-op, returns self)
  - __call__(batch_tensor) -> (logits, values)

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
        return logits, values


class MandalaStrategyBot:
    """Heuristic Mandala bot following basic strategy.

    Action space (30 actions):
      0-11:  BUILD_MOUNTAIN  = mandala_idx * 6 + color
      12-23: GROW_FIELD      = 12 + mandala_idx * 6 + color
      24-29: DISCARD         = 24 + color

    Tensor encoding (83 channels, broadcast across 8x8):
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
    """

    NUM_ACTIONS = 30

    def __init__(self, num_actions=30):
        self.num_actions = num_actions

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, batch):
        N = batch.shape[0]
        logits = torch.full((N, self.num_actions), -10.0, device=batch.device)

        # Decode state tensor (all channels are broadcast, value at [ch, 0, 0])
        hand = batch[:, 0:6, 0, 0] * 18           # (N, 6) card counts per color
        mtn0 = batch[:, 6:12, 0, 0] * 18          # mountain 0 occupancy
        mtn1 = batch[:, 12:18, 0, 0] * 18         # mountain 1 occupancy
        my_f0 = batch[:, 18:24, 0, 0] * 18        # my field, mandala 0
        my_f1 = batch[:, 24:30, 0, 0] * 18        # my field, mandala 1
        opp_f0 = batch[:, 30:36, 0, 0] * 18       # opp field, mandala 0
        opp_f1 = batch[:, 36:42, 0, 0] * 18       # opp field, mandala 1
        my_river = batch[:, 42:48, 0, 0] * 6      # river positions
        game_ending = batch[:, 58, 0, 0]           # game_ends_next_mandala

        # Count colors on each mountain (how close to completion)
        mtn0_colors = (mtn0 > 0).float().sum(dim=1)  # (N,)
        mtn1_colors = (mtn1 > 0).float().sum(dim=1)

        # My field advantage per mandala
        my_f0_total = my_f0.sum(dim=1)
        opp_f0_total = opp_f0.sum(dim=1)
        my_f1_total = my_f1.sum(dim=1)
        opp_f1_total = opp_f1.sum(dim=1)

        for c in range(6):
            h = hand[:, c]  # (N,) count of this color in hand

            # --- GROW_FIELD (actions 12-23) ---
            # Prefer colors with many copies (playing all at once = big field)
            # Higher weight for more cards of this color
            for m_idx in range(2):
                action = 12 + m_idx * 6 + c
                weight = h * 1.5  # more cards = stronger field play

                # Bonus if we already have field presence in this mandala
                if m_idx == 0:
                    weight = weight + (my_f0[:, c] > 0).float() * 2.0
                else:
                    weight = weight + (my_f1[:, c] > 0).float() * 2.0

                logits[:, action] = weight

            # --- BUILD_MOUNTAIN (actions 0-11) ---
            # Prefer colors with few copies in hand (less valuable to keep)
            # Avoid completing mandala when opponent has field advantage
            for m_idx in range(2):
                action = m_idx * 6 + c
                weight = torch.clamp(4.0 - h, min=0.0)  # fewer in hand = higher

                # Penalize if this would complete a mandala (6th color) and
                # opponent has more field cards there
                if m_idx == 0:
                    completing = (mtn0_colors >= 5)
                    opp_advantage = (opp_f0_total > my_f0_total)
                    weight = weight - (completing & opp_advantage).float() * 5.0
                else:
                    completing = (mtn1_colors >= 5)
                    opp_advantage = (opp_f1_total > my_f1_total)
                    weight = weight - (completing & opp_advantage).float() * 5.0

                # Bonus for completing mandala when WE have field advantage
                if m_idx == 0:
                    our_advantage = (my_f0_total > opp_f0_total) & completing
                else:
                    our_advantage = (my_f1_total > opp_f1_total) & completing
                weight = weight + our_advantage.float() * 3.0

                logits[:, action] = weight

            # --- DISCARD (actions 24-29) ---
            # Low priority — only useful for hand cycling
            action = 24 + c
            logits[:, action] = torch.clamp(h - 2, min=-1.0) * 0.5

        values = torch.zeros(N, 1, device=batch.device)
        return logits, values


class LostCitiesStrategyBot:
    """Heuristic Lost Cities bot following basic strategy.

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
        logits = torch.full((N, self.num_actions), -10.0, device=batch.device)

        # Decode tensor
        hand_count = batch[:, 0:5, 0, 0] * 8        # (N, 5) cards per color
        my_exp_top = batch[:, 10:15, 0, 0] * 10     # my expedition top value
        my_exp_len = batch[:, 15:20, 0, 0] * 12     # my expedition length
        my_exp_wagers = batch[:, 20:25, 0, 0] * 3   # my expedition wagers
        opp_exp_len = batch[:, 30:35, 0, 0] * 12    # opp expedition length
        discard_top = batch[:, 40:45, 0, 0] * 10    # discard pile top value
        deck_size = batch[:, 45, 0, 0] * 44         # cards left in deck
        progress = batch[:, 49, 0, 0]               # game progress 0-1

        # Per-hand-position card info
        hand_colors = batch[:, 50:58, 0, 0] * 6 - 1    # (N, 8) color, -1 = empty
        hand_values = batch[:, 58:66, 0, 0] * 11 - 1   # (N, 8) value, -1 = empty

        for hp in range(8):
            card_color = hand_colors[:, hp]     # (N,) float, -1 = empty
            card_value = hand_values[:, hp]     # (N,) float, 0 = wager, 1-10 = numbered
            has_card = (card_color >= 0)         # (N,) bool

            for dest in range(2):
                for draw_src in range(6):
                    action = hp * 12 + dest * 6 + draw_src
                    weight = torch.full((N,), -10.0, device=batch.device)

                    if dest == 0:
                        # PLAY TO EXPEDITION
                        # Need to figure out if this card can extend our expedition
                        for c in range(5):
                            is_this_color = has_card & (card_color.round().long() == c)
                            if not is_this_color.any():
                                continue

                            exp_len_c = my_exp_len[:, c]
                            exp_top_c = my_exp_top[:, c]
                            hcount_c = hand_count[:, c]

                            # Base: prefer playing to existing expeditions
                            play_weight = torch.where(
                                exp_len_c > 0,
                                torch.tensor(3.0, device=batch.device),  # continuing
                                torch.where(
                                    hcount_c >= 3,
                                    torch.tensor(1.0, device=batch.device),  # starting with 3+
                                    torch.tensor(-3.0, device=batch.device),  # starting with <3
                                )
                            )

                            # Bonus for high-value cards on existing expeditions
                            play_weight = play_weight + card_value * 0.2

                            # Bonus for going toward 8-card bonus
                            close_to_bonus = (exp_len_c >= 5) & (exp_len_c < 8)
                            play_weight = play_weight + close_to_bonus.float() * 2.0

                            weight = torch.where(is_this_color.float() > 0, play_weight, weight)

                    else:
                        # DISCARD
                        for c in range(5):
                            is_this_color = has_card & (card_color.round().long() == c)
                            if not is_this_color.any():
                                continue

                            exp_len_c = my_exp_len[:, c]
                            opp_len_c = opp_exp_len[:, c]

                            # Prefer discarding colors we're not collecting
                            discard_weight = torch.where(
                                exp_len_c > 0,
                                torch.tensor(-5.0, device=batch.device),  # we're collecting
                                torch.tensor(1.0, device=batch.device),   # not collecting
                            )

                            # Penalize feeding opponent
                            discard_weight = discard_weight - (opp_len_c > 0).float() * 2.0

                            # Prefer discarding low-value cards
                            discard_weight = discard_weight - card_value * 0.1

                            weight = torch.where(is_this_color.float() > 0, discard_weight, weight)

                    # Draw source preference
                    if draw_src == 0:
                        # Draw from deck — default safe choice
                        weight = weight + 0.5
                    else:
                        # Draw from discard pile (draw_src - 1 = color)
                        dc = draw_src - 1
                        has_discard = discard_top[:, dc] > 0
                        # Only worthwhile if we're collecting that color
                        collecting = my_exp_len[:, dc] > 0
                        weight = torch.where(
                            has_discard & collecting,
                            weight + 1.0,
                            weight - 5.0,  # heavily penalize drawing colors we don't need
                        )

                    logits[:, action] = weight

        values = torch.zeros(N, 1, device=batch.device)
        return logits, values
