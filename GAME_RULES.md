# Mandala - Official Game Rules

**Players:** 2
**Age:** 10+
**Duration:** 20-30 minutes
**Publisher:** Lookout Games

## Overview

Mandala is a tactical 2-player card game where players compete to create and destroy Mandalas by playing colored sand cards. Players earn victory points by collecting cards into their River based on strategic timing and color management.

## Components

- **108 Sand Cards**: 6 colors × 18 cards each
  - 🔴 Red
  - 🟢 Green
  - 🟣 Purple
  - 🟠 Orange
  - 🟡 Yellow
  - ⚪ White

- **2 Mandalas**: Each Mandala has three areas:
  - **Mountain**: Shared central area (neutral)
  - **Field (Player 0)**: One player's personal area
  - **Field (Player 1)**: Other player's personal area

- **Player Areas**:
  - **Hand**: Cards available to play (max 8)
  - **Cup**: Face-down cards for endgame scoring
  - **River**: Face-up cards scored for victory points

## Setup

1. Shuffle all 108 sand cards to form a face-down **deck**
2. For each of the **2 Mandalas**:
   - Draw 2 cards and place face-up in the **Mountain**
3. For each **player**:
   - Draw 6 cards for their starting **hand**
   - Draw 2 cards face-down for their **cup**

**Starting player:** Player 0 goes first

## The Rule of Color

**CRITICAL RULE:** In each Mandala, each color may only appear in **exactly one area** (Mountain, Field P0, or Field P1).

- If a color is in the Mountain, it cannot be played to either Field
- If a color is in your Field, it cannot be played to the Mountain or opponent's Field
- This is the core strategic constraint of the game

## Gameplay

On your turn, you **must** take exactly **one** of these three actions:

### 1. BUILD MOUNTAIN
Play **exactly 1 card** from your hand to a Mountain of your choice.

**Rules:**
- Choose any color from your hand
- Choose which Mandala (0 or 1) to play to
- The color must NOT already be in either player's Field for that Mandala
- Place the card face-up in the Mountain

**Strategy:** Building the Mountain is neutral - both players can claim these cards later.

### 2. GROW FIELD
Play **all cards of one color** from your hand to your Field in one Mandala.

**Rules:**
- Choose a color where you have multiple cards (or just 1)
- Choose which Mandala (0 or 1) to play to
- Play ALL cards of that color to your Field
- The color must NOT already be in the Mountain or opponent's Field
- Cards go in your Field area (not the Mountain)

**Strategy:** Cards in your Field give you first claim when the Mandala completes.

### 3. DISCARD
Discard **all cards of one color** from your hand, then draw the same number of cards.

**Rules:**
- Choose a color and discard ALL cards of that color
- Draw exactly that many cards from the deck
- If the deck is empty, you still discard but don't draw

**Strategy:** Use this to cycle through your hand when you can't make good plays.

## Completing a Mandala

A Mandala is **complete** when it contains **all 6 colors** (across Mountain and both Fields combined).

When a Mandala completes:

### Collection Phase

Players take turns collecting cards from the completed Mandala, starting with the **player who has more cards in their Field** for that Mandala.

**Turn order:**
1. Player with more Field cards goes first
2. Players alternate taking cards
3. Continue until all cards are collected

**On your collection turn:**
- Choose any color from the Mandala (Mountain or either Field)
- Take **all cards of that color** from the Mandala
- **Decision:** Put these cards either in your:
  - **River** (face-up): Score victory points at game end
  - **Cup** (face-down): Hidden cards for endgame scoring

**River scoring positions:**
- Cards are placed in a row from left to right
- Position matters for scoring (see Scoring section)
- You can only have ONE card of each color in your River
- **Rule:** If you take a color that's already in your River, you MUST put the new cards in your Cup instead

### After Collection

After a Mandala is completely collected:
- Each player draws 1 card from the deck
- Remove the completed Mandala from play
- Play continues with the remaining Mandala
- If both Mandalas complete, the game ends

## Game End

The game ends **immediately** when either:

1. **Both Mandalas have been completed and collected**
2. **One player has all 6 colors in their River**
3. **The deck is exhausted** (no more cards to draw)

## Scoring

### River Scoring
Cards in your River score based on their **position** (left to right):

| Position | 1st | 2nd | 3rd | 4th | 5th | 6th |
|----------|-----|-----|-----|-----|-----|-----|
| **Points per card** | 1 | 2 | 3 | 4 | 5 | 6 |

**Example:**
- If you have 3 Red cards in position 1: 3 × 1 = 3 points
- If you have 5 Green cards in position 4: 5 × 4 = 20 points

### Cup Scoring
At game end, **reveal your Cup** cards.

For each card in your Cup:
- **If that color IS in your River:** Score points equal to that color's River position value
- **If that color is NOT in your River:** 0 points (worthless!)

**Example:** You have 4 Green cards in your Cup and Green is in River position 3. Score: 4 × 3 = 12 points.

**Strategy:** Collect many Cup cards in colors you've placed early in your River for maximum points.

### Final Score

**Total score = River points + Cup points**

**Highest score wins!**

**Tiebreaker:** If tied, player with more cards in Cup wins. If still tied, player with more River cards wins.

## Strategy Tips

1. **Control the colors:** Get colors into your Field early to control them
2. **Timing matters:** Sometimes it's better to let your opponent complete a Mandala
3. **River position:** Try to fill early positions first, but get high-quantity colors in later positions
4. **Cup insurance:** Use your Cup for backup colors you don't have in River
5. **Watch the deck:** When the deck is running low, plan your endgame carefully
6. **Mountain neutrality:** Cards in the Mountain can be claimed by either player - use this strategically

## Implementation Notes

This implementation uses:
- **Action encoding:** 30 possible actions (12 BUILD_MOUNTAIN + 12 GROW_FIELD + 6 DISCARD)
- **State representation:** 50-channel tensor encoding game state
- **MCTS:** 800 simulations per move for strong play
- **Neural network:** ResNet with policy and value heads

## Sources

- [Mandala on BoardGameGeek](https://boardgamegeek.com/boardgame/264241/mandala)
- [Official Rules - UltraBoardGames](https://www.ultraboardgames.com/mandala/game-rules.php)
- [Lookout Spiele Official Page](https://www.lookout-spiele.de/en/games/mandala.html)
- [Play Online - Board Game Arena](https://en.boardgamearena.com/gamepanel?game=mandala)
