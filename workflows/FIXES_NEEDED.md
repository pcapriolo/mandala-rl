# Critical Fixes Needed for Flyte Workflow

## 1. Fix evaluate_checkpoint (CRITICAL)

The function calls methods that don't exist. Replace evaluation logic:

```python
# OLD (BROKEN):
action, _ = new_worker.get_action(game_state)

# NEW (FIXED):
from mandala_rl.mcts.search import MCTS

new_mcts = MCTS(game, new_network.predict, mcts_simulations, c_puct)
prev_mcts = MCTS(game, prev_network.predict, mcts_simulations, c_puct)

# In game loop:
canonical_state = game_state.get_canonical_form()
if current_player == 1:
    action_probs, _ = new_mcts.get_action_prob(canonical_state, temperature=0, add_noise=False)
else:
    action_probs, _ = prev_mcts.get_action_prob(canonical_state, temperature=0, add_noise=False)
action = np.argmax(action_probs)
```

## 2. Fix Game API calls

Check if these methods exist in MandalaGame:
- `game.get_current_player(state)` → might be `state.current_player`
- `game.get_result(state, player)` → might be `game.get_reward(state, player)`

## 3. Disable cache for stochastic tasks

```python
@task(cache=False)  # Changed from cache=True
def generate_games_from_input(...)
```

## 4. Fix aggregate_games worker creation

```python
# Remove SelfPlayWorker - just use static logic:
all_examples = []
for game_record in all_games:
    outcome = game_record.outcome
    for i, (state, policy, player) in enumerate(zip(
        game_record.states,
        game_record.policies,
        game_record.current_players
    )):
        value = outcome if player == 0 else -outcome
        all_examples.append((state, policy, value))
```

## 5. Use correct workflow for webhook

Your webhook should call `run_distributed_selfplay`, NOT `full_training_iteration`.

The callback expects:
```json
{
  "outputs": {
    "num_games": 4,
    "num_examples": 187,
    "examples": [...]
  }
}
```

But `run_distributed_selfplay` returns just a FlyteFile with examples.

You need to wrap it:

```python
@workflow
def distributed_selfplay_for_webhook(...) -> dict:
    examples_file = run_distributed_selfplay(...)

    # Download and count
    examples_path = examples_file.download()
    with open(examples_path, 'rb') as f:
        examples = pickle.load(f)

    # Convert to serializable format
    serializable_examples = []
    for state, policy, value in examples:
        serializable_examples.append({
            'state': state.tolist() if hasattr(state, 'tolist') else state,
            'policy': policy.tolist() if hasattr(policy, 'tolist') else policy,
            'value': float(value)
        })

    return {
        'num_games': total_games,  # Need to track this
        'num_examples': len(examples),
        'examples': serializable_examples
    }
```
