#!/usr/bin/env python3
"""Minimal: play 5 games at supply=7, print all summary keys + argmax fields."""
import yaml, torch, numpy as np, mcts_cpp
import torch.nn.functional as F
from mandala_rl.network.model import MandalaNet
import pprint, sys

with open("/root/mandala-dom/configs/dominion.yaml") as f:
    cfg = yaml.safe_load(f)
nc = cfg["network"]
net = MandalaNet(input_channels=nc["input_channels"], num_actions=nc["num_actions"],
                 num_res_blocks=nc["num_res_blocks"], channels=nc["channels"],
                 belief_size=nc.get("belief_size", 31))
ckpt = torch.load("/workspace/dominion_data/checkpoints/model_latest.pt", map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
net.load_state_dict(state)
net = net.cuda().eval()
print(f"Checkpoint iter: {ckpt.get('iteration')}", flush=True)

mgr = mcts_cpp.BatchedMCTS(
    game_type="dominion", seed=42,
    num_simulations=800, c_puct=1.5,
    dirichlet_alpha=0.15, dirichlet_epsilon=0.15,
    temperature=1.0, temperature_threshold=25,
    explore_epsilon=0.0, leaves_per_game=8,
    action_explore_boost=0.0, action_buy_force_rate=0.0, action_play_force_rate=0.0,
    max_action_cards=0, big_money_force_rate=0.0,
    forced_kingdom_cards=[], disabled_basic_supply=[6,16],
    province_supply=7, max_turns=50)
mgr.init_games(5)

steps = 0
while not mgr.all_done():
    rt = mgr.begin_move()
    if len(rt) == 0: break
    with torch.no_grad():
        b = torch.from_numpy(np.stack(rt)).cuda()
        pols = F.softmax(net(b)[0], dim=1).cpu().numpy()
    mgr.set_root_policies(pols)
    for _ in range(800 // 8):
        lt = mgr.simulate_step()
        if len(lt) > 0:
            with torch.no_grad():
                b = torch.from_numpy(np.stack(lt)).cuda()
                _logits, _v, scores, _be = net(b)
                lp = F.softmax(_logits, dim=1).cpu().numpy()
                lv = torch.tanh(scores.squeeze(-1)).cpu().numpy()
            mgr.apply_nn_results(lp, lv)
    done = mgr.finish_move()
    for idx in done:
        s = dict(mgr.get_game_summary(idx))
        print(f"\n=== Game {idx} complete ===", flush=True)
        pprint.pprint(sorted(s.keys()))
        print(f"\nKey values:")
        for k in ['mcts_province_argmax_pct', 'mcts_province_decision_count',
                  'mcts_province_pct', 'province_buys', 'total_buys',
                  'total_coins_at_buy', 'turn_number', 'game_length']:
            print(f"  {k}: {s.get(k)}")
        sys.exit(0)
    steps += 1
    if steps > 100:
        print("Too many steps without a game completing", flush=True)
        break
