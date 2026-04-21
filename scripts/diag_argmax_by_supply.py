#!/usr/bin/env python3
"""
Diagnostic: measure mcts_province_argmax_pct at varying province_supply sizes
using the current checkpoint. No effect on training (runs in a separate process,
optionally on CPU to avoid GPU contention).

Theory being tested: mcts_argmax% is structurally bounded by province_supply
(98% at supply=1 due to sharp terminal value, ~58% at supply=7 due to Q-gap
clustering). If confirmed, Q-gap-widening interventions (more sims, Q-based
targets) are the right class of fix. If flat across supply, weights are broken
and different approach needed.

Usage: python scripts/diag_argmax_by_supply.py [--device cpu|cuda] [--games N]
"""
import argparse
import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

import mcts_cpp
from mandala_rl.network.model import MandalaNet


def run_supply(supply: int, cfg: dict, net: MandalaNet, device: str,
               num_games: int, seed: int, leaf_eval_source: str = "score") -> dict:
    """Play num_games at given province_supply, return aggregated stats.

    leaf_eval_source: "score" (tanh'd score head, default) or "value" (value head,
    already tanh'd by the model). Determines the signal fed to MCTS leaf eval.
    """
    mcts_cfg = cfg["mcts"]
    sp_cfg = cfg["selfplay"]

    mgr = mcts_cpp.BatchedMCTS(
        game_type="dominion",
        seed=seed,
        num_simulations=mcts_cfg["num_simulations"],
        c_puct=mcts_cfg["c_puct"],
        dirichlet_alpha=mcts_cfg["dirichlet_alpha"],
        dirichlet_epsilon=mcts_cfg["dirichlet_epsilon"],
        temperature=sp_cfg["temperature"],
        temperature_threshold=sp_cfg["temperature_threshold"],
        explore_epsilon=0.0,
        leaves_per_game=sp_cfg["leaves_per_game"],
        action_explore_boost=mcts_cfg.get("action_explore_boost", 0.0),
        action_buy_force_rate=mcts_cfg.get("action_buy_force_rate", 0.0),
        action_play_force_rate=mcts_cfg.get("action_play_force_rate", 0.0),
        max_action_cards=cfg.get("max_action_cards", 0),
        big_money_force_rate=cfg.get("big_money_force_rate", 0.0),
        forced_kingdom_cards=cfg.get("forced_kingdom_cards", []) or [],
        disabled_basic_supply=cfg.get("disabled_basic_supply", []) or [],
        province_supply=supply,
        max_turns=cfg.get("max_turns", 50),
    )
    mgr.init_games(num_games)

    sim_steps = max(1, mcts_cfg["num_simulations"] // sp_cfg["leaves_per_game"])
    summaries = []
    move_count = 0

    while not mgr.all_done():
        root_tensors = mgr.begin_move()
        if len(root_tensors) == 0:
            break
        with torch.no_grad():
            batch = torch.from_numpy(np.stack(root_tensors)).to(device)
            logits = net(batch)[0]
            policies = F.softmax(logits, dim=1).cpu().numpy()
        mgr.set_root_policies(policies)

        for _ in range(sim_steps):
            leaf_tensors = mgr.simulate_step()
            if len(leaf_tensors) > 0:
                with torch.no_grad():
                    batch = torch.from_numpy(np.stack(leaf_tensors)).to(device)
                    logits, vals, scores, _beliefs = net(batch)
                    pols = F.softmax(logits, dim=1).cpu().numpy()
                    if leaf_eval_source == "value":
                        vals_np = vals.squeeze(-1).cpu().numpy()
                    else:
                        vals_np = torch.tanh(scores.squeeze(-1)).cpu().numpy()
                mgr.apply_nn_results(pols, vals_np)

        done_indices = mgr.finish_move()
        for idx in done_indices:
            summaries.append(dict(mgr.get_game_summary(idx)))
        move_count += 1
        if move_count % 10 == 0:
            print(f"  [supply={supply}] move {move_count}, games done: {len(summaries)}/{num_games}",
                  flush=True)

    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/mandala-dom/configs/dominion.yaml")
    ap.add_argument("--checkpoint", default="/workspace/dominion_data/checkpoints/model_latest.pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--games", type=int, default=15, help="games per supply level")
    ap.add_argument("--supplies", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sims", type=int, default=None, help="override num_simulations from config")
    ap.add_argument("--leaf-eval", default="score", choices=["score", "value"],
                    help="MCTS leaf eval source: score head (current) or value head (proposed fix)")
    ap.add_argument("--compare-modes", action="store_true",
                    help="run score AND value modes back-to-back and print A/B table")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.sims is not None:
        cfg["mcts"]["num_simulations"] = args.sims

    # Load network
    net_cfg = cfg["network"]
    net = MandalaNet(
        input_channels=net_cfg["input_channels"],
        num_actions=net_cfg["num_actions"],
        num_res_blocks=net_cfg["num_res_blocks"],
        channels=net_cfg["channels"],
        belief_size=net_cfg.get("belief_size", 31),
        phase_aware_policy=net_cfg.get("phase_aware_policy", False),
        factored_policy=net_cfg.get("factored_policy", False),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    net.load_state_dict(state)
    net = net.to(args.device).eval()

    print(f"Loaded: {args.checkpoint} (iter={ckpt.get('iteration', '?')})")
    print(f"Config: num_sims={cfg['mcts']['num_simulations']}, "
          f"eps={cfg['mcts']['dirichlet_epsilon']}, "
          f"alpha={cfg['mcts']['dirichlet_alpha']}, "
          f"disabled={cfg.get('disabled_basic_supply', [])}")
    print(f"Device: {args.device}, games/supply: {args.games}")
    print()

    def sweep(leaf_eval_source: str) -> list:
        print(f"\n##### leaf_eval_source = {leaf_eval_source} #####", flush=True)
        rows = []
        for supply in args.supplies:
            t0 = time.time()
            print(f"=== Running supply={supply} (leaf={leaf_eval_source}) ===", flush=True)
            summaries = run_supply(supply, cfg, net, args.device, args.games,
                                   args.seed + supply, leaf_eval_source=leaf_eval_source)
            dur = time.time() - t0

            argmax_pcts, visit_pcts, provs, turns = [], [], [], []
            for s in summaries:
                visit_pct = s.get("mcts_province_pct", 0) or 0
                argmax_pct = s.get("mcts_province_argmax_pct", 0) or 0
                if visit_pct > 0 or argmax_pct > 0:
                    argmax_pcts.append(argmax_pct)
                    visit_pcts.append(visit_pct)
                provs.append(sum(s.get("province_buys", (0, 0))))
                turns.append(s.get("turn_number", 0))

            row = {
                "leaf_eval_source": leaf_eval_source,
                "supply": supply,
                "games": len(summaries),
                "games_with_prov_decisions": len(argmax_pcts),
                "argmax_pct_mean": float(np.mean(argmax_pcts) * 100) if argmax_pcts else 0.0,
                "argmax_pct_std": float(np.std(argmax_pcts) * 100) if argmax_pcts else 0.0,
                "visit_pct_mean": float(np.mean(visit_pcts) * 100) if visit_pcts else 0.0,
                "avg_provinces": float(np.mean(provs) / 2) if provs else 0.0,
                "avg_turns": float(np.mean(turns)) if turns else 0.0,
                "duration_s": round(dur, 1),
            }
            rows.append(row)
            print(f"  argmax%={row['argmax_pct_mean']:.1f} ± {row['argmax_pct_std']:.1f}, "
                  f"visit%={row['visit_pct_mean']:.1f}, "
                  f"games_with_$8+={row['games_with_prov_decisions']}/{row['games']}, "
                  f"avg_prov={row['avg_provinces']:.2f}, "
                  f"duration={row['duration_s']}s", flush=True)
        return rows

    if args.compare_modes:
        score_rows = sweep("score")
        value_rows = sweep("value")
        all_rows = score_rows + value_rows

        # A/B comparison table
        print("\n" + "=" * 90)
        print("A/B COMPARISON: MCTS leaf eval source (same checkpoint, same sims, same noise)")
        print("=" * 90)
        print(f"{'supply':>6} {'score argmax%':>14} {'±std':>6} | {'value argmax%':>14} {'±std':>6} | {'Δ (val−sco)':>12}")
        print("-" * 90)
        for sr, vr in zip(score_rows, value_rows):
            delta = vr['argmax_pct_mean'] - sr['argmax_pct_mean']
            print(f"{sr['supply']:>6} "
                  f"{sr['argmax_pct_mean']:>14.1f} {sr['argmax_pct_std']:>6.1f} | "
                  f"{vr['argmax_pct_mean']:>14.1f} {vr['argmax_pct_std']:>6.1f} | "
                  f"{delta:>+12.1f}")
        print("=" * 90)
        print("\nGate (per plan): value-mode argmax% must beat score-mode by ≥8 points at supply=5 AND 7.")
        critical = [(s, v) for s, v in zip(score_rows, value_rows) if s['supply'] in (5, 7)]
        if all(v['argmax_pct_mean'] - s['argmax_pct_mean'] >= 8.0 for s, v in critical):
            print("RESULT: GATE PASSED — structural fix confirmed. Flip leaf_eval_source to 'value' in config.")
        elif all(v['argmax_pct_mean'] - s['argmax_pct_mean'] >= 0.0 for s, v in critical):
            print("RESULT: directional but below gate. Consider more games or investigate Q distribution.")
        else:
            print("RESULT: GATE FAILED — value-mode does not improve argmax%. Hypothesis refuted.")
    else:
        all_rows = sweep(args.leaf_eval)
        print("\n" + "=" * 80)
        print(f"{'supply':>6} {'games':>6} {'with_$8+':>10} {'argmax%':>9} {'±std':>6} {'visit%':>7} {'avg_prov':>9}")
        print("-" * 80)
        for r in all_rows:
            print(f"{r['supply']:>6} {r['games']:>6} {r['games_with_prov_decisions']:>10} "
                  f"{r['argmax_pct_mean']:>9.1f} {r['argmax_pct_std']:>6.1f} "
                  f"{r['visit_pct_mean']:>7.1f} {r['avg_provinces']:>9.2f}")
        print("=" * 80)

    out = Path("/workspace/dominion_data/diag_argmax_by_supply.json")
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "iteration": ckpt.get("iteration"),
        "config_sims": cfg["mcts"]["num_simulations"],
        "config_eps": cfg["mcts"]["dirichlet_epsilon"],
        "compare_modes": args.compare_modes,
        "rows": all_rows,
    }, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
