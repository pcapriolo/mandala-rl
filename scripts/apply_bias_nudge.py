import torch

CKPT = "/workspace/dominion_data/checkpoints/model_latest.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
bias = ckpt["model_state_dict"]["fc_policy.bias"]
old = bias[39].item()
bias[39] = 0.5
ckpt["model_state_dict"]["fc_policy.bias"] = bias

if "optimizer_state_dict" in ckpt:
    for pi, state in ckpt["optimizer_state_dict"]["state"].items():
        if "exp_avg" in state and state["exp_avg"].shape == torch.Size([131]):
            state["exp_avg"][39] = 0.0
            state["exp_avg_sq"][39] = 0.0
            print("Reset Adam bias state")

torch.save(ckpt, CKPT)

ckpt2 = torch.load(CKPT, map_location="cpu", weights_only=False)
b2 = ckpt2["model_state_dict"]["fc_policy.bias"]
print(f"Province bias: {old:+.4f} -> {b2[39].item():+.4f}")
print(f"Iter: {ckpt2.get('iteration', '?')}")
