"""One-time surgical weight transplant: copy BUY[Gold] weight row to BUY[Province].

The network has unlearned Province buying over 50+ iterations of degraded play.
Gold's weight row encodes "buy expensive treasure when economy is ready" — the same
conditions under which Province should fire (8+ coins). We copy Gold's learned
pattern to Province and set a higher bias so Province is preferred over Gold when
both are affordable. Training will refine from here.
"""
import torch
import shutil
import torch.nn.functional as F

CKPT_PATH = '/workspace/dominion_data/checkpoints/model_latest.pt'
BACKUP_PATH = '/workspace/dominion_data/checkpoints/model_pre_weight_transplant.pt'

BUY_GOLD_IDX = 36      # BUY_OFFSET(34) + Gold card_id(2)
BUY_PROVINCE_IDX = 39  # BUY_OFFSET(34) + Province card_id(5)
PROVINCE_BIAS = 1.5     # Above Gold (-0.21) and Silver (+0.47) — Province is the best buy at 8+ coins
WEIGHT_SCALE = 1.1      # Slightly amplify Gold's pattern (Province needs stronger signal, costs more)

print('=== Province Weight Transplant ===')
print()

ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
print(f'Loaded checkpoint: iteration {ckpt.get("iteration", "?")}')

shutil.copy2(CKPT_PATH, BACKUP_PATH)
print(f'Backup saved to {BACKUP_PATH}')
print()

weight = sd['fc_policy.weight']  # (131, 2048)
bias = sd['fc_policy.bias']      # (131,)

# Before state
print('BEFORE:')
for name, idx in [('Silver', 35), ('Gold', 36), ('Duchy', 38), ('Province', 39), ('END_BUYS', 1)]:
    print(f'  {name:10s} idx={idx}: bias={bias[idx].item():+.6f}  weight_norm={weight[idx].norm().item():.2f}')
cos_before = F.cosine_similarity(weight[BUY_GOLD_IDX].unsqueeze(0), weight[BUY_PROVINCE_IDX].unsqueeze(0)).item()
print(f'  Gold-Province cosine similarity: {cos_before:.4f}')
print()

# Transplant: copy Gold's weight row to Province, scaled up
old_prov_weight = weight[BUY_PROVINCE_IDX].clone()
old_prov_bias = bias[BUY_PROVINCE_IDX].item()

weight[BUY_PROVINCE_IDX] = weight[BUY_GOLD_IDX].clone() * WEIGHT_SCALE
bias[BUY_PROVINCE_IDX] = PROVINCE_BIAS

sd['fc_policy.weight'] = weight
sd['fc_policy.bias'] = bias

# Reset Adam optimizer state for the Province weight row and bias
if 'optimizer_state_dict' in ckpt:
    opt = ckpt['optimizer_state_dict']
    for param_idx, state in opt['state'].items():
        # Reset bias Adam state (shape 131)
        if 'exp_avg' in state and state['exp_avg'].shape == torch.Size([131]):
            print(f'Resetting Adam state for fc_policy.bias[{BUY_PROVINCE_IDX}]')
            state['exp_avg'][BUY_PROVINCE_IDX] = 0.0
            state['exp_avg_sq'][BUY_PROVINCE_IDX] = 0.0
        # Reset weight Adam state (shape 131x2048)
        if 'exp_avg' in state and state['exp_avg'].shape == torch.Size([131, 2048]):
            print(f'Resetting Adam state for fc_policy.weight[{BUY_PROVINCE_IDX}]')
            state['exp_avg'][BUY_PROVINCE_IDX] = 0.0
            state['exp_avg_sq'][BUY_PROVINCE_IDX] = 0.0
    ckpt['optimizer_state_dict'] = opt
    print()

ckpt['model_state_dict'] = sd
torch.save(ckpt, CKPT_PATH)

# Verify
ckpt2 = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
w2 = ckpt2['model_state_dict']['fc_policy.weight']
b2 = ckpt2['model_state_dict']['fc_policy.bias']

print('AFTER:')
for name, idx in [('Silver', 35), ('Gold', 36), ('Duchy', 38), ('Province', 39), ('END_BUYS', 1)]:
    print(f'  {name:10s} idx={idx}: bias={b2[idx].item():+.6f}  weight_norm={w2[idx].norm().item():.2f}')
cos_after = F.cosine_similarity(w2[BUY_GOLD_IDX].unsqueeze(0), w2[BUY_PROVINCE_IDX].unsqueeze(0)).item()
print(f'  Gold-Province cosine similarity: {cos_after:.4f}')
print()

print(f'Summary:')
print(f'  Province bias:   {old_prov_bias:+.4f} -> {b2[BUY_PROVINCE_IDX].item():+.4f}')
print(f'  Province weight: copied from Gold row, scaled {WEIGHT_SCALE}x')
print(f'  Cosine sim:      {cos_before:.4f} -> {cos_after:.4f}')
print(f'  Adam state:      reset for Province weight row + bias')
print('Done.')
