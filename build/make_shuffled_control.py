"""Second control (added 2026-11-11): the edit's own positions and the edit's own deltas, randomly re-assigned.

For each edited tensor: take the edit's flat indices, compute the edit's deltas (edited bf16 value - base value), permute
the deltas (seed 20261111), add them back at the same indices, and cast to bf16. Same weights touched, same multiset of
changes, random arrangement. If this also removes the loops, the edit's specific arrangement does not matter.
  LM_UNDERSTAND_LOGS=<logs clone> BASE_CKPT=<base ckpt dir> python3 make_shuffled_control.py OUT_DIR
"""
import json, os, shutil, sys
import numpy as np
LOGS = os.environ.get('LM_UNDERSTAND_LOGS', 'lm_understand-logs')
sys.path.insert(0, os.path.join(LOGS, 'exp8_math/code'))
import st_numpy as S

BASE_DIR = os.environ.get('BASE_CKPT', 'ckpts/base')
ED = os.path.join(LOGS, 'exp8_math/runs/p50c/edit')
OUT = sys.argv[1]
man = json.load(open(os.path.join(ED, 'manifest.json'))); z = np.load(os.path.join(ED, 'edit.npz'))
rng = np.random.default_rng(20261111)
tensors = S.load(os.path.join(BASE_DIR, 'model.safetensors'))
for i, t in enumerate(man['tensors']):
    dt, b = tensors[t['name']]
    flat = b.reshape(-1).copy()
    idx = z[f't{i}_idx'].astype(np.int64); val = z[f't{i}_val']
    base_f = S.bf16_to_f32(flat[idx].reshape(-1, 1)).reshape(-1)
    delta = S.bf16_to_f32(val.reshape(-1, 1)).reshape(-1) - base_f
    flat[idx] = S.bf16_bits(base_f + rng.permutation(delta))
    tensors[t['name']] = (dt, flat.reshape(b.shape))
    print(t['name'], 'positions', len(idx), 'delta std', float(delta.std()), 'mean', float(delta.mean()))
os.makedirs(OUT, exist_ok=True)
S.save(os.path.join(OUT, 'model.safetensors'), tensors, {'format': 'pt'})
for f in os.listdir(BASE_DIR):
    if f != 'model.safetensors':
        shutil.copyfile(os.path.join(BASE_DIR, f), os.path.join(OUT, f))
print('shuffled control ->', OUT)
