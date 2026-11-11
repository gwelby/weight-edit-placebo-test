#!/usr/bin/env python3
"""Confirm that a checkpoint differs from its base only where an edit says (CPU only).
  python3 check_edit.py BASE_DIR OTHER_DIR EDIT_DIR
EDIT_DIR holds the authors' manifest.json and edit.npz (exp8_math/runs/p50c/edit). For each edited tensor it prints how
many weights differ between BASE and OTHER, and whether every difference lies at one of the edit's indices.
Edited checkpoint: 130,188 differences, all inside. Placebo: 130,741 differences, mostly outside (random columns)."""
import json, os, sys
import numpy as np
from safetensors import safe_open

base, other, ed = sys.argv[1:4]
man = json.load(open(os.path.join(ed, "manifest.json"))); z = np.load(os.path.join(ed, "edit.npz"))
total = 0
for i, t in enumerate(man["tensors"]):
    n = t["name"]
    with safe_open(os.path.join(base, "model.safetensors"), "pt") as fb, \
         safe_open(os.path.join(other, "model.safetensors"), "pt") as fo:
        b = fb.get_tensor(n).float().reshape(-1).numpy(); o = fo.get_tensor(n).float().reshape(-1).numpy()
    diff = np.nonzero(b != o)[0]; total += len(diff)
    inside = np.isin(diff, z[f"t{i}_idx"].astype(np.int64))
    print(f"{n}: {len(diff)} weights differ; {int(inside.sum())} at edit indices, {int((~inside).sum())} elsewhere")
print("total differing:", total)
