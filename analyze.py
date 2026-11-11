#!/usr/bin/env python3
"""Recompute every number in the paper's Section 4 from the four result files alone.
Usage: python3 analyze.py [results_dir]   (standard library only)"""
import json, sys, math, zlib, hashlib, statistics as st, os

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARMS = [("Qwen3.5-0.8B base", "qwen35_0p8b_base.jsonl"),
        ("Qwen3.5-0.8B placebo", "qwen35_0p8b_placebo.jsonl"),
        ("Qwen3.5-0.8B edited", "qwen35_0p8b_edited_p50c.jsonl"),
        ("Gemma 4 E2B base", "gemma4_e2b_it_base.jsonl")]
CAP = 32768


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def fisher(a, b, n1, n2):
    """Two-sided Fisher exact p for k=a of n1 vs k=b of n2."""
    K, N = a + b, n1 + n2
    pr = lambda x: math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, K - n2), min(K, n1) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def mcnemar(x, y):
    """Paired by problem id: (only x correct, only y correct, exact two-sided p)."""
    b = sum(1 for i in x if x[i]["correct"] and not y[i]["correct"])
    c = sum(1 for i in x if not x[i]["correct"] and y[i]["correct"])
    n, k = b + c, min(b, c)
    return b, c, (min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)


def zratio(t):
    b = t.encode()
    return len(zlib.compress(b)) / max(1, len(b))


R = {}
print("== Files")
for name, f in ARMS:
    path = os.path.join(D, f)
    print(f"{name:22} {f:34} sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
    R[name] = {r["idx"]: r for r in (json.loads(l) for l in open(path) if l.strip())}
ids = set(R[ARMS[0][0]])
assert all(set(v) == ids for v in R.values()) and len(ids) == 50, "arms must cover the same 50 problems"

print("\n== Table 1: accuracy, cap rate, tokens (95% Wilson intervals)")
for name, _ in ARMS:
    rs = list(R[name].values()); n = len(rs)
    k = sum(bool(r["correct"]) for r in rs); c = sum(bool(r["cut"]) for r in rs)
    tok = sum(r["n_tokens"] for r in rs); capped_tok = sum(r["n_tokens"] for r in rs if r["cut"])
    print(f"{name:22} correct {k}/{n} ({100*k/n:.0f}%, {wilson(k,n)[0]:.0f}-{wilson(k,n)[1]:.0f})  "
          f"capped {c}/{n} ({100*c/n:.0f}%, {wilson(c,n)[0]:.0f}-{wilson(c,n)[1]:.0f})  "
          f"median tok {st.median(r['n_tokens'] for r in rs):.0f}  max {max(r['n_tokens'] for r in rs)}  "
          f"tokens/correct {tok/max(k,1):,.0f}  share of tokens in capped runs {100*capped_tok/tok:.0f}%")

print("\n== Table 2: paired comparisons (same 50 problems)")
b, ctl, ed = (R[a[0]] for a in ARMS[:3])
for lab, x, y in [("edited vs base", ed, b), ("edited vs placebo", ed, ctl), ("placebo vs base", ctl, b)]:
    f_, g_, p = mcnemar(x, y)
    kx = sum(bool(r["correct"]) for r in x.values()); ky = sum(bool(r["correct"]) for r in y.values())
    cx = sum(bool(r["cut"]) for r in x.values()); cy = sum(bool(r["cut"]) for r in y.values())
    print(f"{lab:18} fixed {f_} broken {g_}  McNemar p {p:.2g}  Fisher correct p {fisher(kx, ky, 50, 50):.2g}  "
          f"Fisher capped p {fisher(cx, cy, 50, 50):.2g}")
print("capped in base but finished when edited:", sum(1 for i in ids if b[i]["cut"] and not ed[i]["cut"]),
      "| finished in base but capped when edited:", sum(1 for i in ids if not b[i]["cut"] and ed[i]["cut"]))

print("\n== Table 3: compressibility of the last 400 characters (zlib ratio; lower = more repetitive)")
for name, _ in ARMS:
    rs = list(R[name].values())
    out = []
    for lab, sel in (("capped", [r for r in rs if r["cut"]]), ("finished", [r for r in rs if not r["cut"]])):
        out.append(f"{lab} " + (f"median {st.median(zratio(r['text_tail']) for r in sel):.2f}, below 0.35: "
                                f"{sum(zratio(r['text_tail']) < 0.35 for r in sel)}/{len(sel)}" if sel else "- (n=0)"))
    print(f"{name:22} " + "   ".join(out))
