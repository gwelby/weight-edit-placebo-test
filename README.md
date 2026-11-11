# A Placebo Test for Agent-Written Weight Edits: resource

The results and scripts behind the paper of the same name (Kaggle Gemma 4 Developer Agent Paper Track, November 2026).
Our files are Apache 2.0 (`LICENSE`). `third_party/` holds two MIT-licensed scripts by the authors of
`RikiyaT/lm_understand-logs`, with their licence (`third_party/LICENSE-MIT-lm_understand-logs`).

## Check every number in the paper (seconds, no GPU)

```
python3 analyze.py
```
Standard library only. It prints each result file's sha256 and recomputes §4: cap rate and accuracy with 95% Wilson
intervals, tokens per correct answer, Fisher and exact McNemar tests paired by problem, and the compressibility of the
last 400 characters of each output.

| File | Arm | sha256 |
|---|---|---|
| `results/qwen35_0p8b_base.jsonl` | Qwen3.5-0.8B (text part), unedited | `b45d768ab90159d3cd241406b2b1feae2f83674bccd94b8bb3c990e33612af4c` |
| `results/qwen35_0p8b_placebo.jsonl` | random columns, Gaussian noise at the edit's scale | `bd6cc63a92a345a69518ebd43886af649b37b24c7eb1cd1b87680eb7605b4b9c` |
| `results/qwen35_0p8b_edited_p50c.jsonl` | the authors' published 50-pass edit | `c17be466d0cc30245755748e7168136c0b978d1ae7d84270228c3af33d4261fc` |
| `results/gemma4_e2b_it_base.jsonl` | Gemma 4 E2B-it, unedited | `5d09544103b3b2302655a4214113f3138fda0dc49cae23a99d2183a9493a96e1` |

One line per problem: `idx` (MATH-500 test index), `sample`, `n_tokens`, `cut` (reached the 32,768-token cap),
`correct`, `gold`, `text_tail` (last 400 characters of the output).

## Re-run the experiment

**Inputs (not redistributed):**
- `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`;
- `RikiyaT/lm_understand-logs` at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, for the edit
  (`exp8_math/runs/p50c/edit`), the text-model config files and `exp8_math/code/build_ckpt.py` (logs and data CC BY 4.0,
  scripts MIT);
- `HuggingFaceH4/MATH-500` (revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`); the 50 problems are listed by index in
  `subset_math500_ids.json`, and the problem text is not included here;
- `unsloth/gemma-4-E2B-it-GGUF` at commit `0314792d7f1f7e229411f620751375812bb9faf2`: `gemma-4-E2B-it-BF16.gguf`
  (sha256 `1eafd61d010ce8ca09db38f370aadd64c6d792db269c365ad0d9ea2709701890`) and the draft model
  `MTP/mtp-gemma-4-E2B-it-BF16.gguf` (sha256 `496d95c94c7b8b367fa7be722783d9fe14e458b92e60792ca690c4fb4fc63425`);
  `google/gemma-4-E2B-it` for its tokenizer and chat template.

```
export LM_UNDERSTAND_LOGS=/path/to/lm_understand-logs
export PYTHONPATH=$PWD/third_party        # mathqa.py (grader, needs math-verify) and st_numpy.py
```

| Step | Command |
|---|---|
| 1. Select the 50 problems (seed 42, 10 per level); compare with `subset_math500_ids.json` | `python3 build/make_subset.py` |
| 2. Text-only Qwen base checkpoint | `QWEN_SNAPSHOT=<snapshot dir> python3 build/make_text_ckpt.py ckpts/base` (adapted from the authors' recipe) |
| 3. Edited checkpoint | the authors' `exp8_math/code/build_ckpt.py` with `runs/p50c/edit` |
| 4. Placebo checkpoint | `BASE_CKPT=ckpts/base python3 build/make_control_edit.py`, then `python3 build/apply_control.py` (tensor names and `n_cols=64` are set for this edit) |
| 5. Check a checkpoint against the edit | `python3 check_edit.py ckpts/base <ckpt> $LM_UNDERSTAND_LOGS/exp8_math/runs/p50c/edit` (edited: 130,188 differences, all at edit indices; placebo: 130,741) |
| 6. Evaluate a Qwen arm (transformers) | `python3 harness/eval_run_gpu.py <ckpt> subset_math500.jsonl out.jsonl --max_tokens 32768 --dtype float16 --samples 1 --batch 1` |
| 7. Evaluate Gemma (koboldcpp 1.123.1 server) | serve: `koboldcpp gemma-4-E2B-it-BF16.gguf --port PORT --gpulayers 999 --contextsize 40000 --jinja --usemtp --draftmodel mtp-gemma-4-E2B-it-BF16.gguf --flashattention`; then `python3 harness/eval_run_gguf.py <gemma tokenizer dir> subset_math500.jsonl out.jsonl --server http://127.0.0.1:PORT` |

Both harnesses send temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, thinking on, at most 32,768 new tokens,
one sample per problem. They are different implementations: the transformers harness applies its own presence-penalty
processor and seeds each problem; the llama.cpp server applies its own penalty, may ignore top-k, and chooses its own seed.

`build/make_shuffled_control.py` builds a second, structured placebo (the edit's own deltas, shuffled within the edit's
own positions). It was added after the paper's runs; see the paper for whether its arm was evaluated.

**Changes from the scripts as run:** machine-specific absolute paths were replaced with the environment variables and
relative paths above; nothing else. `ORIGINAL_SCRIPTS.sha256` lists each script's hash as it ran.
