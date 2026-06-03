# PHASE_0C_DESIGN — investigation plan for the absolute accuracy gap

**Status:** ✅ RESOLVED (2026-05-29). See [REPORT_05.md](../reports/05_hardware_root_cause.md) for the full investigation writeup and the root-cause analysis.
**Outcome:** the 40pp gap was caused by P100 (Pascal, sm_60) fp16-accumulated-in-fp16 numerics silently collapsing RecursiveMAS Sequential-Light's deep recursive latent rollouts. Switching to A100 (Ampere, sm_80, Tensor Cores with fp32 accumulator) restores accuracy from 35% to 86% on math500. Hypotheses H1-H7 below are all FALSIFIED; H9 (test-time backprop, not in original list) also falsified; **H10 (hardware/precision) CONFIRMED**.
**Created:** 2026-05-27
**Resolved:** 2026-05-29
**Predecessors:** [REPORT_04.md](../reports/04_kaggle_p100_RETRACTED.md) (Phase 0.B closed with a 40pp gap from paper baseline)
**Successor:** [REPORT_05.md](../reports/05_hardware_root_cause.md) (root cause)

---

> The original plan below is preserved for the historical record. It described 8 hypotheses to bisect; reality required 2 we hadn't enumerated:
>
> - **H9 — Test-time backprop during recursive rounds.** Raised by the user. Falsified by grep of upstream `inference_mas.py` (zero `backward`/`optimizer`/`requires_grad=True` occurrences; all forwards in `@torch.no_grad()`).
> - **H10 — Hardware precision (Pascal fp16-accumulator vs Ampere Tensor Core fp32-accumulator).** Confirmed by Phase 0.E isolated ablation: A100 b=8 = 86% vs P100 b=8 = 35%, single-variable swap.

---

## 1. The gap we need to close

| | math500 accuracy |
|---|---|
| Paper RecursiveMAS Sequential-Light @ r=1 | **75.8%** |
| Our cleanest greedy baseline (n=100 shuf b=16 max_tok=2000) | **30.0%** |
| **Gap** | **−45.8 pp** |

This gap appears in **every baseline cell** we've run (no quantization) and is **independent of Variant B**. Per REPORT_04 §5.1, we've ruled out: `enable_thinking`, `max_new_tokens`, `batch_size`, subset selection bias, `num_recursive_rounds`. 

The remaining hypotheses are listed below. **Publication requires either matching the paper number or providing a controlled-ablation explanation for the gap.**

---

## 2. Hypotheses ranked

For each hypothesis we list: probability of being the cause, cost to test on our hardware, what passes/fails the test.

### H1 — `num_rollouts > 1` with self-consistency voting (PROBABILITY: HIGH)

Math reasoning papers commonly report pass@k with majority voting (3-5 sample rollouts per problem, majority vote on extracted answer). Boost is typically +10-20 pp on math500 for small models. `run.py` defaults to `--num_rollouts 1`, but the paper might have used voting silently.

- **Test cost:** ~1 cell at 3× inference time = ~3h GPU
- **Code change needed:** modify our baseline.py to (a) enable `--num_rollouts 3`, (b) verify the inference loop returns multiple answers per problem, (c) implement majority vote on extracted answers.
- **Passes** if accuracy lifts to ≥45-55% (would explain a big chunk of gap)
- **Fails** if no improvement (rule out)

### H2 — HF-released checkpoints ≠ paper-era checkpoints (PROBABILITY: MEDIUM)

Released models might be fine-tuned differently than what produced the paper numbers. Hard to verify without authors.

- **Test cost:** indirect via H4 (single-agent), or direct via GitHub issue (free, days to resolve)
- **Passes** if authors clarify, or if our single-agent Solver baseline matches Qwen2.5-Math-1.5B published numbers (rules OUT this hypothesis for the base model — but multi-agent training could still differ)

### H3 — single-agent baseline gap (PROBABILITY: MEDIUM-HIGH for diagnosis)

If we run JUST the Solver (Qwen2.5-Math-1.5B) on math500 with no multi-agent orchestration, does its accuracy match the **published Qwen2.5-Math-1.5B paper** number (~50-60% on math500)? If yes → our pipeline is broken/different from paper at the multi-agent level. If no → our checkpoint or general setup is the issue.

- **Test cost:** 1 cell ~30 min (no multi-agent, just one model)
- **Code change:** small new kernel script using only the Solver model with HF tokenizer.apply_chat_template + greedy generation
- **Passes (matches Qwen number)** → orchestration is the issue → focus on H1, H5
- **Fails (also low)** → the issue is upstream (release checkpoint differences, chat template, prompt engineering)

### H4 — different answer-extraction / comparison logic (PROBABILITY: MEDIUM)

The paper might use a more lenient comparator (latex-aware, semantic equivalence, etc.) than our strict boxed-answer-text-match. Our extractor lives in RecursiveMAS `inference_utils/answer_utils.py:extract_boxed_answer + compare_answers` — we delegate to it, so we ARE using RecursiveMAS's own comparator. But the paper might post-process differently.

- **Test cost:** ~30 min reading code + checking a sample of "wrong" answers manually
- **Code change:** none
- **Passes (lenient comparator gives much higher)** → re-evaluate with proper comparator
- **Fails (current comparator is the right one)** → rule out

### H5 — chat template / system prompt mismatch (PROBABILITY: LOW-MEDIUM)

Paper might use a math-specific system prompt or chat template variant. Our pipeline uses `prompts.py:SYSTEM_PROMPT = "You are a helpful assistant."` (very generic).

- **Test cost:** ~30 min reading paper appendix + RecursiveMAS prompts.py; test by changing system prompt and re-running 1 cell
- **Code change:** new kernel with modified prompts.py wrapper
- **Passes** if a different prompt lifts accuracy significantly
- **Fails** if no change

### H6 — N=500 vs N=100 variance (PROBABILITY: LOW for the full gap, MEDIUM contribution)

Sample variance at n=100 is ±5pp SE, but between-subset differences can be larger (~10-15pp) if math500 has clusters of similar problems. Paper averages over all 500; we sample 100. Could account for ~5-15pp of the gap.

- **Test cost:** ~2-3 hours GPU for 1 cell at n=300, or ~4-5 hours at n=500 (batch=8 only since batch=16 OOM borderline)
- **Code change:** none, just change N_SAMPLES
- **Passes** if accuracy with n=300+ stabilizes substantially higher than n=100
- **Fails** if accuracy stays around 30-34% regardless of N (confirming N=100 was already representative)

### H7 — `batch_size=32` (paper) vs 16 (us) interacts with sampling RNG (PROBABILITY: LOW)

Different batch sizes can give different generations even at fixed seed if RNG isn't perfectly batch-invariant. Effect typically 1-3pp.

- **Test cost:** need bigger GPU (≥24 GB) for batch=32; A100 on Modal/RunPod ~$1.50/h × 1h = $1.50
- **Code change:** none, just bump BATCH_CAP
- **Passes** if batch=32 lifts accuracy 5-10pp (unlikely)
- **Fails** if same as batch=16 (likely)

### H8 — Sequential-Scaled instead of Sequential-Light (PROBABILITY: MEDIUM for narrowing gap)

Paper reports both. Sequential-Scaled uses Qwen3.5-4B + Llama3.2-3B + Gemma3-4B (~11 GB models total in fp16). Paper might report higher numbers for Scaled. We've been running Light because of disk/GPU budget.

- **Test cost:** needs ~12 GB models loaded + activations → likely needs A100 24 GB
- **Code change:** swap `style="sequential_light"` to `"sequential_scaled"`
- **Passes** if accuracy is significantly higher (and matches paper's Scaled number)
- **Fails** if same gap appears at Scaled too (rules out small-model explanation)

---

## 3. Investigation order

Each step is a stop-condition: if it explains a big chunk of the gap, we may not need the next steps.

### Step 1 — Free diagnostics (~30 min, no Kaggle compute)
- Re-read `external/RecursiveMAS/inference_utils/inference_mas.py` argparse defaults for any flag we missed (especially num_rollouts handling)
- Re-read paper appendix B.3 and Section 5 for explicit setup we haven't reproduced
- Manually inspect 5 wrong-answer cases in our greedy baseline log: are answers really wrong, or extractor missing a correct boxed number?
- Open issue on `github.com/RecursiveMAS/RecursiveMAS` asking the authors for exact reproduction recipe (likely 1-7 day response)

### Step 2 — H3 single-agent Solver baseline (~30 min Kaggle)
1 cell: load only Solver Qwen2.5-Math-1.5B, run greedy on math500 n=100 shuffled. Compare to published Qwen-Math paper number.
- If 50-60%: our generation is correct, gap is in multi-agent
- If <40%: gap is upstream of multi-agent

### Step 3 — H1 self-consistency voting (~3h Kaggle)
1 cell: baseline with `num_rollouts=3` + majority vote in post-processing. Compare to single-rollout baseline.
- If +10-20pp: explains a big chunk of gap, document this
- If no change: rule out

### Step 4 — H4 answer comparator audit (~30 min local)
Sample 10 of our "wrong" baseline answers, manually verify against gold answer with a strict and a permissive comparator. If many are technically correct but our extractor missed them → fix the extractor and re-evaluate.

### Step 5 — H6 N=300 cell (~3h Kaggle)
1 cell: baseline at n=300 shuffled to reduce sample variance. If accuracy is dramatically different from n=100, sample variance is a bigger driver than we thought.

### Step 6 — H8 Sequential-Scaled (REQUIRES GPU UPGRADE)
Needs A100 24+ GB. Test on Modal/RunPod ($1.50-3 one-shot).
- If Scaled-baseline matches paper's Scaled number: pipeline is correct, just our scale is too small
- If Scaled-baseline also has 40pp gap: deeper issue

### Step 7 — H7+H6 full N=500 batch=32 reproduction (REQUIRES GPU UPGRADE)
Definitive test. Needs A100 40 GB. ~5-10 GPU-hours = $15-30 on Modal.
- If our baseline matches 75.8%: pipeline + checkpoint correct, our P100 results were just sample-noisy
- If our baseline is still 30-50%: confirm gap is deeper (checkpoint version, paper underspecified)

---

## 4. Stopping rules

We stop investigating and write the publication when one of these is true:

1. **Reproduction succeeded** — Some combination of H1+H6+H8 brings baseline to within ±5pp of paper. Then we re-run our Variant B cells under the matching setup and report.
2. **Diagnostic explanation found** — Single-agent Solver baseline + ablation of one specific flag explains the gap mechanistically (e.g., "paper used num_rollouts=3", document with controlled ablation).
3. **Hard stop: GitHub authors' response** — If RecursiveMAS authors confirm a flag/setup we missed, fix and re-run. If they confirm our reproduction matches their numbers within sample variance, the publication target was wrong, write what we have.
4. **Budget exhausted** — If after spending ~$30 on cloud GPU we can't close gap, document the gap and the ruled-out hypotheses as an honest negative result.

---

## 5. Resource requirements

| Step | Kaggle GPU-h | Cloud $ | Real $ |
|---|---|---|---|
| 1 (read + issue) | 0 | 0 | 0 |
| 2 (single-agent) | 0.5 | 0 | 0 |
| 3 (num_rollouts=3) | 3 | 0 | 0 |
| 4 (comparator audit) | 0 | 0 | 0 |
| 5 (N=300) | 3 | 0 | 0 |
| 6 (Sequential-Scaled) | 0 | A100 1h | ~$2-3 |
| 7 (full N=500 batch=32 A100) | 0 | A100 10h | ~$15-30 |
| **Total** | **~6.5 GPU-h** | A100 ~11h | **~$20-35** |

Step 1-5 fit in current Kaggle weekly quota (30 GPU-h). Steps 6-7 require one-shot cloud spending under $35.

---

## 6. Probability-weighted outcome forecast

Assuming we run steps 1-5 (no cloud) and then decide on 6-7:

- **45%** — H1 (num_rollouts) explains most of gap → clean reproduction, easy publication
- **25%** — H3 (single-agent) shows pipeline-level issue → debug-publish path
- **15%** — H4 (comparator) shows accounting issue → fix extractor, re-evaluate everything
- **10%** — Authors' response solves it
- **5%** — All steps fail to close gap → submit as "negative" / methods write-up

In any outcome, our Variant B claim from REPORT_04 remains valid — we'd just be repositioning the framing of the baseline reproduction.

---

## 7. What to do TODAY (proposal)

Two-step batch, minimum risk:

1. **Manual diagnostic (~30 min, you + me)**:
   - I re-read the paper Section 5 + appendix B.3 carefully
   - I sample 10 wrong-answer cases from our greedy baseline log and analyze them
   - I check if `inference_mas.py` has any rollout logic we haven't enabled

2. **Launch H3 (single-agent Solver baseline, ~30 min)** as a Kaggle cell:
   - If Solver alone gives 50-60% on math500 → orchestration is the issue, run H1 next
   - If Solver alone gives 30% → setup issue, dig into H4/H5

This isolates "is the bug in the multi-agent dance, or in our overall setup" with one cheap experiment.

After we see those results, propose H1 (num_rollouts) which is the most likely systemic fix.
