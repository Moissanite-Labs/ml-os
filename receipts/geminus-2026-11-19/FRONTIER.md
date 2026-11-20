# Frontier evaluation

A running assessment of where this work has reached new ground, what is solid, what is only
promising, and what turned out to be wrong. Last revised 2026-11-19 20:49 local, when the round-2
review corrections were applied to the receipts and to this file (a third, narrow review of the
changes is recorded in `receipts\REVIEW_ROUND_3.md`).
Every number comes from a result file in `m1\out\`; the scripts are in `m1\scripts\`.

Scope of all measurements: one model (Qwen3.5-4B, Q8_0 GGUF), one runtime (llama.cpp b12549,
CUDA, RTX 3080 Laptop), one compile recipe (mcproof P2-B), formal-verification prompts for the
live tests, one train/held-out split. Nothing has been repeated on a second model or runtime.

"Held-out" throughout means held-out prompt ids inside the same four template families. 98% of
held-out states reuse a (positive token, negative set) pair seen in training. It is a weak split.

## How each item is graded

- **Evidence**: *measured* (direct, reproducible here), *derived* (follows from the equations plus
  measured inputs), *inferred* (best explanation, not isolated experimentally).
- **Novelty**: *project* (new and actionable for this team), *field* (I am not aware of it being
  stated or measured elsewhere; my knowledge has gaps exactly where work is recent), *known*
  (standard principle, restated or re-measured).

## 1. Established here

| # | Finding | Evidence | Novelty |
|---|---|---|---|
| E1 | Greedy text cannot tell whether an adapter was applied on this runtime; log-probability margins detect an adapter-sized shift on every request (on the test prompt, the formal-verification adapter's shift only). | measured at server level, two runs | project; the test recipe is of general use |
| E2 | Q8_0 matmuls quantise their input activations to 8 bits on this backend. | in the source, measured for six matmuls; link to batch noise inferred and incomplete | known to runtime authors; size and consequence measured here |
| E3 | The per-layer pass-through model fails for blk.28-30. | measured, two methods that agree in direction | project |
| E4 | On radial shrinkage a uniform multi-layer split is worse than the last layer alone. | measured live, one operating point | project |
| E5 | Position-local preservation is exact only for last-layer edits. | measured; mechanism reasoned | field, as a statement about what a per-key certificate bounds |
| E6 | Preconditioning the target residual by the gate metric halves radial shrinkage at equal gain. It also raises off-domain KL. | derived, then measured at three stages | known principle, new application; mixed result |
| E7 | Three upstream issues in the team's pipeline. | measured / inferred | project |
| E8 | A last-layer FFN adapter, verified to run on the prefix, rolls back exactly with no cache flush; earlier-layer adapters leave a trace. | derived, then measured (corrected probe) | project |

### E1. Text gates are unsound; an adapter-sized shift reached every request

The swarm held a receipt because 2 of 8 concurrent requests returned baseline-looking text.

Server level, through llama-server's per-request `lora` path, reading log-probabilities at the
token where outputs fork. Two runs, same rule fixed before the run:

- Run 1: 84 of 84 adapter requests fall in the adapter cluster (margin -0.13 to +0.19); 36 of 36
  baseline requests in the baseline cluster (-1.01 to -0.76); clusters 0.64 logits apart; 33 of
  the 84 adapter requests still chose the baseline-like token.
- Run 2: 84 of 84 and 36 of 36 again, clusters 0.69 logits apart; 16 of the 84 chose the
  baseline-like token.
- Scope (round-2 correction): on this prompt the margin sees the formal-verification adapter
  only. Single requests with each adapter subset give fv alone -0.131 (adapter side), corpus
  alone -0.925, design alone -0.534, corpus+design -0.634 (all W0 side, threshold -0.370). The
  test shows an adapter-sized shift reached every request; it does not show all three adapters
  were applied.
- The server never batches different adapter configurations together, so the mixed rounds equal
  the two reference requests exactly. Effective sample in run 2: 34 distinct adapter margins and
  14 distinct W0 margins.

Library level: an identical decode repeats to exactly 0.0. A different batch composition moves
logits by up to 0.26-0.46. The adapters shift the fork margin from -0.94 to -0.10 when the prompt
is decoded whole, so the top token does not even change there.

The original "LoRA changes output" gate passed on a 0.08-logit margin created by how the server
splits the prompt. This is one prompt and one token position.

**Use**: test adapter application and rollback on log-probabilities, not text.

### E2. Where the noise comes from

In the pinned source the CUDA backend quantises the activation operand of quantised-weight
matmuls (`mmvq.cu:1562`, `mmq.cu:291`, `quantize.cu:89-90`) and picks the kernel by batch row count.

| Reconstruction of six FFN matmuls | Relative error vs runtime |
|---|---|
| exact float64 on the dequantised weights | 3e-3 to 9e-3 |
| inputs rounded to float16 | unchanged |
| inputs quantised to int8 per block of 32 | 1e-5 to 1.1e-4 |
| same with blocks of 16 or 64 | unchanged |

That this is what makes logits depend on batch composition is an inference. There is no
unquantised control, because an F16 copy of this model does not fit in 8 GiB. The mechanism is
also incomplete (round-2 correction): eight identical sequences in one batch still differ row to
row (W0 fork margins -1.04 to -0.80 across the 8 rows of `all8_whole`), which kernel choice by
batch row count cannot explain, since all rows share one batch size. Treat the cause of
batch-composition noise as open.

It gives the stack a **resolution limit**: about 0.09 in final-residual norm (sqrt(0.116² -
0.069²), the measured anchor change under the corpus adapter minus the exact-arithmetic
prediction), 0.2% of the radius. A leakage bound tighter than that is valid arithmetic with no
observable consequence here.

### E3 and E4. The allocation question, measured instead of modelled

One real rank-8 `W_up` LoRA per layer, loaded together, with the scale vector as the allocation.

| Layer | First-order pass-through gain | Live gain | Ratio |
|---|---|---|---|
| blk.28 | 1.081 | 0.344 | 0.32 |
| blk.29 | 1.232 | 0.505 | 0.41 |
| blk.30 | 2.460 | 0.644 | 0.26 |
| blk.31 | 5.348 | 5.183 | 0.97 (99.96% of the exact float64 value) |

A control-vector probe gives the same direction and a smaller loss (52% to 78% retained).

| Allocation (share on 28/29/30/31) | eta per gain² vs last layer | Anchor movement vs last layer |
|---|---|---|
| uniform | 4.5x worse | 18x (49x against the float64 baseline) |
| pass-through "optimal" | 8.8x worse | 18x |
| team's product-law shares | 1.4x worse | 12x |
| 5 / 5 / 10 / 80 | 21% better | 8x |

Limits that matter:

- The uniform figure is measured where blk.28-30 run above scale 1 and leave the quadratic
  regime. Adding scale-1 costs incoherently gives 3.0x. Worse either way.
- "5/5/10/80 is best" is the best of nine hand-picked splits at one gain level, with no interval.
- The per-layer targets pull the final-norm gradient back through each layer's own `W_down`, as
  the team's synthetic experiment does. It may be the adapters that are mis-aimed, not the layers
  that are expensive. Truncation error is also higher at blk.28-30 (0.51-0.65 against 0.44).
- The loss may be cross-position transport rather than distortion of a local perturbation. If so,
  position-local per-layer gradients will not repair it.
- Radial shrinkage is not the only harm. Three of the five mixes halve on-domain KL per unit
  gain², and four give the positive token more log-probability.

My first analysis, using the pass-through model, predicted that a uniform split would be 2.6x
*better* than the last layer alone. The live test reversed it.

### E5. What a per-key leakage certificate bounds

A preservation key is the anchor prompt's decision position. An FFN weight edit acts at every
position. At an earlier layer the unconstrained positions are perturbed, and the later attention
and recurrent blocks carry those perturbations into the preserved one. After the last layer
nothing mixes positions.

So a certificate of leakage on listed keys bounds the final effect only for last-layer edits.
The measurement is solid; the mechanism is reasoned and not isolated experimentally.

### E6. Target-residual preconditioning: a mixed result

The harvester's target is the contrast gradient pulled back by the transpose,
`du = s * (W_down^T g)`. For a last-layer edit `dh = W_down (s * du)` exactly, so this gives
`dh = M g` with `M = W_down diag(s²) W_down^T`. Its energy per unit gain² exceeds the minimum by
a factor measured at a median of 9.1. `du = s * W_down^T (M + lambda I)^-1 g` removes most of it.

| At equal gain, lambda = mean eigenvalue | Plain | Preconditioned |
|---|---|---|
| eta, per state, exact | 1.00 | 0.21 |
| eta per gain², rank-8 compile, live | 0.00096 | 0.00047 |
| log P of the positive token, live | +0.288 | +0.375 |
| residual movement on unrelated prompts | 0.0555 | 0.0353 |
| **KL on unrelated prompts** | **0.138** | **0.215** |
| on-domain KL | 0.612 | 0.628 |
| weight norm | 3.17 | 4.66 |

Live gain and eta match the float64 prediction to three or four digits.

What does not hold up:

- KL on unrelated prompts rises by 55%, although the residual moves less. I first reported the
  smaller residual movement as "off-domain prompts move less". A reviewer caught that the
  distribution moves more. A smaller change in the residual is not a smaller change in behaviour.
- The unregularised minimum-norm target is worse once compiled (+0.172 log P, 8x the weight norm).
- A milder setting (lambda 10) improves the mean but not the per-state median of eta.

**Sizing**: this is the gradient versus Gauss-Newton distinction, which is standard. It matters
here because this project edits the up-projection, upstream of a gate and a down-projection.

### E7. Upstream issues found by independent replication

| Issue | Basis |
|---|---|
| The E8 `r_norms` match post-norm FFN key norms within 2.2% at all four layers; residual radii rise from 41 to 59. | The match is measured. That their script reads key norms is inferred, and the state sets differ. |
| 3 of the 8 design-pack training pairs have identical positive and negative token sets, so a zero target; 4 more share their first token. | Token ids, plus the harvester code path. Not confirmed by recompiling the pack. |
| The shipped formal-verification adapter moves unrelated corpus prompts by 13.1% of the residual radius, its own states by 12.7%, the 24 anchors by 0.4%. | Measured live, matches float64. n is 8 prompts. |

### E8. Rollback without a cache flush

Decode a prefix with an adapter on, switch it off without clearing memory, decode k more tokens,
compare with a run that never had the adapter. Corrected probe (v2): an output row on every
prefix token, so the adapter actually runs on the prefix. 14 sequences.

| Adapter at | max logit difference, k = 1 | k = 16 | Exact matches |
|---|---|---|---|
| blk.28 | 0.30 median, 0.51 worst | 0.12 median | 0 of 14 |
| blk.29 | 0.19 median, 0.45 worst | 0.11 median | 0 of 14 |
| blk.30 | 0.12 median, 0.17 worst | 0.11 median | 0 of 14 |
| blk.31 | 0.0000 | 0.0000 | 14 of 14 |

The blk.31 FFN adapter is verified to run on the prefix: while on, it shifts prefix-end logits by
a median of 1.6 to 1.9 across k. Controls: a clean repeat gives 0.0, and flushing then decoding
the whole sequence gives 0.0. The residue at blk.28..30 is small at these strengths (KL below
1e-3, no top-token change), about the size of batch-composition noise, and does not decay away
within 16 tokens.

The first probe (v1, 16 sequences) decoded the prefix with no output rows. This model evaluates
the last-layer FFN on output rows only, so the blk.31 adapter never ran on the prefix and its
"16 of 16 exact" was true by construction. It is kept as a labelled replicate for blk.28..30
only; the corrected rerun is exact for 14 of 14 at k = 1, 4 and 16.

Nothing follows the last FFN, so a blk.31 adapter cannot write into any cache. The cache-flush
step of the rollback contract is needed for earlier-layer adapters and not for last-layer ones.
This is library level: llama-server already clears a slot's cache when its adapter configuration
changes, so the server path does not exercise mid-sequence rollback without a flush.

### A baseline worth knowing: how much the edit actually achieves

On the 112 held-out states the base model ranks the positive token first at 51.8%. A rank-8
last-layer adapter under the team's recipe raises that to between 56% and 59% across a tenfold
range of strength (eta 0.002 to 0.02), fixing 15% to 17% of the 54 contested states. At the strongest
setting it also breaks 7% of the states that were right. On-domain KL over that range is 0.23 to
1.27. n is small (54 contested states), and the split is weak, so read this as an order of
magnitude: the edit moves the distribution a good deal more readily than it flips decisions.

### Supporting validation

- This harvest against the team's bf16 PyTorch harvest on the Mac, at their 332 recorded states:
  ratio 0.9984, sd 0.0055, Pearson r 0.99977. Tokenisation reproduces their recorded ids.
- A second full live run reproduced the first to every recorded digit.
- The server's reference margins equal the library emulation to four decimals.

## 2. Convergences

**Where per-step guarantees stop adding up.** The team's note argues that all four streams share
one pattern: check one inequality per step, and the guarantee for the whole run follows because
the per-step terms add. Three findings are the same observation from different sides.

- Per-layer shrinkage costs do not add (E3, E4).
- Per-key leakage does not bound the final effect except at the last layer (E5).
- A text-level check is not a check of the claim it stands for (E1).

**The last layer is special for one reason.** Nothing after the last FFN mixes positions or writes
state. So only there does gain reach the output intact (E3), does a per-position constraint bound
the final effect (E5), and does switching the adapter off leave no trace (E8).

**Residual norm is not the harm.** E6 reduced the norm of the change and increased the change in
the off-domain distribution. E4's best split by shrinkage is not the best by on-domain KL. The
project optimises and certifies norms; what users experience is distributions. That gap is the
most important open design question this work surfaced.

**A floor under all of it.** E2: the runtime's own rounding is larger than some of the quantities
being certified.

A useful two-part question for every certificate in the stack: does the per-step quantity
actually add along the run, and is the bound above the noise floor of what will execute it?

## 3. Threads worth pulling

| # | Thread | Why it might pay | Cost |
|---|---|---|---|
| T1 | Per-layer targets from autograd through the full forward pass | Would show whether multi-layer editing can be made efficient, which E3 leaves open | easy on the Mac; PyTorch on CPU here |
| T3 | Preservation that generalises | Anchors hold and neighbours do not. Noise keys are dominated by one direction (participation ratio 5 to 7) | cheap: add off-domain prompts as a metric first |
| T5 | Certificates stated relative to the runtime noise floor | Makes "certified" mean "observable" | design work |
| T6 | Batch-composition invariance as a requirement | Replayable transcripts are not replayable if neighbours in the batch change the output | needs a runtime decision |
| T7 | Higher-rank or state-conditional realisation of distribution-space targets | They work per state and fail at rank 8 (see retired) | open |
| T8 | Thermal attribution | The excursion followed the end of CPU work on the host; needs CPU and fan telemetry | cheap |

## 4. Retired along the way

Each of these was believed for a while.

| Hypothesis | What killed it |
|---|---|
| The server drops the adapter on some slots. | Log-probability margins: 84 of 84 (E1). |
| Multi-layer splitting cuts shrinkage roughly as 1/L. | Predicted 2.6x better by my own pass-through analysis; measured worse (E4). |
| The minimum-norm target is the best target. | It collapses after compilation (E6). |
| A preconditioned adapter disturbs unrelated prompts less. | Their residual moves less and their KL rises 55% (E6). Caught in review. |
| Targets chosen in distribution space fix the KL problem. | Per state they cut typical collateral KL about 15x. Through a rank-8 compile they give 3x the weight norm, more shrinkage than the plain target, and no KL benefit. They differ too much from state to state for a low-rank fit. |
| Spending the edit only where the base is wrong, or contrasting against the base's real top token, will work better. | The base already ranks the positive token first at 54.5% of states. Training only on the contested ones, or against the actual top-1 or top-3 competitors, gave no gain in held-out accuracy and more collateral change at every strength tested (`contested_exp_v1.json`). The team's recipe was as good or better. |
| The team's `gammas` equal the analytic gate-metric sensitivity. | No match; their definition is unconfirmed. |
| The first harvest ran on the GPU. | It ran on CPU; llama.cpp fell back silently. Caught by 0 MiB of VRAM in use. |
| The thermal excursion was the GPU overheating on its own. | The first 84 C sample follows the last working-agent post by 14 s (the Opus workers stopped 39 s before onset), but the stall monitor kept posting afterwards and CPU load was not recorded. A fan curve keyed to CPU temperature would fit. Undetermined. |

## 5. Review record

The four outgoing receipts were recomputed from raw files by two independent reviewers with no
access to my reasoning. Every number reproduced. They still found 24 problems, all in framing:
a library result presented as a server result, a constructed quantity presented as a harvested
one, an understated ratio whose baseline sat at the noise floor, a claim that was wrong, a missed
host event in the thermal trace, and several inferences worded as fact. `receipts\REVIEW_ROUND_1.md`
lists each with its resolution. The largest correction led to a new experiment (the server-level
test in E1) that turned an inference into a measurement.

A second round (2026-11-19, about 19:25 to 19:48) reproduced every number again and raised 11
further points, recorded in `receipts\REVIEW_ROUND_2.md`. The ones that change what a claim
means: the server margin sees the formal-verification adapter only (new subset-reference data,
`server_logit_check_v2.json`); the mixed rounds are serialised, so the effective sample is 34
distinct adapter margins and 14 W0 margins; the blk.31 "exact rollback" was true by construction
because the prefix had no output rows (corrected probe re-run, `rollback_probe_v2.json`: the
adapter shifts prefix-end logits by a median of 1.6 to 1.9 while on, and rollback is exact for
14 of 14 at k = 1, 4, 16); and the noise mechanism is incomplete because identical rows in one
batch still differ. The rest are wording and scope. All round-2 fixes were applied on 2026-11-19
evening by the takeover session, the receipts rebuilt in the order 010, 009, 011, 005, and this
file updated (E1, E2, E8). A third, narrow review of only these changes is recorded in
`receipts\REVIEW_ROUND_3.md`.

## 6. Not new, and limits

- Deterministic cross-ISA compilation, the certificate design and the composition tests are the
  team's work. This session confirmed parts of them and did not extend them.
- The noise-floor mechanism is standard runtime behaviour. The contribution is measuring it and
  connecting it to verification.
- n is modest (112 held-out states of 32 prompts, 24 anchors, 8 prompts per small pack) and
  intervals resample states, not prompts.
- Claims about the other team's code are inferences where marked. Their multi-layer harvest
  script is not public.
