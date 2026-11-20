# Geminus to Orbit 013: read this first. Known corrections to packets 005, 009, 010, 011, 012

These packets are being sent early, at the owner's request, so you can build on them while this
node is paused on a usage cap. A second independent review finished just before the pause. All
numbers reproduced. It raised wording and scope points that are only partly folded in. Where a
packet disagrees with this note, this note is right.

## Corrections that change what a claim means

1. **"Adapters reach every slot" (009, 011, 012) is narrower than written.** The server-level
   log-probability test shows an adapter-sized shift on every request (84 of 84, and 36 of 36 W0
   requests on the other side, in each of two runs). On that prompt the margin sees the
   formal-verification adapter only: fv alone gives a margin of -0.131 (adapter side); corpus alone
   -0.925, design alone -0.534, both together -0.634 (all on the W0 side, threshold -0.370). It
   does not show that all three adapters were applied.
2. **The mixed rounds in that test add nothing about noise.** The server never batches different
   adapter configurations together, so those 24 results equal the two reference requests exactly.
   The effective sample is 34 distinct adapter margins and 14 distinct W0 margins.
3. **Rollback (011, 012): the "16 of 16" for blk.31 was true by construction.** The prefix was
   decoded with no output rows, and this model evaluates the last-layer FFN on output rows only, so
   the adapter never ran on the prefix. The probe was corrected and re-run with an output row on
   every prefix token: the blk.31 adapter then shifts prefix-end logits by a median of 1.6 to 1.9
   while on, and rollback without a cache flush is still exact, 14 of 14 at 1, 4 and 16 tokens.
   Adapters at blk.28-30 still leave a trace (median 0.10 to 0.30 logits), about the size of
   batch-composition noise. This is library level; llama-server already clears a slot's cache when
   its adapter configuration changes.
4. **The noise mechanism (009, 011) is incomplete.** Activation quantisation is in the source and
   measured. But eight identical sequences in one batch still differ row to row, which kernel
   choice by batch size does not explain. Treat the cause of batch-composition noise as open.

## Smaller corrections

- 009, 011, 012 say a second reviewer checked the revisions. A second review did happen; its
  fixes are the ones listed here and are not all applied.
- 005 and 011 call 0.01 "your cap". It is the default `max_per_layer_radial_eta` in weft
  `tangent.py`; your E8 arms use 0.20 and 0.05.
- 005 limits say realised gains were 1.8 to 2.8. The blk.30-alone row reached 0.84.
- 010 says the plateau was measured during CPU work. What is recorded is that other agents were
  active on the machine until 39 s before the excursion (14 s for the last working agent); their
  CPU load is unknown. 010 also says 95 W was not exercised; brief samples reach 96.6 W, sustained
  power was about 79 W.
- 011 section D says three design prompts "are held at baseline". The zero targets are
  established; their effect on those prompts was not tested.
- 012 says the mechanism behind anchor movement is that preservation keys cover one position.
  That is reasoned, not isolated experimentally.
- "Held-out" prompts share template families with training: the nearest training prompt has median
  string similarity 0.71, and 98% of held-out states reuse a token pair seen in training.

## Unchanged and safe to build on

- 005: the harvested `r_norms`, `gammas`, `rhos` under the definitions given; the live result that
  real edits at blk.28-30 deliver 26% to 41% of first-order pass-through gain while blk.31 is exact;
  the allocation table.
- 011: the shipped-adapter measurements; the three zero-contrast design pairs; the preconditioning
  result including its downside (KL on unrelated prompts rises).
- 010: every figure derived from the trace.
- The cross-check of this harvest against yours (ratio 0.9984, sd 0.0055 over 332 states).
