# Independent review, round 1 (2026-11-19, about 18:43 to 19:05)

Two reviewers with no access to the author's reasoning recomputed the four draft receipts from raw
files. Both returned SEND AFTER FIXES for every receipt. Their scratch scripts are in `verify-a\`
and `verify-b\`. They could not save their own report files, so the findings are recorded here.
"Resolution" says what was done about each.

## Reviewer A: 005 (M1 summary) and 011 (runtime and pipeline findings)

Almost every number reproduced. Problems were in definitions, wording and two noise-limited ratios.

| # | Finding | Resolution |
|---|---|---|
| A1 | Receipts were stale against builders edited after they were generated. | Regenerated after all fixes; re-reviewed. |
| A2 | 005 top-level `r_norms`/`gammas`/`rhos` were a constructed set, true by construction for four adapters at scale 1; `g_pred` cancels so the two factors were individually arbitrary. | Pass-through (harvested) set moved to the top level. Live costs kept under a named key and labelled as constructed. |
| A3 | Anchor baseline for blk.31 (0.0014) is at the runtime noise floor; float64 gives about 0.0005, so "8 to 18 times" understates. | Reported as "at least", with the float64 baseline and the resulting range beside it. |
| A4 | `g_pred` is first-order; exact float64 gain at blk.31 is 99.96% of live, so "97%" is linearisation. Probe (60-78% retained) and adapters (26-41%) were presented as agreeing. Median and mean mixed. | Labelled first-order, exact figure quoted, both ranges and statistics stated, disagreement stated. |
| A5 | "Same decline with depth" compared unlike quantities. | Sentence removed. Numbers shown side by side with "no match found". |
| A6 | F1 was worded as fact; Mac final radius differs 5-6% from ours so the state sets differ; the action misdescribed their hooks. | Reworded as a match and an inference; state-set difference stated; hooks described correctly. |
| A7 | "About 80% on the last layer" rests on one hand-picked split, one gain level, 112 states of 32 prompts, no interval; 21% not 22%; uniform's 4.5x is measured outside the quadratic regime (scale-1 costs give about 3.0x); single-layer etas are 7-13x the team's 0.01 cap. | All stated. Claim reduced to "best of nine tested". |
| A8 | "Held-out" prompts share four template families with training (median string similarity 0.71; 98% reuse a token pair seen in training); anchors evaluated are the compile-time keys. | Stated wherever "held-out" appears. |
| A9 | "Off-domain prompts move less" holds for residual movement only; corpus-prompt KL rises 55% at lambda 1. "On-domain KL within 10%" is false live for lambda 0.1 and minimum-norm. | Corrected in the receipt, in FRONTIER.md, and to the owner. |
| A10 | Summary A stated as fact from one prompt and a library emulation; B's causal link has no unquantised control; the quantisation is in the pinned source (`mmq.cu:291`, `mmvq.cu:1562`), and block size 32 is specific. | A is now backed by a server-level logit test (84/84, 36/36). B cites the source lines and labels the link to composition noise as inferred. |
| A11 | Design pack is weakest only in E8's single-layer arm; "base already right" overstated (positive token is top-1 at 54.5% of states, 88.6% is against listed negatives only). | Both reworded with the exact figures. |
| A12 | "About half" the on-domain KL holds for 3 of 5 mixes; KL per gain squared is not scale-invariant; 4 of 5 mixes give more log-probability. | Exact counts and the scale caveat stated. |
| A13 | Scrub regex damaged `orbit-to-geminus`; `n_states` 342 should be 345; one result file had no generating script; rounding after division; bootstrap resamples states not prompts. | Fixed; script saved as `passthrough_vs_live.py`. |

Alternative explanations the reviewer raised, now stated in the receipts:
- Pass-through failure may be cross-position transport rather than distortion of a local perturbation.
  If so, position-local per-layer gradients will not repair it.
- The last-layer advantage may partly be a property of these adapters (mis-aimed targets, higher
  truncation error, large scales). Scale-1 costs still differ 9 to 23 times.
- Equal mean gain can hide redistribution across states. Lambda 1 survives on medians; lambda 10 does not.

## Reviewer B: 009 (GPU smoke) and 010 (thermal)

Every number reproduced.

| # | Finding | Resolution |
|---|---|---|
| B1 | A library-level result was presented as a server-level one; the probe never used the server's per-request path. | Server-level logit test run (`server_logit_check.py`): 84/84 and 36/36. Gate renamed to say which level each piece of evidence is from. |
| B2 | Server and emulation disagree in places not mentioned: a 7-of-8 round with three arrival waves that was not emulated; the mixed emulation reconstructs run 1 and predicts one row where run 1 showed two. Rounds 0-1 match one emulation exactly. | All four facts stated. |
| B3 | "Cause (measured)" overstated: six matmuls measured, link to composition noise inferred; evidence file not cited or hashed. | Reworded, file hashed. |
| B4 | 010 omitted a host change: other agents were writing files until t+628 s, then stopped; the excursion begins 39 s later. | Both facts stated beside the interpretation. |
| B5 | The hashed server logs contain a user path. | Logs are not shipped; the receipt says so. |
| B6 | Order of events unclear; the re-run overlapped a CPU job (22.4 tok/s per slot against 35.5-36.9). | Order and overlap stated. |
| B7 | "17 W0-identical" is wrong: they match for 60 characters only; none of the 24 equals a W0 output in full. | Reworded; full-content hash split reported. |
| B8 | VRAM prediction used a literal, not the launch arguments; both tolerances were chosen after the data. | `-c` now parsed from the recorded flags; tolerances labelled as chosen after the data. |
| B9 | 010 wording: temperature was still rising at abort, not steady; first 84 C at t+666.8 s; plateau not flat; "lid open, on AC" and "80 W limit" not recorded in any artifact. | All corrected or removed. |
| B10 | No lease record is embedded; "free MiB before/after the lease" is the stamp's snapshot. | Reworded. |
| B11 | Minor rounding; a string that read like a scrubber placeholder; 009's title did not say a gate failed. | Fixed. |
