# Independent review, round 3 (2026-11-19, about 21:00 to 21:35 EST)

Reviewer: third independent reviewer (verify-d). Narrow scope: **only the round-2 changes** listed in
`REVIEW_ROUND_2.md` (11 findings), plus regression spot-checks of untouched numbers and consistency of
the digest / FRONTIER / 013 with the rebuilt receipts. No access to the author's reasoning; judgments
are from the files alone. Prior reviewers' scratch (verify-a/b/c) was read for method hints only;
every number below was recomputed from the raw files with new scripts in `verify-d/`.

**Headline: every round-2 numeric correction reproduces from raw data, and every required framing
correction is present in the body text. Two one-line framing/consistency residuals remain (009 title,
005 review field), plus one clarity recommendation about the git push. Nothing numeric is wrong.**

Context from `HANDOFF.md`: the 19:58 drop (pre-fix 009/010/011; 005 with round-2 text fixes but two
stale script hashes) was already SENT by Taildrop and pushed to git. The rebuilt packets are the
pending revision; this review gates the resend.

## Method

- Python (`m1/.venv/Scripts/python.exe -I`, numpy available), scripts `verify-d/d1_structure.py` …
  `d10_stage1.py`. All checks recompute from raw files (`m1/out/*.json`, `mlos/phase3/thermal-nvsmi.csv`,
  `mlos/phase1/run2/*`, `m1/scripts/*`); receipt values are compared, never trusted.
- Rounding: receipts use 4 decimals for O(0.1–1) values, 4 significant figures for small ones, and
  1–3 decimals / 1 significant figure in prose. A stated value passes if it equals the raw value under
  any of these conventions. All prose ranges below were checked against true min/max, not just endpoints.
- Margins recomputed as `fence_logprob − think_logprob` per request; classification re-applied with the
  midpoint rule; distinct counts recomputed under several counting rules (stated below).
- Rollback statistics recomputed from the 42 per-sequence rows (14 sequences × k ∈ {1,4,16}), not from
  the raw file's own summary; the raw summary was then checked against the recomputation.
- Thermal figures recomputed from the CSV with t0 = first `sustained_load` sample (16:40:05.521 local);
  window conventions cross-checked against verify-c's method hints and re-derived independently.
- sha256 recomputed for **every** file referenced in all four `artifact_sha256` blocks (60 entries).
- Structural diff of the rebuilt receipts against the pre-round-2 originals preserved in
  `verify-c/orig_receipts/` to enumerate exactly what changed (28 / 32 / 10 / 70 field-level diffs in
  005 / 009 / 010 / 011, plus the digest line diff). Every changed string was reviewed for new overclaim.
- Builder staleness check: `make_005.py` / `make_011.py` mtimes (20:57:47) postdate the rebuild
  (20:51:54). Sandboxed copies with output redirected to `verify-d/rebuild_out/` were run in the order
  010, 009, 011, 005 against the current raw files: **all four receipts and all four auxiliary files
  (gates, embedded workloads) reproduce byte-identically.** The post-rebuild edits are output-neutral
  hygiene (output path; scrub regex assembled from string parts, semantically identical). No staleness.
- Rebuild order confirmed from mtimes: 010 (20:51:54.061) → 009 (.574) → 011 (.738) → 005 (.936).

## Per-checklist results

### 1. Finding 2 - fv-only scope (PASS)

From `server_logit_check_v2.json` (recomputed, full precision):

| subset | margin | side of threshold −0.3699007 |
|---|---|---|
| formal_verification_only | −0.13101577758789062 | adapter |
| corpus_only | −0.9251994490623474 | W0 |
| weft_eido_only | −0.5338401794433594 | W0 |
| corpus_and_weft_eido | −0.6341400146484375 | W0 |
| w0_again | −0.8180294036865234 | W0 |
| three_again | +0.07822799682617188 | adapter |

- `decision_threshold_margin_midpoint` = (−0.8180294 + 0.0782280)/2 = **−0.3699007034301758**, identical
  in v1 and v2, and equal to the value embedded in 009 (both runs). 011 quotes −0.3699; FRONTIER E1 and
  013 quote −0.370. All consistent.
- lora_echo scales of each subset match its label (fv = id2 only, etc.).
- Both runs reclassified request-by-request: **v1 84/84 adapter, 36/36 W0, 33 think-token; v2 84/84,
  36/36, 16 think-token.** Ranges: v1 adapter [−0.1271, 0.1860], W0 [−1.0092, −0.7639], gap 0.6368;
  v2 adapter [−0.1161, 0.1860], W0 [−1.0092, −0.8060], gap 0.6899. All match 009/011 (4 dp) and the
  digest/FRONTIER prose (0.64 / 0.69; −0.13…+0.19; −1.01…−0.76).
- 009's embedded per-round margins/classified/second_token/requested equal the raw data (4 dp) for all
  15 rounds of each run; 011's run_v1/run_v2 blocks equal the raw summaries.
- The required claim wording is present in 009 (headline note, gate threshold SCOPE clause,
  server_level_logprob_test.reading), 011 (summary A, claim, what_the_margin_sees), 012 (009 headline,
  011 line 1), FRONTIER E1, 013 correction 1: "an adapter-sized shift reached every request; the margin
  sees the formal-verification adapter only; the test does not show all three adapters were applied."
  No affirmative "all three adapters applied" claim remains anywhere (grep). **Residual: 009's title
  was not updated - finding R3-1 below.**
- Nuance (observation O2): corpus+design do shift the margin by +0.184 vs W0 (fv alone +0.687, triple
  +0.896); "sees … only" is about threshold crossing, and 011's own clause "cannot distinguish 'all
  three applied' from 'fv applied alone'" states the limitation correctly.

### 2. Finding 6 - serialised mixed rounds / effective sample (PASS)

Counting rule used and stated: **distinct margin values among all adapter-request results (84) and all
W0-request results (36), mixed and staggered rounds included**, at raw precision. Result:
**34 distinct adapter margins, 14 distinct W0 margins** - identical counts at 4-decimal rounding.
Excluding the mixed rounds: adapter still 34; W0 = 13 (the 14th W0 value is the mixed-round reference
−0.818029, which never occurs in the pure W0 rounds). The receipt's numbers match the raw file's own
summary under the all-results rule; see observation O1.

- Mixed rounds: exactly 24 results; every margin **bit-equal** to one of the two single-request
  references (−0.8180294036865234 / +0.07822799682617188); `mixed_rounds_note` = {n: 24,
  distinct_margins: [−0.818029, 0.078228], equal_to_single_request_references: true} - reproduced.
- 009 gate "measured" text, 009 note 4, 011 `effective_sample`, FRONTIER E1 and 013 correction 2 all
  state 34/14 and "the mixed rounds equal the two references exactly / show config switching only".
  Consistent everywhere.

### 3. Finding 3 - corrected rollback probe (PASS)

`rollback_probe_v2.json`: `prefix_outputs` = **"all prefix tokens"**; 42 rows = **14 sequences × k ∈
{1,4,16}**, same 14 ids at each k. Recomputed from rows (medians/min/max/means/fractions), then
compared to the raw summary (exact match) and to 011 (stated rounding, all fields):

- blk31 `adapter_effect_at_prefix_end_median` = **1.7396 / 1.8740 / 1.6137** at k = 1/4/16 → "median
  1.6 to 1.9 while on" ✓; mins 0.6278 / 0.4110 / 0.5250 (all > 0: the adapter demonstrably ran on the
  prefix). `exact_match_frac` = **1.0 at all three k (14 of 14)**; max_abs_logit_diff = 0.0 everywhere;
  KL = 0; no top-1 change.
- blk28–30 trace medians across k: min **0.1024** (blk30, k=4), max **0.2978** (blk28, k=1) → "0.10 to
  0.30" ✓; exact_match_frac 0.0 everywhere; all KL means < 1e-3; top1_changed_frac 0.
- Controls: clean_repeat and flush-then-whole-decode max_abs_diff = 0.0 at every k.
- FRONTIER E8 table vs raw: k=1 medians 0.2978→**0.30**, 0.1921→**0.19**, 0.1170→**0.12**; worst
  0.5118→**0.51**, 0.4450→**0.45**, 0.1662→**0.17**; k=16 medians 0.1233→**0.12**, 0.1098→**0.11**,
  0.1079→**0.11**; blk31 0.0000, 14 of 14; others 0 of 14. All ✓.
- v1 kept correctly: `rollback_probe_v1.json` has **no** `prefix_outputs` key, 48 rows = 16 sequences ×
  3 k, and no adapter-effect-at-prefix-end field for blk31 (never ran). 011 `v1_replicate` carries the
  true-by-construction caveat ("the blk.31 adapter never ran on the prefix … the v1 blk.31 row carries
  no evidence about rollback"), and its by_k numbers equal the recomputed v1 statistics. Digest line 3
  and FRONTIER E8 carry the same caveat ("16 of 16 was true by construction").
- Library-level scope: 011 `consequence` - "This is library level: llama-server already clears a slot's
  cache when its adapter configuration changes, so the server path does not exercise mid-sequence
  rollback without a flush." Same label in digest line 3, FRONTIER E8, 013 correction 3. ✓
- File hashes: v1 fb7cc2b3…, v2 76c6541d… match 011 `source_sha256` and `artifact_sha256`; the four
  adapter hashes inside rollback_probe_v2 equal the plain_blk28..31 GGUF files.
- Wording nit (optional, R3-5): FRONTIER E8 says the residue "does not decay"; medians drop k=1→4
  (0.30→0.13 blk28) then persist to k=16. "Does not decay away within 16 tokens" would be exact.

### 4. Finding 7 - noise mechanism incomplete (PASS)

`probe_analysis_v1.json` `p2_by_variant.all8_whole.w0_fork_margin_min_max` =
**[−1.0376853942871094, −0.802459716796875]**, n_rows 8. Independently recomputed from the 8 per-row
`w0_fork_margin` values in `probe_v1.json`: min/max identical; 7 distinct values over 8 rows (two rows
tie at −1.0377), all top-1 = think. `probe.py` confirms the variant decodes the same tokenised prompt
T on all 8 sequences in one `decode_multi` batch - "8 identical sequences in one batch" is accurate.
Spread 0.2352 → 011 "about 0.24" ✓.

- 011 `identical_rows_in_one_batch` [−1.0377, −0.8025] ✓; 009 note "−1.038 to −0.802" ✓; FRONTIER E2
  "−1.04 to −0.80" ✓; digest line 2 ✓.
- OPEN wording present everywhere required: 009 ("Treat the cause of batch-composition noise as open"),
  011 B ("INFERRED, AND INCOMPLETE … The cause is open"), 011 summary B, digest ("The cause of the
  noise is open"), FRONTIER E2 ("Treat the cause … as open"). ✓

### 5. Finding 8 - eta cap attribution (PASS, with a local-verification limit)

011 `C_shipped_adapters.eta_note`: "0.01 is the default max_per_layer_radial_eta in weft tangent.py,
not a rule the team set (the E8 arms use 0.20 and 0.05); these single-layer adapters were compiled
without a per-layer cap." - exactly the required correction; no "your cap" remains (grep). 005
`operating_point_caveats` carries the same attribution (`weft_tangent_default_per_layer_eta_cap: 0.01`,
E8 arms 0.20/0.05). The eta figures in the note reproduce from probe_analysis p3 (fv 0.02905→0.029,
max 0.1917→0.192; corpus 0.04728→0.047; design 0.03341→0.033). **Limit:** weft/mcproof sources are not
on this machine, so the default value itself is accepted as recorded in REVIEW_ROUND_2 (both prior
rounds state it); it could not be re-verified against tangent.py here.

### 6. Finding 9 - prose numbers (PASS, one optional precision note)

(a) `actq_check_v1.json`, all six matmuls (ffn_up/ffn_gate/ffn_down × blk28/blk31):
q8_input = {2.805e-05, 1.388e-05, 2.999e-05, 9.437e-05, 7.995e-05, **1.133e-04**}; exact_f64 =
{5.315e-03, 3.023e-03, 8.628e-03, 3.675e-03, 3.119e-03, 4.865e-03}. True ranges: q8 **1.388e-05 …
1.133e-04**, exact **3.023e-03 … 8.628e-03**. 009's note ("1e-05-1e-04 where exact arithmetic differs
by 3e-03-9e-03"), FRONTIER E2's table and digest line 2 now cover all six including ffn_gate and
ffn_down; endpoints are 1-significant-figure roundings of the true min/max. Optional: the q8 upper end
reads 13% low as a strict bound (R3-4). 011 B's by_layer / block-size / rounding-size tables all equal
actq under stated rounding.

(b) `compile_exp_v1.json` `results.lam10.equal_gain.held_out.gain_exact.median` = **3.230022791594015**
→ 011's lambda-10 sentence now quotes "median gain 1.2153 plain, **3.23 at lambda 10**" (plain median
1.2152742 ✓). The old lambda-1 value (2.1035) is gone from that sentence. Eta medians 0.0038252653 /
0.0042911353 → "0.003825 plain, 0.004291 at lambda 10" ✓.

(c) `thermal-nvsmi.csv`: max power.draw = **96.62 W** (one sample at t+258.9 s); **2 samples > 95 W**;
steady-window (t 120–660 s) mean power **78.90 W**. 010 now says "Sustained power settled near 78.9 W;
brief samples reach 96.6 W (2 samples above 95 W), so the profile's dgpu_power_cap_w=95 W was touched
but not sustained." The phrase "95 W was not exercised" is gone (grep). ✓

### 7. Finding 10 - inference as fact (PASS)

- 010 no longer asserts the plateau/CPU link as fact: "That the plateau was measured during their CPU
  work, and that the excursion follows the end of that work, are inferences from the timing, not
  measurements." ✓ "The CPU load of those agents is not recorded" ✓ (`cpu_load_of_other_agents: "not
  recorded"`). "Whether a second process shared the GPU was not recorded" ✓ (no "no second GPU
  process" claim anywhere).
- Stall monitor: `stall_monitor_last_post_at_t_s` = **2425.0**. With t0 = first sustained sample
  16:40:05.521 (builder truncates to 16:40:05 for whole-second host events): 17:20:30 − 16:40:05 =
  2425 ✓; opus stop 16:50:33 → **628.0** ✓; last worker post 16:50:58 → **653.0** ✓; first 84 C sample
  t+666.8 (recomputed 666.854) → 666.8−653 = **13.8** ✓, 666.8−628 = **38.8** ✓. Note text: "The stall
  monitor kept posting to the board until t+2425 s, after the run aborted, so host agent activity did
  not cease at the excursion." ✓ (Run start 16:39:59 local = stamp 21:39:59Z ✓; abort at 743 s < 2425 ✓.)
  Exact-timestamp deltas are 14.3/39.3 s; the prose "14 s"/"39 s" matches either convention; the
  whole-second convention is disclosed in `host_activity.source` (observation O3).
- Digest: "The antigravity swarm on this node (**three of its six agents ran Opus, the rest Gemini**)"
  ✓; "the Opus swarm" is gone. 010's "the two Opus workers stopped at t+628 s" is consistent (two
  quota-stopped workers; the last working-agent post is later, t+653) - observation O6.
- Digest 005-line-5: "The **reasoned** mechanism is that preservation keys cover one position; **it is
  not isolated experimentally**." ✓ (matches 005 F4 confidence and FRONTIER E5.)
- 011 D: "The zero targets are established; **their effect on those three design prompts was not
  tested**." ✓ in `consequence`, summary D, digest 011-line-5, 013. "Held at baseline" is gone.

### 8. Finding 4 - review descriptions (PASS with two residuals)

- 011 `review`, 009 review note, 010 review note: two rounds, all numbers reproduced, round-2 fixes
  applied **by the author**, "a third, narrow review of the revision is recorded in
  receipts/REVIEW_ROUND_3.md". No remaining "checked by a second reviewer" overstatement for the
  round-2 fixes (grep; the only occurrence is 013's historical record of the old wording). ✓
- Digest: "A second review then raised further points; those fixes are now applied in this revision" ✓.
- **Residual R3-2:** 005's `review` field ends "The round-2 fixes were applied by the author **and not
  independently re-checked**." That was true at rebuild time but now contradicts 013's header
  ("re-checked in a third, narrow review") and the sibling packets, which all reference this report.
- **Residual R3-3 (shared):** 013's header, the digest, FRONTIER §5 and the ROUND_2 status line assert
  in the past tense that the third review "is recorded" / was done - written 20:49–20:51, before this
  review existed. The references are anticipatory; they are satisfied by this file now, but the
  past-tense "re-checked" in 013 was premature when written.

### 9. Finding 11 - benign labels (PASS)

009 carries one note covering both: "Embedded-data notes: 127.0.0.1 and the ephemeral ports in the raw
records are loopback-only server endpoints, not network exposure. The harness label w0_identical means
the first 60 characters match a W0 output, not full equality; full-content hashes are reported
separately." ✓ The classification rule ("first 60 chars") and the full-content hash split (0 of 24
equal any W0 output in full - recomputed from the embedded content_sha256 values) are consistent.

### 10. Finding 1 - 005 stale hashes + headline unchanged (PASS)

All 11 `script:*` entries equal the sha256 of the current `m1/scripts` files, and the list is exactly
the fixed 11 (no extras; the old glob-all-scripts entries are gone):
harvest.py ddf80760…, m1_analyze.py 0c2a0c83…, probe.py 210f6709…, probe_analyze.py 6f27f015…,
alloc_build.py 33c00906…, alloc_live.py 368ce3d2…, passthrough_vs_live.py d86d86ae…,
llama_ctypes.py a61a96b8…, ggufw.py 9e3e7dfc…, jobs.py 58e2f1e3…, render.py 2c1d8fca… - all match.
Headline numbers unchanged and reproduced from raw:

- `r_norms` = [41.4068, 44.5046, 49.6783, 59.0168] = m1_analysis l_out-28..31 means ✓ (gammas
  [1.1072, 1.2767, 1.6478, 3.1828], rhos [8.4151, 7.3811, 2.5609, 1.1584], kappa means, ci95s,
  product-law shares [0.1692, 0.1675, 0.3612, 0.3021] also reproduced).
- live/first-order ratios from alloc_passthrough_vs_live: 0.3181 / 0.4099 / 0.2619 / 0.9692 →
  **0.32 / 0.41 / 0.26 / 0.97** ✓ (005, digest, FRONTIER E3 table).
- uniform eta/gain² relative to only31 = **4.5473 → "4.5x"** ✓; late_heavy = **0.7863 → 21.4% better
  → "21%"** ✓. All 9 allocations' scales (formula α·G/g_live with unrounded G = 2.591389), gains, etas,
  KLs, anchor/corpus figures and relative columns reproduce from alloc_live_v2 + pvl; float64 anchor
  ratios = live/0.0005260626 (pvl baseline) ✓; limits gain range 0.8379–2.7511 → "0.8 to 2.8 (1.8 to
  2.8 without blk.30-alone)" ✓; held-out median similarity 0.7097 → 0.71, pair reuse 0.9821 → 98% ✓
  (the pvl file was regenerated with the difflib autojunk fix: vs verify-c's rerun only the two
  similarity fields changed, 0.5344→0.7097 and 0.25→0.375 - round-2 finding 5's fix, confirmed).
- Note for the resend: the **sent/pushed** 005 (git f6a50a3) already had the round-2 text fixes but
  carried stale hashes for `script:harvest.py` and `script:probe.py`; the rebuilt 005 corrects exactly
  those two and nothing else (diff of pushed vs current 005: only those two hashes).

### 11. Regression spot-checks (PASS)

- 011 shipped-adapter selectivity from probe_analysis p3: fv-on-own-states **0.1271**, corpus prompts
  **0.13137→0.1314**, design **0.08544→0.0854**, anchors **0.0044378→0.00444** ✓; digest/FRONTIER
  13.1% / 12.7% / 0.4% ✓. The entire C adapters block (4 adapters × 4 groups) matches p3 under stated
  rounding.
- 011 E live: alloc_live_v2 pass2 c31_plain/c31_lam1 eta/gain² **0.00095635→0.00096 / 0.000468→0.00047**
  (0.49x ✓), per-state medians 0.000861/0.000406 → 0.00086/0.00041 ✓, corpus KL **0.138374→0.138 /
  0.215096→0.215** ✓ (+55.4% → FRONTIER "55%", digest "about half" ✓), dlogp 0.288/0.375 ✓, corpus dh
  0.0555/0.0353 ✓. All five stage_3 arms match pass2; stage_2 all five arms match compile_exp
  (equal_gain + adapter truncation/fro/k0); stage_1 all six lambdas match target_residual_exp; kappa
  median 9.1085→9.11, n 124, participation ratio 125.708→125.7 ✓; native-scale live-vs-f64 figures ✓.
- 009 VRAM gates from run2 raw: peak **7096 MiB**, **112 samples at 0.5 s intervals**, baseline 0 ✓; prediction
  4274.8 + 2048.0 + 0.5 = **6323.3** (KV arithmetic 8 layers × 65536 × 4 kv-heads × 256 × 2 B × 2 =
  2048 MiB ✓); gap **772.7 MiB = 10.89% → 10.9%** ✓; −c 65536 parsed from recorded flags ✓;
  server_load_s 18.49 ✓; stamp clean, free 8018/8018, 23:36:02–23:37:01Z ✓; text classes 7/17/0 with
  slots [[0,1,2,6,7],[0,1,2,6,7],[0,1,2,3,5,6,7]] ✓; library gate min corr 0.99696→0.9970, projection
  [0.9873, 1.1110] → [0.987, 1.111] ✓; both post-hoc tolerances labelled ✓.
- 011 A library probe block (repeat 0.0; composition diffs 0.264/0.4624/0.4147/0.3924; fork margins
  −0.9405/−0.1035; corr mins; projection ranges; rows 4/5/1) all equal probe_analysis p1/p2 ✓; 55+4
  emulation equals the server references to 4 dp ✓.
- 011 D related_observation 0.545/0.886/8.7/n332 = pvl base_model_on_formal_verification_states ✓.
- Stamps/durations: probe 884047 ms, harvest 78625 ms, live 217609 ms, all clean with free 8018/8018 ✓;
  order of events (probe 23:06:46–23:21:30Z, text re-run 23:36:02–23:37:01Z, run1 00:11:56–00:12:35Z,
  run2 00:50:57–00:51:12Z) matches the stamps, and v2 request sent_at values fall inside the run-2
  window ✓. 011 gpu_runs.thermal (86 C, 20 pauses) = probe_analysis thermal ✓.
- 009 throughput note: "36.3 tok/s" = run1 8-slot mean 36.327 ✓; "3.05 s" = run1 server_load_s ✓;
  "22.5 tok/s" = median of the 24 re-run requests (22.454); the all-24 mean is 21.96 (observation O5).
- All 60 artifact_sha256 entries across 005/009/010/011 match the current files; embedded workloads
  contain the raw workload files unchanged; 009/010 gates files equal the receipt gates blocks. The
  model GGUF itself is not on this machine; its quoted sha256 could not be re-verified locally.

### 12. Consistency (PASS with the residuals already listed)

- Digest 012 vs receipts: 005 six lines (ratios, 4.5x/3.0x, 5/5/10/80 21%, 8–18x / 21–49x anchor
  ranges, 3-of-5 KL halving, similarity 0.71/98%) all match 005 and raw; 011 six lines (84/84+36/36
  both runs, subset margins, 33/16 think counts, mechanism open, 14-of-14 rollback + caveat +
  library-level, 13.1/12.7/0.4%, zero targets not tested, mixed preconditioning result) all match 011
  and raw. No contradictions found.
- FRONTIER E1/E2/E8 vs receipts and raw: all figures and scope clauses consistent (details in items
  1–4 above). E8 "does not decay" nit: R3-5. §5 review record accurately describes rounds 1–2 (24 + 11
  findings), the fix application and the rebuild order; its round-3 reference is anticipatory (R3-3).
- 013 status header: corrections 1–4 and all seven smaller corrections are verified folded into the
  rebuilt packets (items 1–10 above). Two accuracy caveats: "re-checked in a third, narrow review" was
  premature when written (R3-3), and "the rebuilt packets no longer disagree with it" is contradicted
  by 009's unchanged title, which still carries the exact claim correction 1 narrows (R3-1).
- New claims introduced by the revision: the digest's git-push sentence (R3-6 below); the
  "supersedes the 19:58 drop" header (accurate per HANDOFF ledger); 005 F4's expanded confidence
  sentence and the "different state set or a different tensor" disjunction (properly hedged, supported
  by the measured 62.70-vs-59.02 mismatch and the 0.2% agreement on the 332 shared states). No new
  numeric overclaim found in any of the 140 changed fields.

## Findings

| # | Severity | File(s) | Exact quote / location | Recomputed / verified fact | Suggested fix |
|---|---|---|---|---|---|
| R3-1 | Minor (framing; required before resend) | 009 `title` | "…: adapters reach every slot by log-probability test; the greedy-text gate fails and is shown to be the wrong test" | Round-2 finding 2 requires "an adapter-sized shift reached every request; the margin sees the formal-verification adapter only". Body, gate SCOPE clause, digest and FRONTIER are corrected; the title is the one remaining instance of the old claim, and it contradicts 013's "the rebuilt packets no longer disagree with it". | Retitle, e.g. "…: an adapter-sized shift reaches every request by log-probability test (on this prompt the margin sees the fv adapter only); the greedy-text gate fails and is shown to be the wrong test". Rebuild 009 (and its hash in any referencing list). |
| R3-2 | Minor (consistency; required before resend) | 005 `review` vs 013 header / 009 / 010 / 011 / 012 | 005: "The round-2 fixes were applied by the author and not independently re-checked." | True at rebuild time; now stale: this report exists, and every sibling packet plus 013 says the third review is recorded. The drop would ship two contradictory statements about its own review status. | Align 005's review field with the siblings: "…applied by the author; a third, narrow review of the revision is recorded in receipts/REVIEW_ROUND_3.md." (Alternatively soften 013's header; one of the two must change.) |
| R3-3 | Low (review-status wording) | 013 header; 012 header; FRONTIER §5; REVIEW_ROUND_2 status | 013: "…rebuilt … and re-checked in a third, narrow review (receipts/REVIEW_ROUND_3.md)." | Written 20:49–20:51, before any third review existed; the references were anticipatory. Satisfied by this report now, but the past-tense "re-checked" asserted a completed review that had not happened. | No change strictly required now that this file exists; for the record, future status headers should say "a third review is requested/under way" until it is complete. If 013 is resent unchanged, note that its header's claim is made true by this report. |
| R3-4 | Low (optional precision) | 009 note 8; 012 011-line-2; FRONTIER E2 table | "reproduces the runtime to 1e-05-1e-04 where exact arithmetic differs by 3e-03-9e-03" | True q8 range over the six matmuls: 1.388e-05 … **1.133e-04** (blk.31 ffn_down). Upper end is a 1-sig-fig rounding; as a strict bound it reads 13% low. Exact range 3.023e-03…8.628e-03 → "3e-03-9e-03" fine. | Optional: "1e-5 to 1.1e-4". Acceptable as-is under stated rounding. |
| R3-5 | Low (optional wording) | FRONTIER E8 | "The residue at blk.28..30 … about the size of batch-composition noise, and does not decay." | Medians fall k=1→4 (blk28 0.2978→0.1306; blk29 0.1921→0.1253; blk30 0.1170→0.1024) then persist to k=16 (0.1233/0.1098/0.1079). 011/digest wording ("still present 16 tokens later") is already exact. | Optional: "does not decay away within 16 tokens". |
| R3-6 | Low (provenance clarity) | 012 "What we need from you" item 4 | "This round's work is now pushed to Moissanite-Labs/ml-os on branch geminus/2026-11-19-m1-runtime-findings (pull request 6)…" | Branch push verified (remote-tracking ref; commit f6a50a3, 19:58:14). But that commit carries the **pre-fix** 009/010/011 (its message: "Round-2 fixes for 009/010/011 are still owed"), a 005 with two stale script hashes, and the old 013 - i.e. the superseded 19:58 drop. PR number itself not verifiable offline. A reader of PR 6 would build on superseded text (the pushed 013 does warn that fixes are partly owed). | Either update the branch with the rebuilt packets at resend, or add a clause: "PR 6 carries the 19:58 drop; the corrected revision follows." |

No new overclaim was found in any changed numeric or scope statement. The gate identifier
`lora_applied_on_every_slot_server_level_logprob_margin` (009) still names the old claim, but its
threshold text carries the SCOPE correction and identifiers are referenced elsewhere; renaming is
optional and not counted as a finding.

## Observations (no action required)

- **O1 - distinct-margin counting rule.** 34/14 counts distinct values over **all** 84 adapter / 36 W0
  results including the serialised mixed rounds (same at raw precision and 4 dp). Pure W0 rounds alone
  give 13; the 14th value is the mixed-round reference −0.818029. The receipts' effective-sample
  sentence is accurate under this rule; readers should not interpret 14 as 14 independent noise draws.
- **O2 - "margin sees fv only".** corpus+design contribute a sub-threshold +0.184-logit shift (fv alone
  +0.687, triple +0.896 vs W0). The claim is about classification, and 011 states the indistinguishability
  explicitly.
- **O3 - host-time convention.** Host events use whole-second board times against t0 truncated to
  16:40:05 (628/653/2425 exact); excursion times use the fractional first sustained sample
  (16:40:05.521). The mixed convention yields 13.8/38.8 s; exact deltas are 14.3/39.3 s; the prose
  "14 s"/"39 s" matches either. Disclosed in `host_activity.source` ("whole-second resolution").
- **O4 - drift estimator.** 0.029 C/min is the half-window mean-difference slope over t 120–660 s;
  least squares on the same window gives 0.020 C/min. Both support "drifting up". Pre-existing figure,
  untouched by round 2.
- **O5 - "22.5 tok/s per slot"** (009, pre-existing) reproduces as the **median** of the 24 re-run
  requests (22.454); the mean is 21.96. Optionally name the statistic.
- **O6 - "two Opus workers" (010) vs "three of six agents ran Opus" (012).** Consistent: two workers
  stopped on quota at t+628; the last working-agent post (t+653) is later; the monitor is a separate
  session. No contradiction.
- **O7 - builder mtimes.** make_005/make_011 edited 20:57, after the 20:51 rebuild; sandboxed rebuild
  reproduces all outputs byte-identically, so the edits are output-neutral. Rebuild order
  010→009→011→005 confirmed from mtimes.
- **O8 - 010 "which shows how much heat the chassis was holding"** is a mild inference on the measured
  61→85 C change; not flagged in earlier rounds; optional softening ("consistent with chassis heat soak").
- **Limits of this review.** weft `tangent.py` (0.01 default) and the E8 arm caps (0.20/0.05) are not
  verifiable on this machine (repos absent) - accepted as recorded in round 2. The model GGUF sha256
  and GitHub PR #6 could not be re-verified locally. The swarm-board absolute times (16:50:33 /
  16:50:58 / 17:20:30) were taken as given by the task and round-2 record; the CSV-relative arithmetic
  was verified. Bootstrap CIs and the design-pair token ids were verified in rounds 1–2 and are
  unchanged; not recomputed here.

## Verdicts

| Packet | Verdict | Reason |
|---|---|---|
| 005 | **SEND AFTER FIXES** | All numbers, all 11 script hashes and every round-2 text fix verified. One sentence: align the `review` field with the siblings / 013 (R3-2). Resend also corrects the two stale script hashes in the already-sent copy. |
| 009 | **SEND AFTER FIXES** | Everything verified except the title, which still carries the pre-correction claim (R3-1). One-line fix, rebuild, resend. |
| 010 | **SEND** | Every figure reproduces from the CSV/workload/stamp under disclosed conventions; all finding-10 framing corrections present; no residuals. |
| 011 | **SEND** | All round-2 corrections (fv-only, effective sample, v2 rollback + v1 caveat, open mechanism, eta attribution, lambda-10 median, zero-target-not-tested, review field) verified against raw; no residuals. |
| 012 | **SEND** | Consistent with the rebuilt receipts and raw data; required wording fixes present. Recommend addressing R3-6 (PR 6 carries the 19:58 drop) at resend, and note R3-3 for the record. |

After the two one-line fixes (009 title, 005 review field) and a rebuild in the order 010, 009, 011,
005, all five files are fit to resend as the revision. The reboot (round-2 step 5) still needs the owner.

## Scratch

`verify-d/`: d1_structure.py, d2_server.py, d3_rollback.py, d4_misc.py, d4b_fixups.py, d5_thermal.py,
d5b_windows.py, d6_smoke_005.py, d6b_005.py, d7_hashes.py, d8_diff.py, d9_final.py, d9b_tail.py,
d10_stage1.py; sandboxed builders `make_*_sandbox.py` and their byte-identical outputs in
`rebuild_out/`; `pushed_005.json` (extract of git f6a50a3). No file outside verify-d and this report
was modified.


## Author resolutions (appended by the takeover session, 2026-11-19 21:55 local)

| # | Resolution |
|---|---|
| R3-1 | Fixed: 009 retitled to "an adapter-sized shift reaches every request by log-probability test (on this prompt the margin sees the fv adapter only)"; 009 rebuilt. |
| R3-2 | Fixed: 005 review field now references receipts/REVIEW_ROUND_3.md like the sibling packets; 005 rebuilt. |
| R3-3 | Noted for the record: the status headers were anticipatory and are satisfied by this report. Future headers will say "under way" until the review completes. |
| R3-4 | Adopted: q8 upper end now 1.1e-4 in the 009 note, the digest and the FRONTIER E2 table. |
| R3-5 | Adopted: FRONTIER E8 now says "does not decay away within 16 tokens". |
| R3-6 | Resolved by action: the corrected revision is pushed to the same branch before the resend, so PR 6 carries the corrected packets, not the 19:58 drop; the digest says so explicitly. |

Punctuation of this report was normalized for the repo markdown gate (em dashes to hyphens,
one at-sign pattern reworded); no wording or numbers changed.

Rebuild after fixes: 009 then 005 (010 and 011 unchanged, verdicts SEND). The gate identifier
`lora_applied_on_every_slot_server_level_logprob_margin` keeps its name (referenced elsewhere);
its threshold text carries the SCOPE correction, as the reviewer noted.
