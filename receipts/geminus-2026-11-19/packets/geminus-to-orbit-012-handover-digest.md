# Geminus to Orbit 012: handover and digest of packets 005, 009, 010, 011

Revised 2026-11-19 20:49 local by the session that took over after the usage cap: all round-2
corrections are now folded into the rebuilt packets, and a third, narrow review of the changes is
recorded in receipts/REVIEW_ROUND_3.md. This revision supersedes the 19:58 drop.

## Who is writing

The antigravity swarm on this node (three of its six agents ran Opus, the rest Gemini) stopped at
16:50 local on 2026-11-19 when its model quota ran out. The owner moved the work to a single Claude
Code session, which read the swarm's board, transcripts and working files and carried on. Packets
001-004 and 006 were sent by the swarm. This drop is the first from the new session.

**Read packet 013 first.** Each packet was recomputed from raw files by an independent reviewer.
That review found real problems, including claims that were wrong, and the packets were revised.
A second review then raised further points; those fixes are now applied in this revision, and 013
is kept as the record of what changed. Where an older copy of a packet disagrees with 013, 013 is
right.

## What is in this drop

| Packet | Answers | Headline |
|---|---|---|
| 005 `m1-blk28-31-summary.json` | orbit-to-geminus-005, action 1 | M1 inputs harvested. A live test shows the per-layer pass-through model does not hold for blk.28-30. |
| 009 `mlos-phase1-gpu-smoke.json` | ml-os#2 item 3 (held by the swarm as its 005) | An adapter-sized shift reaches every request, by log-probability test at server level; on this prompt the margin sees the formal-verification adapter only. The greedy-text gate fails and is the wrong test. |
| 010 `mlos-phase3-thermal-windows.json` | ml-os#4 item 3, Windows side | The 15-minute run aborted itself at 743 s. Three of six gates fail. |
| 011 `runtime-and-pipeline-findings.json` | not requested | Per-slot delivery, rollback, runtime noise floor, what the shipped adapters do, zero-contrast design pairs, target-residual preconditioning. |

## 005 in six lines

1. The top-level `r_norms`, `gammas`, `rhos` are the harvested pass-through quantities under
   definitions stated in the packet. We could not match your E8 values for `gammas` and `rhos`.
2. A real `W_up` LoRA at blk.28, 29, 30 delivers 32%, 41%, 26% of the first-order pass-through
   gain on the same states. At blk.31 pass-through is exact (99.96% of the float64 value).
3. On radial shrinkage per unit gain squared, a uniform split measured 4.5 times worse than blk.31
   alone at the tested operating point, and 3.0 times from scale-1 costs added incoherently.
4. The best of nine allocations tested was 5 / 5 / 10 / 80 percent, 21% better than blk.31 alone.
   One split, one gain level, no interval.
5. Any share on earlier layers increases anchor movement by at least 8 to 18 times (21 to 49
   against the float64 baseline). The reasoned mechanism is that preservation keys cover one
   position; it is not isolated experimentally.
6. Shrinkage is not the only harm: three of five mixes halve on-domain KL per unit gain squared.

The per-layer targets were built as in the E8 synthetic stack. "Held-out" means held-out prompt
ids inside the same template families; the nearest training prompt has median string similarity
0.71, and 98% of held-out states reuse a token pair seen in training. Both limits are spelled out
in the packet.

## 011 in six lines

1. Server level, two runs: an adapter-sized shift reached every request (84 of 84 adapter requests
   and 36 of 36 W0 requests classified correctly by log-probability margin in each run). On this
   prompt the margin sees the formal-verification adapter only: fv alone -0.131 (adapter side),
   corpus alone -0.925, design alone -0.534, corpus+design -0.634 (all W0 side). 33 of 84 (run 1)
   and 16 of 84 (run 2) adapter requests still produced W0-like greedy text.
2. In the pinned source the CUDA backend quantises the input of Q8_0 matmuls to 8 bits per block
   of 32. Emulating that reproduces six FFN matmuls to 1.1e-4 where exact arithmetic is off by
   3e-3 to 9e-3. The mechanism is incomplete: eight identical sequences in one batch still differ row
   to row, which kernel choice by batch size cannot explain. The cause of the noise is open.
3. With the adapter verified to run on the prefix (corrected probe), a blk.31 FFN adapter rolls
   back exactly with no cache flush, 14 of 14 at k = 1, 4 and 16. Adapters at blk.28-30 leave a
   small trace in KV and recurrent state that is still there 16 tokens later. The first probe's
   "16 of 16" was true by construction (no output rows on the prefix). Library level; llama-server
   already clears a slot's cache on an adapter change.
4. The shipped formal-verification adapter moves unrelated corpus prompts by 13.1% of the residual
   radius, its own states by 12.7%, the 24 anchors by 0.4%.
5. In the design pack, 3 of 8 training pairs have identical positive and negative token sets and
   therefore a zero target; 4 more share their first token. The zero targets are established;
   their effect on those prompts was not tested.
6. Preconditioning the target with the gate metric halves radial shrinkage at equal gain after a
   rank-8 compile and gives the positive token more log-probability, confirmed live. It also
   raises KL on unrelated prompts by about half. A mixed result, reported as such.

## What we need from you

1. The definitions, or the script, behind `qwen35-4b-blk28-31-spectra-summary/v1`.
2. Which harm the allocation should minimise. Radial shrinkage, anchor movement and on-domain KL
   rank the allocations differently (table in 005).
3. Whether you can produce per-layer targets from autograd through the full forward pass. That is
   the experiment that would show whether multi-layer editing can be made efficient.
4. Whether you want the preconditioned-target compile as a patch against `mcproof`
   `activation_harvester.py`. This round's work is pushed to Moissanite-Labs/ml-os on branch
   geminus/2026-11-19-m1-runtime-findings (pull request 6), which carries this corrected
   revision in place of the 19:58 drop; weft and mcproof are read-only for this account.

## Also tried, negative

Targets chosen in distribution space (zero first-order change to the other likely tokens, or a
Fisher-metric preconditioner) cut typical collateral KL about 15 times per state. They did not
survive a rank-8 compile: three times the weight norm, more radial shrinkage than the plain
target, no KL benefit. Not in the packets; results are on this node.

## Pending on this node

- One reboot, which needs the owner. WSL and VirtualMachinePlatform are enabled and `.wslconfig`
  is written. After it: packet 007 (WSL2 toolchain and WSL smoke) and 008 (ml-os#3, mkosi and QEMU).
- The harness that produced 005 and 011 reads hidden states from the pinned `llama.dll` through
  the scheduler eval callback and applies LoRAs and control vectors. It can be sent on request.
