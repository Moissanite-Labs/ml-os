# Independent review, round 2 (2026-11-19, about 19:25 to 19:48) and where work stopped

A second reviewer rechecked the four receipts and the digest. All numbers reproduced again. All 24
round-1 findings were confirmed fixed or partly fixed. It raised 11 new points. Verdict on every
file: SEND AFTER FIXES. **Nothing has been sent.** Work paused at 19:55 on the owner's instruction
(usage cap) with the fixes below only partly applied.

## Round-2 findings

| # | Finding | State |
|---|---|---|
| 1 | 005 was stale: it hashed every script in `m1\scripts`, and scripts changed after it was built. | Builder patched to hash a fixed list. Not rebuilt. |
| 2 | The server-level test cannot tell three adapters from the formal-verification adapter alone. | New data taken (`server_logit_check_v2.json`): fv-only margin -0.131 (adapter side), corpus-only -0.925, design-only -0.534, corpus+design -0.634 (all W0 side). Claim must become "an adapter-sized shift reached every request; the margin sees the formal-verification adapter only". Not yet in 009/011/012. |
| 3 | The blk.31 "exact rollback" was true by construction: the prefix had no output rows, so the adapter never ran there. | Probe corrected and re-run (`rollback_probe_v2.json`): the adapter shifts prefix-end logits by a median of 1.6 to 1.9 while on, and rollback is exact for 14 of 14 at k = 1, 4, 16. Not yet in 011/012. Say "blk.31 FFN adapter", say it is library level, and note that llama-server already clears a slot's cache on a LoRA change. |
| 4 | The digest and the `review` fields misdescribed the review. | 005 field patched. 011 field and digest 012 still to do. |
| 5 | Held-out similarity was understated (difflib autojunk). | Fixed and recomputed: median 0.71, 37.5% above 0.8. Flows into 005 on rebuild. |
| 6 | The mixed server rounds were necessarily serialised, so they show config switching only. Effective sample: 34 distinct adapter margins and 14 distinct W0 margins in the v2 run. | To state in 009 and 011. |
| 7 | The noise mechanism is incomplete: eight identical sequences in one batch still differ row to row, which kernel choice by batch size cannot explain. | To state in 011 section B and 009. |
| 8 | "Your 0.01 cap" is weft's default, not a rule the team set; the E8 arms use 0.20 and 0.05. | 005 patched. 011 `eta_note` still to do. |
| 9 | Prose numbers: 005 gain range omits blk.30 (0.84); "0.09" noise floor has no stated source (it is sqrt(0.116² - 0.069²)); 009 note omits `ffn_gate` from a range; 011 quotes a lambda-1 median in a lambda-10 sentence; 010 says 95 W was not exercised though samples reach 96.6 W. | 005 patched. Others to do. |
| 10 | Inference worded as fact: 010 "plateau measured during CPU work" (CPU load is unknown; the stall monitor kept posting until 17:20; "no second GPU process" was not recorded); 012 "because preservation keys cover one position"; 005 F4 "exact"; 005 "therefore a different state set"; 011 D "held at baseline"; 012 "the Opus swarm" (3 of 6 agents were Opus). | 005 patched. 010, 011, 012 to do. |
| 11 | Benign: `127.0.0.1` and ephemeral ports in 009/010; the harness label `w0_identical` (means first 60 characters match) appears in embedded raw data. | Add one explanatory note in 009. |

## Exact resume steps

1. Apply the remaining text fixes to `make_011.py`, `make_009.py`, `make_010.py` and the digest.
   `make_011.py` must read `server_logit_check_v2.json` and `rollback_probe_v2.json`
   (keep v1 of the server run as a replicate: both runs give 84/84 and 36/36).
   A patch for `make_011.py` was written and failed on shell quoting before it ran; the file is
   unchanged. Use the Edit tool, not a shell heredoc.
2. Rebuild in this order: 010, 009, 011, then 005 last.
3. Update `FRONTIER.md` sections E1 (adapter-sized shift; fv only), E2 (mechanism incomplete),
   E8 (v2 figures: 14 of 14, adapter verified to run on the prefix) and the review record.
4. Decide whether to run a third, narrow review of only these changes. Two rounds each found
   problems, so it is probably worth it. Then send the five files with
   `tailscale file cp <files> <MAC_NODE>:` and fill in the ledger in `HANDOFF.md`.
5. Then ask the owner about the reboot.
