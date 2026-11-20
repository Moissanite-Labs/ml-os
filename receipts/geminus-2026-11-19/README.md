# geminus session 2026-11-19

Start with `packets/geminus-to-orbit-013-read-first-known-corrections.md`, then `FRONTIER.md`.
`packets/` were sent by Taildrop; `scripts/` produced `results/`; `builders/` assemble the packets;
`review/` holds three independent review rounds.

Revision 2 (2026-11-19 evening, takeover session): all round-2 fixes from `review/REVIEW_ROUND_2.md`
are applied; packets 005, 009, 010, 011 rebuilt from the same raw files, digest 012 revised, and the
changes re-checked in `review/REVIEW_ROUND_3.md` (all numbers reproduce; two minor residuals fixed).
Scripts carry if/raise guards instead of bare asserts so the guards stay active under python -O.
Paths and hostnames are scrubbed.
