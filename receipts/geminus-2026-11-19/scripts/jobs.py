"""Shared calibration-job builder (identical state selection to harvest.py; verified against its meta)."""
from __future__ import annotations

import json

CORPUS = [
    ("Write a status update explaining that the parser test suite passed.", "The parser tests passed.", "Notably, it is worth noting — the parser tests passed;"),
    ("Summarize why the build failed on missing configuration.", "The build failed because config.json is missing.", "At its core — the build failed;"),
    ("Explain what a SHA-256 digest checks in one plain sentence.", "A SHA-256 digest checks that file bytes did not change.", "Importantly — a SHA-256 digest verifies provenance;"),
    ("State the review result when two unit tests fail.", "Two unit tests failed in the suite.", "I would say this is a step in the right direction —"),
    ("Describe how the rollback switch works.", "The rollback switch resets adapter scales to zero and clears slot caches.", "Put simply — the rollback choreography resets durable state;"),
    ("Report the memory measurement from the benchmark.", "Peak memory reached 5.1 GiB during the run.", "It turns out that memory — notably — stayed flat;"),
    ("Write the opening line of the README for a proof runner.", "mcproof runs Lean 4 and exact Python checks on formal claims.", "At its core, mcproof is a high-quality end-to-end framework —"),
    ("Frame two options for handling a slow test.", "Option 1 splits the slow test. Option 2 caches the fixture.", "Here are a few thoughts — worth considering;"),
]
EIDO = [
    ("Report the WCAG text contrast check for body text on light wash.", "Contrast ratio is 4.99:1, passing WCAG AA >= 4.5:1.", "Contrast ratio is 3.40:1 with innerHTML."),
    ("Write a safe DOM update for a status badge.", "badge.textContent = label;", "badge.innerHTML = label;"),
    ("Choose a perceptual color space for circular hue interpolation.", "Use OKLCH cylindrical coordinates for uniform lightness across 360 hues.", "Use raw sRGB hex interpolation without contrast checks."),
    ("Scale circle radius for area-proportional chart marks.", "const radius = Math.sqrt(value) * scale;", "const radius = value * scale;"),
    ("Protect a formal proof block inside an HTML research page from prose condensing.", "<section data-eido-skip=\"true\">", "<section class=\"condensed\">"),
    ("Audit light-theme accent text against wash background.", "Light accent scores 3.59:1 (< 4.5:1 WCAG AA) and requires darkening.", "Light accent passes without checking contrast."),
    ("Export a zero-dependency ES module helper for contrast ratio.", "export function contrastRatio(fgL, bgL) {", "const contrastRatio = require('contrast');"),
    ("Bind an illustrative float64 canvas demo to its exact certificate.", "<figure data-illustrative=\"true\" data-certificate-sha256=\"", "<canvas class=\"unverified-math\">"),
]


def build_jobs(ll, tpl, datasets_dir: str, team_meta_path: str):
    compile_rows = {r["id"]: r for r in (json.loads(l) for l in open(datasets_dir + "/compile.jsonl", encoding="utf-8") if l.strip())}
    k0_rows = [json.loads(l) for l in open(datasets_dir + "/hybrid_k0.jsonl", encoding="utf-8") if l.strip()]
    meta = json.load(open(team_meta_path, encoding="utf-8"))
    jobs = []
    by_id: dict[str, list[dict]] = {}
    for s in meta["step_metadata"]:
        by_id.setdefault(s["id"], []).append(s)
    for rid, steps in by_id.items():
        rec = compile_rows[rid]
        prompt_ids = ll.tokenize(tpl.render_user(str(rec["prompt"])))
        pos_ids = ll.tokenize(rec["positive_completion"])
        max_step = max(s["rel_step"] for s in steps)
        full = prompt_ids + pos_ids[: max_step + 1]
        base = len(prompt_ids) - 1
        if len(full) > 1536:
            trim = len(full) - 1536
            full = full[trim:]
            base -= trim
        seen, st = set(), []
        for s in sorted(steps, key=lambda z: z["rel_step"]):
            p = base + s["rel_step"]
            if p < 0 or p >= len(full) or p in seen:
                continue
            seen.add(p)
            st.append(dict(pos=p, pos_toks=s["pos_toks"], neg_toks=s["neg_toks"]))
        jobs.append(dict(group="fv", id=rid, tokens=full, states=st))
    for gname, pairs in (("corpus", CORPUS), ("eido", EIDO)):
        for i, (pr, good, bad) in enumerate(pairs):
            p_ids = ll.tokenize(tpl.render_user(pr))
            jobs.append(dict(group=gname, id=f"{gname}-{i}", tokens=p_ids,
                             states=[dict(pos=len(p_ids) - 1, pos_toks=ll.tokenize(good)[:2], neg_toks=ll.tokenize(bad)[:2])]))
    for rec in k0_rows:
        if rec.get("y0") or rec.get("kind") == "y0_anchor":
            ids = ll.tokenize(tpl.render_user(str(rec["prompt"])))[-128:]
            jobs.append(dict(group="anchor", id=rec["id"], tokens=ids, states=[dict(pos=len(ids) - 1, pos_toks=[], neg_toks=[])]))
    return jobs


def flatten_states(jobs):
    """(job_index, state_dict) in harvest order."""
    out = []
    for ji, job in enumerate(jobs):
        for st in sorted(job["states"], key=lambda z: z["pos"]):
            out.append((ji, st))
    return out
