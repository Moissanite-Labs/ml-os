"""M1 activation harvest for Qwen3.5-4B blk.28..31 on the pinned GGUF + pinned llama.cpp runtime.

Mirrors the state selection of mcproof `compile/src/mcproof_compile/activation_harvester.py`
(same public datasets, same chat rendering, same decision positions) but
  * reads the content-addressed Q8_0 GGUF through llama.dll instead of bf16 safetensors, and
  * captures all four target layers instead of layer 31 only.

State groups
  fv      : decision states of datasets/compile.jsonl, positions taken verbatim from the team's
            harvested_activations.meta.json (id, rel_step, pos_toks, neg_toks)
  corpus  : 8 prose pairs, last prompt position            (text copied from the team harvester)
  eido    : 8 design pairs, last prompt position           (text copied from the team harvester)
  anchor  : 24 y=0 preservation anchors, chat-rendered, last 128 tokens, last position
  noise   : seeded random-vocab sequences (u32 % n_vocab), last position -> K0 keys

Per decision state and layer l: attn_residual (pre-FFN residual), attn_post_norm (W_up/W_gate key),
ffn_gate (gate pre-activation), ffn_up, ffn_out, l_out; plus l_out-27, result_norm, top logits.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np

SCRIPTS, BIN, MODEL, DATASETS, META, OUT_PREFIX = sys.argv[1:7]
N_NOISE = int(sys.argv[7]) if len(sys.argv) > 7 else 2048
sys.path.insert(0, SCRIPTS)
from llama_ctypes import Llama  # noqa: E402
from render import ChatTemplate  # noqa: E402

LAYERS = (28, 29, 30, 31)
LAST = 31
TOPK = 32

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


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.time()
    tpl = ChatTemplate(MODEL)
    ll = Llama(BIN)
    layout = ll.check_layout()
    ll.load(MODEL, n_gpu_layers=99, n_ctx=2048, n_batch=2048)
    n_vocab = ll.n_vocab

    compile_rows = {r["id"]: r for r in (json.loads(l) for l in open(DATASETS + "/compile.jsonl", encoding="utf-8") if l.strip())}
    k0_rows = [json.loads(l) for l in open(DATASETS + "/hybrid_k0.jsonl", encoding="utf-8") if l.strip()]
    meta = json.load(open(META, encoding="utf-8"))

    # ---- build jobs ---------------------------------------------------------------------
    jobs = []  # dict(group, id, tokens, positions[list], per-position pos/neg toks, extra)
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
        seen = set()
        st = []
        for s in sorted(steps, key=lambda z: z["rel_step"]):
            p = base + s["rel_step"]
            if p < 0 or p >= len(full) or p in seen:
                continue
            seen.add(p)
            st.append(dict(pos=p, rel_step=s["rel_step"], pos_toks=s["pos_toks"], neg_toks=s["neg_toks"],
                           ref_ffn_out_norm=s["ffn_out_norm"], family=s["family"]))
        jobs.append(dict(group="fv", id=rid, tokens=full, states=st, profile=False))

    for gname, pairs in (("corpus", CORPUS), ("eido", EIDO)):
        for i, (pr, good, bad) in enumerate(pairs):
            p_ids = ll.tokenize(tpl.render_user(pr))
            g_ids = ll.tokenize(good)
            b_ids = ll.tokenize(bad)
            jobs.append(dict(group=gname, id=f"{gname}-{i}", tokens=p_ids, profile=True,
                             states=[dict(pos=len(p_ids) - 1, rel_step=0, pos_toks=g_ids[:2], neg_toks=b_ids[:2],
                                          ref_ffn_out_norm=None, family=gname)]))

    noise_seqs = []
    for rec in k0_rows:
        if rec.get("y0") or rec.get("kind") == "y0_anchor":
            ids = ll.tokenize(tpl.render_user(str(rec["prompt"])))[-128:]
            jobs.append(dict(group="anchor", id=rec["id"], tokens=ids, profile=True,
                             states=[dict(pos=len(ids) - 1, rel_step=0, pos_toks=[], neg_toks=[],
                                          ref_ffn_out_norm=None, family="anchor")]))
        else:
            noise_seqs.append((rec["id"], [int(t) % n_vocab for t in rec["u32"][:32]]))
    noise_seqs = noise_seqs[:N_NOISE]
    n_states = sum(len(j["states"]) for j in jobs)
    print(f"jobs: {len(jobs)} sequences / {n_states} decision states; noise sequences: {len(noise_seqs)}", flush=True)

    # ---- decision-state harvest ---------------------------------------------------------
    S = n_states
    d, dff = ll.n_embd, 9216
    A = {
        "h_mid": np.zeros((S, 4, d), np.float32), "key": np.zeros((S, 4, d), np.float32),
        "gate_pre": np.zeros((S, 4, dff), np.float32), "up": np.zeros((S, 4, dff), np.float32),
        "ffn_out": np.zeros((S, 4, d), np.float32), "l_out": np.zeros((S, 4, d), np.float32),
        "l_out27": np.zeros((S, d), np.float32), "result_norm": np.zeros((S, d), np.float32),
        "top_ids": np.zeros((S, TOPK), np.int32), "top_logits": np.zeros((S, TOPK), np.float32),
    }
    rows_meta = []
    profile = []  # all-position residual norm profiles for short chat prompts
    si = 0
    t0 = time.time()
    for ji, job in enumerate(jobs):
        toks = job["tokens"]
        pos_list = sorted(s["pos"] for s in job["states"])
        outs = list(range(len(toks))) if job["profile"] else pos_list
        idx_in_out = {p: outs.index(p) for p in pos_list}
        want = {"l_out-27": "all" if job["profile"] else pos_list, "__logits__": True}
        for l in LAYERS:
            cropped = (l == LAST)
            rsel = [idx_in_out[p] for p in pos_list] if cropped else pos_list
            if job["profile"]:
                want[f"attn_residual-{l}"] = "all"
                want[f"l_out-{l}"] = "all"
            else:
                want[f"attn_residual-{l}"] = rsel
                want[f"l_out-{l}"] = rsel
            for stem in ("attn_post_norm", "ffn_gate", "ffn_up", "ffn_out"):
                want[f"{stem}-{l}"] = rsel
        want["result_norm"] = [idx_in_out[p] for p in pos_list]
        res = ll.decode(toks, want, outs)
        if job["profile"]:
            prof = {"id": job["id"], "group": job["group"], "n_tokens": len(toks)}
            prof["l_out-27"] = np.linalg.norm(res["l_out-27"].astype(np.float64), axis=1).tolist()
            for l in LAYERS:
                prof[f"attn_residual-{l}"] = np.linalg.norm(res[f"attn_residual-{l}"].astype(np.float64), axis=1).tolist()
                prof[f"l_out-{l}"] = np.linalg.norm(res[f"l_out-{l}"].astype(np.float64), axis=1).tolist()
            profile.append(prof)
        for k, st in enumerate(sorted(job["states"], key=lambda z: z["pos"])):
            p = st["pos"]
            A["l_out27"][si] = res["l_out-27"][p if job["profile"] else k]
            for li, l in enumerate(LAYERS):
                # residual tensors: row index depends on profile mode / crop
                if job["profile"]:
                    r_res = p  # all rows captured (layer 31: outs == all positions)
                else:
                    r_res = k
                A["h_mid"][si, li] = res[f"attn_residual-{l}"][r_res]
                A["l_out"][si, li] = res[f"l_out-{l}"][r_res]
                A["key"][si, li] = res[f"attn_post_norm-{l}"][k]
                A["gate_pre"][si, li] = res[f"ffn_gate-{l}"][k]
                A["up"][si, li] = res[f"ffn_up-{l}"][k]
                A["ffn_out"][si, li] = res[f"ffn_out-{l}"][k]
            A["result_norm"][si] = res["result_norm"][k]
            lg = res["__logits__"][idx_in_out[p]]
            top = np.argpartition(-lg, TOPK)[:TOPK]
            top = top[np.argsort(-lg[top])]
            A["top_ids"][si] = top
            A["top_logits"][si] = lg[top]
            rows_meta.append(dict(state=si, group=job["group"], id=job["id"], pos=p, n_tokens=len(toks),
                                  rel_step=st["rel_step"], family=st["family"], pos_toks=st["pos_toks"],
                                  neg_toks=st["neg_toks"], ref_ffn_out_norm=st["ref_ffn_out_norm"],
                                  logit_pos=[float(lg[t]) for t in st["pos_toks"]],
                                  logit_neg=[float(lg[t]) for t in st["neg_toks"]],
                                  next_token=int(toks[p + 1]) if p + 1 < len(toks) else None))
            si += 1
        if (ji + 1) % 20 == 0 or ji + 1 == len(jobs):
            print(f"  seq {ji+1}/{len(jobs)} states {si}/{S} elapsed {time.time()-t0:.0f}s", flush=True)
    assert si == S

    # ---- K0 noise harvest (last position) -----------------------------------------------
    N = len(noise_seqs)
    NZ = {"key": np.zeros((N, 4, d), np.float32), "h_mid_norm": np.zeros((N, 4), np.float64),
          "l_out_norm": np.zeros((N, 5), np.float64), "ffn_out_norm": np.zeros((N, 4), np.float64),
          "l_out_last": np.zeros((N, d), np.float32)}
    t0 = time.time()
    for ni, (nid, ids) in enumerate(noise_seqs):
        p = len(ids) - 1
        want = {"l_out-27": [p]}
        for l in LAYERS:
            r = [0] if l == LAST else [p]
            for stem in ("attn_residual", "attn_post_norm", "ffn_out", "l_out"):
                want[f"{stem}-{l}"] = r
        res = ll.decode(ids, want, [p])
        NZ["l_out_norm"][ni, 0] = np.linalg.norm(res["l_out-27"][0].astype(np.float64))
        for li, l in enumerate(LAYERS):
            NZ["key"][ni, li] = res[f"attn_post_norm-{l}"][0]
            NZ["h_mid_norm"][ni, li] = np.linalg.norm(res[f"attn_residual-{l}"][0].astype(np.float64))
            NZ["ffn_out_norm"][ni, li] = np.linalg.norm(res[f"ffn_out-{l}"][0].astype(np.float64))
            NZ["l_out_norm"][ni, li + 1] = np.linalg.norm(res[f"l_out-{l}"][0].astype(np.float64))
        NZ["l_out_last"][ni] = res[f"l_out-{LAST}"][0]
        if (ni + 1) % 256 == 0 or ni + 1 == N:
            print(f"  noise {ni+1}/{N} elapsed {time.time()-t0:.0f}s", flush=True)
    ll.close()

    npz_path = OUT_PREFIX + ".npz"
    np.savez(npz_path, **{f"s_{k}": v for k, v in A.items()}, **{f"n_{k}": v for k, v in NZ.items()})
    info = {
        "schema": "geminus-m1-harvest/v1",
        "model_file": "Qwen3.5-4B-Q8_0.gguf", "model_sha256": sha256_file(MODEL),
        "runtime": {"llama_cpp_build": "b12549", "llama_cpp_commit": "8af68d486", "backend": "CUDA (ggml-cuda.dll, cu12.4)",
                    "binding": "ctypes on llama.dll, scheduler eval callback", "n_gpu_layers": 99,
                    "struct_layout_check": layout},
        "layers": list(LAYERS), "d_model": d, "d_ff": dff, "n_vocab": n_vocab,
        "tensor_names": {"h_mid": "attn_residual-l", "key": "attn_post_norm-l", "gate_pre": "ffn_gate-l",
                         "up": "ffn_up-l", "ffn_out": "ffn_out-l", "l_out": "l_out-l", "l_out27": "l_out-27"},
        "n_decision_states": S, "n_noise": N, "elapsed_s": round(time.time() - t_start, 1),
        "npz_sha256": sha256_file(npz_path),
        "states": rows_meta, "norm_profiles": profile,
        "noise_ids": [nid for nid, _ in noise_seqs],
    }
    with open(OUT_PREFIX + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    print("saved", npz_path, "sha256", info["npz_sha256"], "elapsed", info["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
