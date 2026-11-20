"""GPU probe on the pinned runtime (one short job, four questions).

P1  Noise floor: are logits reproducible, and how much do they move with batch composition?
P2  Does a LoRA delta reach EVERY output row of a multi-sequence batch, including the exact
    mixed composition seen in the 8-slot server log (three sequences prefilled 55+4 tokens,
    five submitted whole)? Per-row test: correlation of (logits_lora - logits_w0) with the
    single-sequence reference delta over the whole vocabulary.
P3  What do the three shipped blk.31 adapters actually do on their own training states?
    Live final-residual change vs the exact float64 prediction W_down (s * (B A x)).
P4  How does a perturbation injected after layers 28..31 propagate to the final residual?
    (tests the residual pass-through assumption used by any per-layer allocation model)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np

SCRIPTS, BIN, MODEL, HARVEST_PREFIX, DATASETS, META, LORA_DIR, OUT_PREFIX = sys.argv[1:9]
sys.path.insert(0, SCRIPTS)
from ggufw import Weights  # noqa: E402
from jobs import build_jobs, flatten_states  # noqa: E402
from llama_ctypes import Llama  # noqa: E402
from render import ChatTemplate  # noqa: E402

LAYERS = [28, 29, 30, 31]
EPS = 1e-6
SMOKE_PROMPT = (
    "You are a formal verification assistant. Explain, step by step, how to prove in Lean 4 that "
    "the sum of the first n odd natural numbers equals n squared, including the induction step, "
    "the key rewriting lemmas, and how `simp` and `ring` close the remaining goals."
)
ADAPTERS = ["corpus", "weft-eido", "formal-verification"]


def silu(z):
    return z / (1.0 + np.exp(-z))


def gpu_temp():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        return int(o[0]), int(o[1])
    except Exception:
        return None, None


class Thermal:
    def __init__(self, hi=85, lo=75):
        self.hi, self.lo, self.pauses, self.max_seen, self.n = hi, lo, [], 0, 0

    def tick(self, every=40):
        self.n += 1
        if self.n % every:
            return
        t, _ = gpu_temp()
        if t is None:
            return
        self.max_seen = max(self.max_seen, t)
        if t >= self.hi:
            t0 = time.time()
            while time.time() - t0 < 240:
                time.sleep(3)
                t2, _ = gpu_temp()
                if t2 is not None and t2 <= self.lo:
                    break
            self.pauses.append({"at_decode": self.n, "temp_c": t, "paused_s": round(time.time() - t0, 1)})


def top_info(ll, lg, k=5):
    idx = np.argsort(-lg)[:k]
    return [[int(i), ll.piece(int(i)), round(float(lg[i]), 4)] for i in idx]


def main():
    t_all = time.time()
    out: dict = {"schema": "geminus-runtime-probe/v1"}
    thermal = Thermal()
    lora_paths = [f"{LORA_DIR}/{n}.f16.gguf" for n in ADAPTERS]

    # =============================== P1 / P2 : multi-sequence context ===============================
    ll = Llama(BIN)
    ll.check_layout()
    ll.load(MODEL, n_gpu_layers=99, n_ctx=8 * 256, n_batch=2048, n_seq_max=8)
    out["gpu_offload_multi"] = {k: ll.gpu_offload[k] for k in ("offloaded_layers", "offloadable_layers", "vram_taken_mib")}
    for p in lora_paths:
        ll.load_lora(p)
    T = ll.tokenize(SMOKE_PROMPT, add_special=True, parse_special=True)
    nl2 = ll.tokenize("\n\n")
    fence = ll.tokenize("```")
    think = ll.tokenize("<think>")
    out["smoke_prompt"] = {"n_tokens": len(T), "nl2": nl2, "fence": fence, "think": think,
                           "pieces": {"nl2": [ll.piece(t) for t in nl2], "fence": [ll.piece(t) for t in fence], "think": [ll.piece(t) for t in think]}}
    assert len(nl2) == 1 and len(fence) == 1 and len(think) == 1, (nl2, fence, think)
    NL, FENCE, THINK = nl2[0], fence[0], think[0]
    n = len(T)
    ALL = list(range(8))

    def variant(name):
        """returns {seq: (first_token_logits, second_token_logits)}"""
        if name == "single_whole":
            a = ll.decode_multi([(0, T, 0, True)])
            b = ll.decode_multi([(0, [NL], n, True)], clear=False)
            return {0: (a[0], b[0])}
        if name == "single_split_55_4":
            ll.decode_multi([(0, T[:55], 0, False)])
            a = ll.decode_multi([(0, T[55:], 55, True)], clear=False)
            b = ll.decode_multi([(0, [NL], n, True)], clear=False)
            return {0: (a[0], b[0])}
        if name == "all8_whole":
            a = ll.decode_multi([(s, T, 0, True) for s in ALL])
            b = ll.decode_multi([(s, [NL], n, True) for s in ALL], clear=False)
        elif name == "all8_split_55_4":
            ll.decode_multi([(s, T[:55], 0, False) for s in ALL])
            a = ll.decode_multi([(s, T[55:], 55, True) for s in ALL], clear=False)
            b = ll.decode_multi([(s, [NL], n, True) for s in ALL], clear=False)
        elif name == "mixed_server_log":
            # slots 1-3 prefilled 55 tokens as their own batch; their last 4 tokens share a batch
            # with the five other slots' whole prompts; then all 8 advance one token in lockstep
            ll.decode_multi([(s, T[:55], 0, False) for s in (1, 2, 3)])
            a = ll.decode_multi([(s, T[55:], 55, True) for s in (1, 2, 3)] + [(s, T, 0, True) for s in (0, 4, 5, 6, 7)], clear=False)
            b = ll.decode_multi([(s, [NL], n, True) for s in ALL], clear=False)
        else:
            raise ValueError(name)
        return {s: (a[s], b[s]) for s in ALL}

    VARIANTS = ["single_whole", "single_split_55_4", "all8_whole", "all8_split_55_4", "mixed_server_log"]
    CONDS = {"w0": {}, "triple": {0: 1.0, 1: 1.0, 2: 1.0}, "fv_solo": {2: 1.0}}
    logits = {}
    for cname, scales in CONDS.items():
        ll.set_loras(scales)
        for v in VARIANTS:
            logits[(cname, v)] = variant(v)
            thermal.tick(every=1)
    # reproducibility: repeat the reference twice more
    ll.set_loras({})
    rep = [variant("single_whole")[0][1] for _ in range(2)]
    ref_w0 = logits[("w0", "single_whole")][0]
    ref_tr = logits[("triple", "single_whole")][0]
    p1 = {"repeat_single_whole_w0_max_abs_logit_diff": [float(np.max(np.abs(r - ref_w0[1]))) for r in rep],
          "reference_w0": {"first_token_top5": top_info(ll, ref_w0[0]), "second_token_top5": top_info(ll, ref_w0[1]),
                           "fork_margin_fence_minus_think": float(ref_w0[1][FENCE] - ref_w0[1][THINK])},
          "reference_triple": {"first_token_top5": top_info(ll, ref_tr[0]), "second_token_top5": top_info(ll, ref_tr[1]),
                               "fork_margin_fence_minus_think": float(ref_tr[1][FENCE] - ref_tr[1][THINK])}}
    ref_delta = {c: (logits[(c, "single_whole")][0][1].astype(np.float64) - ref_w0[1].astype(np.float64)) for c in ("triple", "fv_solo")}
    p1["reference_lora_delta_second_token"] = {c: {"l2_over_vocab": float(np.linalg.norm(dv)), "max_abs": float(np.max(np.abs(dv))),
                                                  "delta_fork_margin": float(dv[FENCE] - dv[THINK])} for c, dv in ref_delta.items()}
    rows = []
    for v in VARIANTS:
        for s in sorted(logits[("w0", v)].keys()):
            w0_2 = logits[("w0", v)][s][1].astype(np.float64)
            rec = {"variant": v, "seq": s,
                   "w0_fork_margin": float(w0_2[FENCE] - w0_2[THINK]), "w0_top1": int(np.argmax(w0_2)),
                   "w0_noise_vs_ref_l2": float(np.linalg.norm(w0_2 - ref_w0[1])), "w0_noise_vs_ref_max_abs": float(np.max(np.abs(w0_2 - ref_w0[1])))}
            for c in ("triple", "fv_solo"):
                l2 = logits[(c, v)][s][1].astype(np.float64)
                dv = l2 - w0_2
                rd = ref_delta[c]
                rec[f"{c}_fork_margin"] = float(l2[FENCE] - l2[THINK])
                rec[f"{c}_top1"] = int(np.argmax(l2))
                rec[f"{c}_delta_l2"] = float(np.linalg.norm(dv))
                rec[f"{c}_delta_corr_with_ref_delta"] = float(np.dot(dv, rd) / (np.linalg.norm(dv) * np.linalg.norm(rd) + 1e-300))
                rec[f"{c}_delta_projection_on_ref"] = float(np.dot(dv, rd) / (np.dot(rd, rd) + 1e-300))
            rows.append(rec)
    out["p1_noise_floor"] = p1
    out["p2_lora_rows"] = rows
    out["token_ids"] = {"fence": FENCE, "think": THINK, "nl2": NL}
    ll.close()
    print(f"P1/P2 done at {time.time()-t_all:.0f}s", flush=True)

    # =============================== P3 / P4 : single-sequence context ===============================
    tpl = ChatTemplate(MODEL)
    ll.load(MODEL, n_gpu_layers=99, n_ctx=2048, n_batch=2048, n_seq_max=1)
    ll._adapters = []
    for p in lora_paths:
        ll.load_lora(p)
    jobs = build_jobs(ll, tpl, DATASETS, META)
    flat = flatten_states(jobs)
    hmeta = json.load(open(HARVEST_PREFIX + ".meta.json", encoding="utf-8"))
    hstates = hmeta["states"]
    assert len(flat) == len(hstates), (len(flat), len(hstates))
    for (ji, st), hs in zip(flat, hstates):
        assert jobs[ji]["id"] == hs["id"] and st["pos"] == hs["pos"] and len(jobs[ji]["tokens"]) == hs["n_tokens"], (jobs[ji]["id"], hs["id"])
    S, d = len(flat), ll.n_embd
    job_states: dict[int, list[int]] = {}
    for si, (ji, st) in enumerate(flat):
        job_states.setdefault(ji, []).append(si)

    P3_CONDS = [("base", {}), ("corpus", {0: 1.0}), ("weft-eido", {1: 1.0}), ("formal-verification", {2: 1.0}), ("triple", {0: 1.0, 1: 1.0, 2: 1.0})]
    H = np.zeros((len(P3_CONDS), S, d), np.float32)
    contrast = np.full((len(P3_CONDS), S), np.nan)
    top1 = np.zeros((len(P3_CONDS), S), np.int64)
    top1_logit = np.zeros((len(P3_CONDS), S))
    for ci, (cname, scales) in enumerate(P3_CONDS):
        ll.set_loras(scales)
        for ji, job in enumerate(jobs):
            sis = job_states[ji]
            pos_list = sorted(flat[si][1]["pos"] for si in sis)
            res = ll.decode(job["tokens"], {"l_out-31": list(range(len(pos_list))), "__logits__": True}, pos_list)
            for k, si in enumerate(sis):
                st = flat[si][1]
                H[ci, si] = res["l_out-31"][k]
                lg = res["__logits__"][k]
                top1[ci, si] = int(np.argmax(lg)); top1_logit[ci, si] = float(lg[top1[ci, si]])
                if st["pos_toks"] and st["neg_toks"]:
                    contrast[ci, si] = float(np.mean(lg[st["pos_toks"]]) - np.mean(lg[st["neg_toks"]]))
            thermal.tick()
        print(f"P3 cond {cname} done at {time.time()-t_all:.0f}s", flush=True)
    ll.set_loras({})
    # determinism of the base pass: repeat base on the first 20 sequences
    rep_diff = []
    for ji, job in list(enumerate(jobs))[:20]:
        sis = job_states[ji]
        pos_list = sorted(flat[si][1]["pos"] for si in sis)
        res = ll.decode(job["tokens"], {"l_out-31": list(range(len(pos_list)))}, pos_list)
        rep_diff.append(float(np.max(np.abs(res["l_out-31"] - H[0, sis]))))
    out["p3_base_repeat_max_abs_diff_l_out31"] = max(rep_diff)

    # ---- P4: control-vector Jacobian probes -----------------------------------------------------------
    W = Weights(MODEL)
    g_out = W.get("output_norm.weight")
    Wd = {l: W.get(f"blk.{l}.ffn_down.weight") for l in LAYERS}
    Z = np.load(HARVEST_PREFIX + ".npz")
    groups = [s["group"] for s in hstates]
    valid = [i for i, s in enumerate(hstates) if s["pos_toks"] and s["neg_toks"] and sorted(s["pos_toks"]) != sorted(s["neg_toks"])]
    fv = [i for i in valid if groups[i] == "fv"]
    sel = [i for i in valid if groups[i] in ("corpus", "eido")] + fv[::8]
    tok_ids = sorted({t for i in sel for t in hstates[i]["pos_toks"] + hstates[i]["neg_toks"]})
    E = W.get_rows("token_embd.weight", tok_ids)
    trow = {t: k for k, t in enumerate(tok_ids)}
    rng = np.random.default_rng(20261119)
    probes = []
    DH = np.zeros((len(sel), 4, d), np.float32)       # odd response to the plain direction at eps_large
    DIRS = np.zeros((len(sel), 4, d), np.float32)
    for k, si in enumerate(sel):
        ji, st = flat[si]
        toks, pos = jobs[ji]["tokens"], st["pos"]
        h0 = H[0, si].astype(np.float64)
        R0 = float(np.sqrt(h0 @ h0 + d * EPS))
        hs = hstates[si]
        c = E[[trow[t] for t in hs["pos_toks"]]].mean(axis=0) - E[[trow[t] for t in hs["neg_toks"]]].mean(axis=0)
        u = g_out * c
        g = (np.sqrt(d) / R0) * (u - (u @ h0) / (R0 * R0) * h0)
        ghat = g / np.linalg.norm(g)
        hhat = h0 / np.linalg.norm(h0)
        rdir = rng.standard_normal(d); rdir -= (rdir @ hhat) * hhat; rdir /= np.linalg.norm(rdir)
        for li, l in enumerate(LAYERS):
            s = silu(Z["s_gate_pre"][si, li].astype(np.float64))
            v = Wd[l] @ ((s * s) * (Wd[l].T @ ghat))
            y = v - (v @ hhat) * hhat
            y /= np.linalg.norm(y)
            DIRS[k, li] = y

            def respond(direction, eps):
                hp = []
                for sign in (+1.0, -1.0):
                    ll.set_cvec({l: sign * eps * direction})
                    r = ll.decode(toks, {"l_out-31": [0]}, [pos])
                    hp.append(r["l_out-31"][0].astype(np.float64))
                    thermal.tick()
                odd = 0.5 * (hp[0] - hp[1])
                even = 0.5 * (hp[0] + hp[1]) - h0
                return odd, even

            rec = {"state": si, "group": groups[si], "layer": l, "R0": R0, "n_tokens": len(toks), "pos": pos,
                   "pred_gain_per_unit": float(ghat @ y)}
            for tag, direction, eps_list in (("plain", y, (0.03 * R0, 0.10 * R0)), ("random", rdir, (0.10 * R0,))):
                for eps in eps_list:
                    odd, even = respond(direction, eps)
                    odd_t = odd - (odd @ hhat) * hhat
                    key = f"{tag}_eps{eps / R0:.2f}"
                    rec[key] = {"amp": float(np.linalg.norm(odd) / eps), "cos_with_injected": float((odd @ direction) / (np.linalg.norm(odd) + 1e-300)),
                                "tangent_energy_ratio": float((odd_t @ odd_t) / (eps * eps)),
                                "gain_real_over_pred": float((ghat @ odd) / (eps * (ghat @ direction))) if abs(ghat @ direction) > 1e-9 else None,
                                "even_over_odd_norm": float(np.linalg.norm(even) / (np.linalg.norm(odd) + 1e-300)),
                                "radial_frac_of_odd": float(abs(odd @ hhat) / (np.linalg.norm(odd) + 1e-300))}
                    if tag == "plain" and abs(eps - 0.10 * R0) < 1e-12:
                        DH[k, li] = odd / eps
            probes.append(rec)
        if (k + 1) % 10 == 0:
            print(f"P4 state {k+1}/{len(sel)} at {time.time()-t_all:.0f}s (max temp {thermal.max_seen}C)", flush=True)
    ll.set_cvec(None)
    ll.close()

    np.savez(OUT_PREFIX + ".npz", H=H, contrast=contrast, top1=top1, top1_logit=top1_logit, sel=np.array(sel), DH=DH, DIRS=DIRS)
    out["p3_conditions"] = [c for c, _ in P3_CONDS]
    out["p4_probes"] = probes
    out["p4_selected_states"] = sel
    out["thermal"] = {"max_temp_c_sampled": thermal.max_seen, "pauses": thermal.pauses}
    out["elapsed_s"] = round(time.time() - t_all, 1)
    json.dump(out, open(OUT_PREFIX + ".json", "w", encoding="utf-8"), indent=1)
    print("saved", OUT_PREFIX + ".json", "elapsed", out["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
