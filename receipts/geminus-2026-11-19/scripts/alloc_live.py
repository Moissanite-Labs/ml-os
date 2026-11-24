"""Experiment L (live, RTX 3080): cross-layer allocation and target-residual arms, measured directly.

Four single-layer rank-8 adapters (blk.28..31, plain targets, same recipe) are loaded together;
the per-adapter scale vector IS the allocation. Pass 1 measures each adapter's realised mean
contrast gain on held-out formal-verification states at scale 1. Pass 2 sets scales
t_l = alpha_l * G / gain_l for several allocations alpha at a common target gain G and measures
the realised gain, the tangent energy of the final-residual change (eta), KL over the full
vocabulary, and off-domain movement (corpus / eido prompts, y=0 anchors).
The blk.31 preconditioned-target arms from Experiment C are measured the same way.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np

SCRIPTS, BIN, MODEL, DATASETS, META, ADAPTER_DIR, OUT_JSON = sys.argv[1:8]
sys.path.insert(0, SCRIPTS)
from jobs import build_jobs  # noqa: E402
from llama_ctypes import Llama  # noqa: E402
from render import ChatTemplate  # noqa: E402

EPS = 1e-6
ALLOC = {
    "only28": [1, 0, 0, 0], "only29": [0, 1, 0, 0], "only30": [0, 0, 1, 0], "only31": [0, 0, 0, 1],
    "uniform": [0.25, 0.25, 0.25, 0.25],
    "passthrough_diag_law": [0.245, 0.184, 0.370, 0.202],
    "passthrough_gram_opt": [0.270, 0.130, 0.438, 0.162],
    "mac_product_law": [0.125, 0.107, 0.225, 0.543],
    "late_heavy": [0.05, 0.05, 0.10, 0.80],
}


def gpu_temp():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        return int(o)
    except Exception:
        return None


def log_softmax(v):
    v = v.astype(np.float64)
    m = v.max()
    return v - m - np.log(np.sum(np.exp(v - m)))


def main():
    t0 = time.time()
    man = json.load(open(ADAPTER_DIR + "/manifest.json", encoding="utf-8"))
    held_ids = set(man["held_out_ids"])
    tpl = ChatTemplate(MODEL)
    ll = Llama(BIN)
    ll.check_layout()
    ll.load(MODEL, n_gpu_layers=99, n_ctx=2048, n_batch=2048)
    names = sorted(man["adapters"].keys())
    idx = {n: ll.load_lora(f"{ADAPTER_DIR}/{man['adapters'][n]['file']}") for n in names}
    jobs = [j for j in build_jobs(ll, tpl, DATASETS, META) if j["group"] != "fv" or j["id"] in held_ids]
    states = []
    for ji, job in enumerate(jobs):
        for k, s in enumerate(sorted(job["states"], key=lambda z: z["pos"])):
            states.append((ji, k, s, job["group"]))
    n_s, d = len(states), ll.n_embd
    grp = np.array([g for _, _, _, g in states])
    has_c = np.array([bool(s["pos_toks"]) and bool(s["neg_toks"]) and sorted(s["pos_toks"]) != sorted(s["neg_toks"]) for _, _, s, _ in states])
    print(f"eval: {len(jobs)} sequences, {n_s} states ({dict((g, int((grp == g).sum())) for g in sorted(set(grp)))}); adapters {names}", flush=True)
    pauses, n_dec = [], [0]

    def run(scales):
        ll.set_loras({idx[n]: float(v) for n, v in scales.items() if v != 0.0})
        H = np.zeros((n_s, d), np.float32)
        LG = []
        si = 0
        for job in jobs:
            pos = sorted(s["pos"] for s in job["states"])
            r = ll.decode(job["tokens"], {"l_out-31": list(range(len(pos))), "__logits__": True}, pos)
            for k in range(len(pos)):
                H[si] = r["l_out-31"][k]
                LG.append(r["__logits__"][k])
                si += 1
            n_dec[0] += 1
            if n_dec[0] % 40 == 0:
                t = gpu_temp()
                if t is not None and t >= 85:
                    tp = time.time()
                    while time.time() - tp < 240:
                        time.sleep(3)
                        t2 = gpu_temp()
                        if t2 is not None and t2 <= 75:
                            break
                    pauses.append({"temp_c": t, "paused_s": round(time.time() - tp, 1)})
        return H, LG

    Hb, LGb = run({})
    base_lp = [log_softmax(l) for l in LGb]
    hb = Hb.astype(np.float64)
    R0sq = np.sum(hb * hb, axis=1) + d * EPS
    hhat = hb / np.linalg.norm(hb, axis=1, keepdims=True)

    def contrast(lg, s):
        return float(np.mean(lg[s["pos_toks"]]) - np.mean(lg[s["neg_toks"]]))
    c_base = np.array([contrast(LGb[i], states[i][2]) if has_c[i] else np.nan for i in range(n_s)])

    def measure(scales):
        H, LG = run(scales)
        dh = H.astype(np.float64) - hb
        par = np.sum(dh * hhat, axis=1)
        dperp = dh - par[:, None] * hhat
        eta = np.sum(dperp ** 2, axis=1) / R0sq
        gain = np.array([contrast(LG[i], states[i][2]) - c_base[i] if has_c[i] else np.nan for i in range(n_s)])
        kl = np.zeros(n_s); top1 = np.zeros(n_s, bool); dlp = np.full(n_s, np.nan)
        for i in range(n_s):
            lp = log_softmax(LG[i])
            kl[i] = float(np.sum(np.exp(base_lp[i]) * (base_lp[i] - lp)))
            top1[i] = int(np.argmax(lp)) != int(np.argmax(base_lp[i]))
            if has_c[i]:
                pt = states[i][2]["pos_toks"]
                dlp[i] = float(np.mean(lp[pt] - base_lp[i][pt]))
        res = {}
        fvm = (grp == "fv") & has_c
        g_, e_ = gain[fvm], eta[fvm]
        okg = g_ > 0.05
        res["held_out_fv"] = {
            "n": int(fvm.sum()), "gain_mean": float(g_.mean()), "gain_median": float(np.median(g_)), "gain_frac_positive": float(np.mean(g_ > 0)),
            "dlogp_pos_mean": float(np.nanmean(dlp[fvm])), "eta_mean": float(e_.mean()), "eta_median": float(np.median(e_)),
            "kl_mean": float(kl[fvm].mean()), "kl_median": float(np.median(kl[fvm])),
            "dh_over_R0_mean": float((np.linalg.norm(dh[fvm], axis=1) / np.sqrt(R0sq[fvm])).mean()),
            "cost_mean_eta_over_mean_gain_sq": float(e_.mean() / (g_.mean() ** 2)) if abs(g_.mean()) > 1e-9 else None,
            "cost_median_per_state_eta_over_gain_sq": float(np.median(e_[okg] / g_[okg] ** 2)) if okg.sum() > 5 else None,
            "kl_per_unit_gain": float(kl[fvm].mean() / g_.mean()) if abs(g_.mean()) > 1e-9 else None,
            "top1_changed_frac": float(top1[fvm].mean()),
        }
        for g in ("corpus", "eido", "anchor"):
            m = grp == g
            res[g] = {"n": int(m.sum()), "dh_over_R0_mean": float((np.linalg.norm(dh[m], axis=1) / np.sqrt(R0sq[m])).mean()),
                      "eta_mean": float(eta[m].mean()), "kl_mean": float(kl[m].mean()), "top1_changed_frac": float(top1[m].mean())}
        return res

    out = {"schema": "geminus-alloc-live/v1", "adapters": man["adapters"], "n_states": n_s, "pass1_scale1": {}, "pass2": {}}
    for n in names:
        out["pass1_scale1"][n] = measure({n: 1.0})
        r = out["pass1_scale1"][n]["held_out_fv"]
        print(f"pass1 {n:14s} gain {r['gain_mean']:+.4f} (median {r['gain_median']:+.4f}) eta {r['eta_mean']:.3e} KL {r['kl_mean']:.4e} cost {r['cost_mean_eta_over_mean_gain_sq']}  t={time.time()-t0:.0f}s", flush=True)

    lay = ["plain_blk28", "plain_blk29", "plain_blk30", "plain_blk31"]
    g1 = {n: out["pass1_scale1"][n]["held_out_fv"]["gain_mean"] for n in names}
    G = 0.5 * g1["plain_blk31"]
    out["target_gain"] = G
    out["pass1_gain_per_unit_scale"] = g1
    for aname, alpha in ALLOC.items():
        sc = {lay[k]: (alpha[k] * G / g1[lay[k]] if alpha[k] and g1[lay[k]] > 1e-6 else 0.0) for k in range(4)}
        m = measure(sc)
        m["scales"] = sc; m["alpha"] = alpha
        out["pass2"][aname] = m
        r = m["held_out_fv"]
        print(f"pass2 {aname:22s} gain {r['gain_mean']:+.4f} eta {r['eta_mean']:.3e} KL {r['kl_mean']:.4e} cost {r['cost_mean_eta_over_mean_gain_sq']:.5f} "
              f"| corpus dh/R0 {m['corpus']['dh_over_R0_mean']:.4f} anchors {m['anchor']['dh_over_R0_mean']:.5f}  t={time.time()-t0:.0f}s", flush=True)
    for n in names:
        if not n.startswith("c31_") or g1[n] <= 1e-6:
            continue
        m = measure({n: G / g1[n]})
        m["scales"] = {n: G / g1[n]}
        out["pass2"][n] = m
        r = m["held_out_fv"]
        print(f"pass2 {n:22s} gain {r['gain_mean']:+.4f} eta {r['eta_mean']:.3e} KL {r['kl_mean']:.4e} cost {r['cost_mean_eta_over_mean_gain_sq']:.5f} dlogp+ {r['dlogp_pos_mean']:+.4f} "
              f"| corpus dh/R0 {m['corpus']['dh_over_R0_mean']:.4f} anchors {m['anchor']['dh_over_R0_mean']:.5f}  t={time.time()-t0:.0f}s", flush=True)
    m = measure({"plain_blk31": 2.0 * G / g1["plain_blk31"]})
    out["pass2"]["only31_double_gain"] = m
    ll.set_loras({})
    Hc, _ = run({})
    out["base_repeat_max_abs_diff"] = float(np.max(np.abs(Hc - Hb)))
    ll.close()
    out["thermal_pauses"] = pauses
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print("saved", OUT_JSON, out["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
