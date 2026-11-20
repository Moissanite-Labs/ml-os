"""Experiment D: spend the edit where the base model is actually wrong (thread T4), blk.31, exact float64.

The team's targets contrast the positive token against a fixed list of negatives at every decision
state. On these states the base already ranks the positive token first at about half of them.
Arms (plain pulled-back targets, P2-B recipe, rank 8; only the choice of states and negatives differs):
    team            all states, listed negatives                               (the current recipe)
    contested       only states where the base top-1 is not a positive token, listed negatives
    vs_top1         contested states, negative = the base model's actual top-1 token
    vs_top3         contested states, negatives = the base model's top-3 non-positive tokens
Every arm is rescaled to the same mean radial shrinkage eta on the held-out states (eta is exactly
quadratic in scale at the last layer), then compared on what the edit is for: does the positive token
become the top prediction, and how much log-probability does it gain; and on collateral change.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from gguf.quants import dequantize

SCRIPTS, WEFT_SRC, MODEL, HARVEST_PREFIX, OUT_JSON = sys.argv[1:6]
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, WEFT_SRC)
from ggufw import Weights  # noqa: E402
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector  # noqa: E402

L, LI, EPS = 31, 3, 1e-6
REL_SCALE, OUT_GAIN, STEP0_W = 0.35, 1.65, 0.20
ARMS = ["team", "contested", "vs_top1", "vs_top3"]
ETA_TARGETS = [0.002, 0.007, 0.02]


def silu(z):
    return z / (1.0 + np.exp(-z))


def log_softmax(v):
    v = v.astype(np.float64)
    m = v.max()
    return v - m - np.log(np.sum(np.exp(v - m)))


def main():
    t0 = time.time()
    meta = json.load(open(HARVEST_PREFIX + ".meta.json", encoding="utf-8"))
    Z = np.load(HARVEST_PREFIX + ".npz")
    st = meta["states"]
    d, dff = meta["d_model"], meta["d_ff"]
    W = Weights(MODEL)
    g_out = W.get("output_norm.weight")
    Wd = W.get(f"blk.{L}.ffn_down.weight")
    te = W.by_name["token_embd.weight"]
    E32 = np.ascontiguousarray(np.asarray(dequantize(te.data, te.tensor_type), dtype=np.float32).reshape(int(te.shape[1]), int(te.shape[0])))
    sqd = np.sqrt(d)
    fv = [i for i, s in enumerate(st) if s["group"] == "fv"]
    ids = sorted({st[i]["id"] for i in fv})
    held_ids = set(ids[2::3])
    train = [i for i in fv if st[i]["id"] not in held_ids]
    held = [i for i in fv if st[i]["id"] in held_ids]
    anchors = [i for i, s in enumerate(st) if s["group"] == "anchor"]
    offdom = [i for i, s in enumerate(st) if s["group"] in ("corpus", "eido")]

    def normed(hv):
        return g_out * hv * (sqd / np.sqrt(hv @ hv + d * EPS))

    def base(i):
        h = Z["s_l_out"][i, LI].astype(np.float64)
        s = silu(Z["s_gate_pre"][i, LI].astype(np.float64))
        x = Z["s_key"][i, LI].astype(np.float64)
        return h, s, x

    def grad(h, pos, neg):
        R0sq = h @ h + d * EPS
        c = E32[pos].astype(np.float64).mean(axis=0) - E32[neg].astype(np.float64).mean(axis=0)
        u = g_out * c
        return (sqd / np.sqrt(R0sq)) * (u - (u @ h) / R0sq * h)

    info = {}
    for i in train + held + anchors + offdom:
        h, s, x = base(i)
        z0 = E32 @ normed(h).astype(np.float32)
        order = np.argsort(-z0)[:8]
        pos = st[i]["pos_toks"]
        info[i] = {"h": h, "s": s, "x": x, "z0": z0, "lp0": log_softmax(z0), "top": [int(t) for t in order],
                   "contested": bool(pos) and int(order[0]) not in pos}
    n_cont_train = sum(info[i]["contested"] for i in train)
    n_cont_held = sum(info[i]["contested"] for i in held)
    print(f"train {len(train)} (contested {n_cont_train}) | held-out {len(held)} (contested {n_cont_held}) | {time.time()-t0:.0f}s", flush=True)

    def targets(arm):
        cols_k, cols_u = [], []
        for i in train:
            v = info[i]
            pos, neg = st[i]["pos_toks"], st[i]["neg_toks"]
            if arm != "team" and not v["contested"]:
                continue
            if arm == "vs_top1":
                neg = [v["top"][0]]
            elif arm == "vs_top3":
                neg = [t for t in v["top"] if t not in pos][:3]
            g = grad(v["h"], pos, neg)
            du = v["s"] * (Wd.T @ g)
            induced = Wd @ (v["s"] * du)
            fo = float(np.linalg.norm(Z["s_ffn_out"][i, LI].astype(np.float64)))
            w_step = STEP0_W if st[i]["rel_step"] == 0 else 1.0
            cols_k.append(v["x"])
            cols_u.append(du * (max(0.05, fo) * REL_SCALE / np.linalg.norm(induced)) * OUT_GAIN * w_step)
        return np.stack(cols_k, axis=1), np.stack(cols_u, axis=1)

    K0 = Z["n_key"][:, LI, :].astype(np.float64).T
    K0 = K0 / np.sqrt(K0.shape[1])
    anc_keys = np.stack([info[i]["x"] for i in anchors], axis=1)
    cfg = BetaEditConfig(d_in=d, d_out=dff, max_rank=8, max_slots=8, lambda_leakage=250.0, lambda_ridge=4.0,
                         nullspace_threshold=1e9, threshold_mode="absolute", radix_scale_exp2=24, quant_dtype="f16")
    proj = BetaEditNullspaceProjector(np.zeros((dff, d)), cfg)
    proj.initialize_anchor_covariance(K0)

    def evaluate(Bm, Am, idx, scale):
        rows = []
        for i in idx:
            v = info[i]
            h = v["h"]
            dh = scale * (Wd @ (v["s"] * (Bm @ (Am @ v["x"]))))
            par = (dh @ h) / (h @ h)
            dperp = dh - par * h
            z1 = E32 @ normed(h + dh).astype(np.float32)
            lp1 = log_softmax(z1)
            pos = st[i]["pos_toks"]
            r = {"eta": float(dperp @ dperp / (h @ h + d * EPS)), "kl": float(np.sum(np.exp(v["lp0"]) * (v["lp0"] - lp1))),
                 "top1_changed": bool(int(np.argmax(z1)) != v["top"][0]), "contested": v["contested"]}
            if pos:
                r["pos_top1_before"] = v["top"][0] in pos
                r["pos_top1_after"] = int(np.argmax(z1)) in pos
                r["dlogp_pos"] = float(np.mean(lp1[pos] - v["lp0"][pos]))
            rows.append(r)
        return rows

    results = {}
    for arm in ARMS:
        K, U = targets(arm)
        patch = proj.stage_edit(K, U, is_residual=True, rank=8, preservation_keys=anc_keys, preservation_weight=2.0, slot_name=arm)
        Bm, Am = np.asarray(patch.factor_b, np.float64), np.asarray(patch.factor_a, np.float64)
        eta1 = float(np.mean([r["eta"] for r in evaluate(Bm, Am, held, 1.0)]))
        res = {"n_train_states_used": int(K.shape[1]), "fro_norm_native": float(np.sqrt(np.sum((Bm.T @ Bm) * (Am @ Am.T)))),
               "truncation_rel_error": float(patch.truncation_rel_error), "held_out_eta_at_native_scale": eta1, "by_eta": {}}
        for et in ETA_TARGETS:
            sc = float(np.sqrt(et / eta1))
            hr = evaluate(Bm, Am, held, sc)
            ar = evaluate(Bm, Am, anchors, sc)
            orr = evaluate(Bm, Am, offdom, sc)
            cont = [r for r in hr if r["contested"]]
            okb = [r for r in hr if not r["contested"]]
            res["by_eta"][f"{et:g}"] = {
                "scale": sc, "fro_norm": sc * res["fro_norm_native"],
                "held_out": {"n": len(hr), "eta_mean": float(np.mean([r["eta"] for r in hr])),
                             "pos_top1_before": float(np.mean([r["pos_top1_before"] for r in hr])), "pos_top1_after": float(np.mean([r["pos_top1_after"] for r in hr])),
                             "contested_n": len(cont), "contested_fixed_frac": float(np.mean([r["pos_top1_after"] for r in cont])),
                             "already_right_n": len(okb), "already_right_broken_frac": float(np.mean([not r["pos_top1_after"] for r in okb])),
                             "dlogp_pos_mean": float(np.mean([r["dlogp_pos"] for r in hr])), "dlogp_pos_mean_contested": float(np.mean([r["dlogp_pos"] for r in cont])),
                             "kl_mean": float(np.mean([r["kl"] for r in hr])), "kl_mean_already_right": float(np.mean([r["kl"] for r in okb]))},
                "off_domain": {"kl_mean": float(np.mean([r["kl"] for r in orr])), "top1_changed_frac": float(np.mean([r["top1_changed"] for r in orr])), "eta_mean": float(np.mean([r["eta"] for r in orr]))},
                "anchors": {"kl_mean": float(np.mean([r["kl"] for r in ar])), "top1_changed_frac": float(np.mean([r["top1_changed"] for r in ar]))},
            }
        results[arm] = res
        print(f"compiled {arm}: {K.shape[1]} states, fro {res['fro_norm_native']:.3f}, eta@1 {eta1:.5f}  {time.time()-t0:.0f}s", flush=True)
    out = {"schema": "geminus-contested-exp/v1", "layer": L, "n_train": len(train), "n_train_contested": int(n_cont_train), "n_held_out": len(held),
           "n_held_out_contested": int(n_cont_held), "eta_targets": ETA_TARGETS, "results": results, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    for et in ETA_TARGETS:
        print(f"\n== every arm rescaled to mean held-out eta = {et:g} ==")
        print(f"{'arm':10s} {'scale':>6s} {'|dW|_F':>7s} | pos top-1 {'before':>6s} {'after':>6s} | contested fixed {'':1s} | right broken | {'dlogp+':>7s} {'(cont.)':>8s} | {'KL':>7s} {'KL(right)':>9s} | off-dom KL {'top1':>5s} | anchors KL")
        for arm in ARMS:
            e = results[arm]["by_eta"][f"{et:g}"]
            h = e["held_out"]
            print(f"{arm:10s} {e['scale']:6.3f} {e['fro_norm']:7.3f} | {'':9s} {h['pos_top1_before']:6.3f} {h['pos_top1_after']:6.3f} | {h['contested_fixed_frac']:15.3f} | {h['already_right_broken_frac']:12.3f} | "
                  f"{h['dlogp_pos_mean']:7.3f} {h['dlogp_pos_mean_contested']:8.3f} | {h['kl_mean']:7.4f} {h['kl_mean_already_right']:9.4f} | {e['off_domain']['kl_mean']:10.4f} {e['off_domain']['top1_changed_frac']:5.2f} | {e['anchors']['kl_mean']:.6f}")


if __name__ == "__main__":
    main()
