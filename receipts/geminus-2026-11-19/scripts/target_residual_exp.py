"""Experiment T: target-residual preconditioning at blk.31, evaluated exactly in float64.

For an edit of W_up at the LAST layer, the final pre-norm residual changes by exactly
    dh = W_down (s * du)                    (s = SiLU gate, du = dW_up x; no approximation)
so the true post-edit logits follow from the RMSNorm formula. Family of target residuals:
    du(lambda) = s * W_down^T (M + lambda I)^{-1} g ,   M = W_down diag(s^2) W_down^T
    lambda -> inf : plain pulled-back residual  s * W_down^T g   (mcproof activation_harvester)
    lambda -> 0   : minimum-norm final perturbation, dh parallel to g
Every member is scaled to the same FIRST-ORDER contrast gain, then evaluated exactly:
    achieved contrast gain, eta = ||dh_perp||^2 / R0^2, base-scale factor q, KL over the top-k
    support, change in log-prob of the positive token, and ||du||^2 (weight-norm proxy).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

SCRIPTS, MODEL, HARVEST_PREFIX, OUT_JSON = sys.argv[1:5]
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 3
sys.path.insert(0, SCRIPTS)
from ggufw import Weights  # noqa: E402

L, LI = 31, 3
EPS = 1e-6
LAMS = [float("inf"), 100.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0]   # in units of mean eigenvalue of M
GAINS = [1.0, 3.0]


def silu(z):
    return z / (1.0 + np.exp(-z))


def log_softmax(v):
    m = v.max()
    return v - m - np.log(np.sum(np.exp(v - m)))


def main():
    meta = json.load(open(HARVEST_PREFIX + ".meta.json", encoding="utf-8"))
    Z = np.load(HARVEST_PREFIX + ".npz")
    st = meta["states"]
    d = meta["d_model"]
    W = Weights(MODEL)
    g_out = W.get("output_norm.weight")
    Wd = W.get(f"blk.{L}.ffn_down.weight")
    valid = [i for i, s in enumerate(st) if s["pos_toks"] and s["neg_toks"] and sorted(s["pos_toks"]) != sorted(s["neg_toks"])]
    fv = [i for i in valid if st[i]["group"] == "fv"]
    sel = [i for i in valid if st[i]["group"] != "fv"] + fv[::STRIDE]
    top_ids = Z["s_top_ids"]
    tok_ids = sorted({int(t) for i in sel for t in list(top_ids[i]) + st[i]["pos_toks"] + st[i]["neg_toks"]})
    trow = {t: k for k, t in enumerate(tok_ids)}
    E = W.get_rows("token_embd.weight", tok_ids)
    sqd = np.sqrt(d)

    def normed(hv):
        return g_out * hv * (sqd / np.sqrt(hv @ hv + d * EPS))

    per_state = []
    t0 = time.time()
    for n_done, i in enumerate(sel):
        h = Z["s_l_out"][i, LI].astype(np.float64)
        s = silu(Z["s_gate_pre"][i, LI].astype(np.float64))
        R0sq = h @ h + d * EPS
        R0 = np.sqrt(R0sq)
        pos, neg = st[i]["pos_toks"], st[i]["neg_toks"]
        c = E[[trow[t] for t in pos]].mean(axis=0) - E[[trow[t] for t in neg]].mean(axis=0)
        u = g_out * c
        g = (sqd / R0) * (u - (u @ h) / R0sq * h)
        gnorm = np.linalg.norm(g)
        Ws = Wd * s[None, :]
        M = Ws @ Ws.T
        mu, Vv = np.linalg.eigh(M)
        mu = np.maximum(mu, 0.0)
        mubar = mu.mean()
        gk = Vv.T @ g
        supp = sorted(set(int(t) for t in top_ids[i]) | set(pos) | set(neg))
        Es = E[[trow[t] for t in supp]]
        base_logits = Es @ normed(h)
        base_lp = log_softmax(base_logits)
        pos_idx = [supp.index(t) for t in pos]
        base_contrast = float(c @ normed(h))
        rec = {"state": i, "group": st[i]["group"], "R0": float(R0), "g_norm": float(gnorm),
               "M_top_eig_frac": float(mu[-1] / mu.sum()), "M_participation_ratio": float(mu.sum() ** 2 / np.sum(mu ** 2)),
               "g_energy_in_top8_eigvecs": float(np.sum(gk[-8:] ** 2) / (gnorm ** 2)),
               "kappa_plain": float((gk ** 2 @ mu ** 2) * gnorm ** 2 / (gk ** 2 @ mu) ** 2), "sweep": {}}
        for lam_rel in LAMS:
            if np.isinf(lam_rel):
                w = mu.copy()                      # dh = M g  (plain), du-energy weight = mu
                du_w = mu.copy()
            else:
                lam = lam_rel * mubar
                den = mu + lam
                safe = den > 1e-14 * mubar
                w = np.where(safe, mu / np.where(safe, den, 1.0), 0.0)                 # dh = V diag(w) V^T g
                du_w = np.where(safe, mu / np.where(safe, den, 1.0) ** 2, 0.0)        # ||du||^2 = sum g_k^2 mu/(mu+lam)^2
            dh_unit = Vv @ (w * gk)
            gain1 = float(g @ dh_unit)                    # first-order contrast gain of the unit step
            du_sq_unit = float(gk ** 2 @ du_w)
            rs = {}
            for Gt in GAINS:
                t = Gt / gain1
                dh = t * dh_unit
                h1 = h + dh
                R1 = np.sqrt(h1 @ h1 + d * EPS)
                par = (dh @ h) / (h @ h)
                dperp = dh - par * h
                eta = float(dperp @ dperp / R0sq)
                exact_gain = float(c @ normed(h1)) - base_contrast
                lp = log_softmax(Es @ normed(h1))
                kl = float(np.sum(np.exp(base_lp) * (base_lp - lp)))
                rs[f"G{Gt:g}"] = {"eta": eta, "q_base_scale": float(R0 / R1 * (1.0 + par)), "exact_contrast_gain": exact_gain,
                                  "kl_topk_base_to_edit": kl, "dlogp_pos_mean": float(np.mean(lp[pos_idx] - base_lp[pos_idx])),
                                  "dh_norm": float(np.linalg.norm(dh)), "du_norm_sq": float(t * t * du_sq_unit),
                                  "top1_changed": bool(np.argmax(lp) != np.argmax(base_lp))}
            rec["sweep"]["inf" if np.isinf(lam_rel) else f"{lam_rel:g}"] = rs
        per_state.append(rec)
        if (n_done + 1) % 20 == 0:
            print(f"  {n_done+1}/{len(sel)} states, {time.time()-t0:.0f}s", flush=True)

    summary = {}
    keys = list(per_state[0]["sweep"].keys())
    for Gt in GAINS:
        gk_ = f"G{Gt:g}"
        tab = {}
        for k in keys:
            col = lambda f: np.array([r["sweep"][k][gk_][f] for r in per_state], dtype=np.float64)
            tab[k] = {"eta_median": float(np.median(col("eta"))), "eta_mean": float(col("eta").mean()),
                      "exact_gain_median": float(np.median(col("exact_contrast_gain"))),
                      "kl_median": float(np.median(col("kl_topk_base_to_edit"))), "kl_mean": float(col("kl_topk_base_to_edit").mean()),
                      "dlogp_pos_median": float(np.median(col("dlogp_pos_mean"))),
                      "du_norm_sq_median": float(np.median(col("du_norm_sq"))), "dh_norm_median": float(np.median(col("dh_norm"))),
                      "top1_changed_frac": float(col("top1_changed").mean())}
        summary[gk_] = tab
    out = {"schema": "geminus-target-residual-exp/v1", "layer": L, "n_states": len(sel),
           "lambda_units": "multiples of mean eigenvalue of M = W_down diag(s^2) W_down^T; inf = plain pulled-back residual; 0 = min-norm (dh || g)",
           "M_spectrum": {"top_eig_frac_median": float(np.median([r["M_top_eig_frac"] for r in per_state])),
                          "participation_ratio_median": float(np.median([r["M_participation_ratio"] for r in per_state])),
                          "g_energy_in_top8_eigvecs_median": float(np.median([r["g_energy_in_top8_eigvecs"] for r in per_state])),
                          "kappa_plain_median": float(np.median([r["kappa_plain"] for r in per_state]))},
           "summary": summary, "per_state": per_state}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"))
    print("M spectrum:", json.dumps(out["M_spectrum"]))
    for gk_, tab in summary.items():
        print(f"\n== equal first-order contrast gain {gk_} (medians over {len(sel)} states) ==")
        print(f"{'lambda/mean_eig':>15s} {'eta':>10s} {'exact_gain':>10s} {'KL_topk':>10s} {'dlogp_pos':>10s} {'|dh|':>8s} {'|du|^2':>12s} {'top1_chg':>8s}")
        for k, v in tab.items():
            print(f"{k:>15s} {v['eta_median']:10.5f} {v['exact_gain_median']:10.4f} {v['kl_median']:10.5f} {v['dlogp_pos_median']:10.4f} "
                  f"{v['dh_norm_median']:8.3f} {v['du_norm_sq_median']:12.4f} {v['top1_changed_frac']:8.3f}")


if __name__ == "__main__":
    main()
