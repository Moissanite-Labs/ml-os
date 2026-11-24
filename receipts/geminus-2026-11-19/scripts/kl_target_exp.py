"""Experiment K: targets chosen in distribution space (thread T2), exact float64 at blk.31, full vocabulary.

Preconditioning by the gate metric minimises the NORM of the final-residual change per unit contrast gain.
It did not reduce on-domain KL. Here the final-residual change dh is chosen to control what happens to
the rest of the next-token distribution.

With n(h) the final RMSNorm, logits z = E n(h), and J the Jacobian of n at h, the first-order logit change
of token j is  dz_j = t_j . dh  with  t_j = J^T E_j  (the tangent-projected, gain-weighted embedding).
The contrast gradient is g = mean_pos t - mean_neg t.

Arms (each scaled to the same first-order contrast gain):
  plain          dh = M g                      (team's current target, M = W_down diag(s^2) W_down^T)
  precond        dh = M (M + lam I)^-1 g       (lam = mean eigenvalue of M)
  minnorm        dh = g
  null_k         dh = P g, P projecting out span{ t_j - t_j0 : j in top-k other tokens }
                 -> zero first-order change in the relative logits of the k most likely other tokens
  fisher_mu      dh = (F + mu I)^-1 g,  F = T^T (diag(p) - p p^T) T over the top-256 tokens
                 -> minimum first-order KL per unit gain, ridge mu in units of trace(F)/d
Reported exactly (no linearisation): contrast gain, eta, KL over all 248320 tokens, change in log P of the
positive tokens, |dh|. Also the KL floor if ONLY the positive and negative logits moved.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from gguf.quants import dequantize

SCRIPTS, MODEL, HARVEST_PREFIX, OUT_JSON = sys.argv[1:5]
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 3
sys.path.insert(0, SCRIPTS)
from ggufw import Weights  # noqa: E402

L, LI, EPS = 31, 3, 1e-6
GAINS = [1.0, 3.0]
NULL_K = [16, 64, 256]
FISHER_MU = [1.0, 0.1, 0.01]
KF = 256


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
    d = meta["d_model"]
    W = Weights(MODEL)
    g_out = W.get("output_norm.weight")
    Wd = W.get(f"blk.{L}.ffn_down.weight")
    te = W.by_name["token_embd.weight"]
    E32 = np.ascontiguousarray(np.asarray(dequantize(te.data, te.tensor_type), dtype=np.float32).reshape(int(te.shape[1]), int(te.shape[0])))
    print(f"embedding loaded {E32.shape} in {time.time()-t0:.0f}s", flush=True)
    sqd = np.sqrt(d)

    valid = [i for i, s in enumerate(st) if s["pos_toks"] and s["neg_toks"] and sorted(s["pos_toks"]) != sorted(s["neg_toks"])]
    fv = [i for i in valid if st[i]["group"] == "fv"]
    sel = [i for i in valid if st[i]["group"] != "fv"] + fv[::STRIDE]

    def normed(hv):
        return g_out * hv * (sqd / np.sqrt(hv @ hv + d * EPS))

    def logits(hv):
        return E32 @ normed(hv).astype(np.float32)

    def cg(matvec, b, tol=1e-6, maxit=150):
        x = np.zeros_like(b); r = b.copy(); p = r.copy(); rs = r @ r
        b2 = b @ b
        for _ in range(maxit):
            Ap = matvec(p)
            a = rs / (p @ Ap)
            x += a * p; r -= a * Ap
            rs2 = r @ r
            if rs2 <= tol * tol * b2:
                break
            p = r + (rs2 / rs) * p; rs = rs2
        return x

    arms = ["plain", "precond", "minnorm"] + [f"null_{k}" for k in NULL_K] + [f"fisher_{m:g}" for m in FISHER_MU]
    per_state = []
    for n_done, i in enumerate(sel):
        h = Z["s_l_out"][i, LI].astype(np.float64)
        s = silu(Z["s_gate_pre"][i, LI].astype(np.float64))
        R0sq = h @ h + d * EPS
        R0 = np.sqrt(R0sq)
        pos, neg = st[i]["pos_toks"], st[i]["neg_toks"]
        z0 = logits(h)
        lp0 = log_softmax(z0)
        p0 = np.exp(lp0)

        def tang(rows):                       # t_j = J^T E_j for a block of embedding rows (n, d)
            u = rows.astype(np.float64) * g_out[None, :]
            return (sqd / R0) * (u - np.outer(u @ h, h) / R0sq)
        g = tang(E32[pos]).mean(axis=0) - tang(E32[neg]).mean(axis=0)
        s2 = s * s
        Mv = lambda v: Wd @ (s2 * (Wd.T @ v))
        mubar = float(np.sum(s2 * np.sum(Wd * Wd, axis=0)) / d)
        order = np.argsort(-z0)
        others = [int(t) for t in order[:KF + len(pos) + len(neg)] if int(t) not in pos and int(t) not in neg][:KF]
        T = tang(E32[others])                 # (KF, d)
        pT = p0[others]
        dirs = {"plain": Mv(g), "minnorm": g.copy()}
        zc = cg(lambda v: Mv(v) + mubar * v, g)
        dirs["precond"] = Mv(zc)
        for k in NULL_K:
            C = T[1:k] - T[0:1]               # relative logits of the k most likely other tokens stay put
            Q, _ = np.linalg.qr(C.T)
            dirs[f"null_{k}"] = g - Q @ (Q.T @ g)
        Tc = T - (pT / max(pT.sum(), 1e-300)) @ T      # centre under p restricted to the support
        F = (Tc * pT[:, None]).T @ Tc
        trF = float(np.trace(F)) / d
        for m in FISHER_MU:
            dirs[f"fisher_{m:g}"] = np.linalg.solve(F + (m * trF + 1e-12) * np.eye(d), g)

        rec = {"state": i, "group": st[i]["group"], "R0": float(R0), "p_pos_base": float(np.sum(p0[pos])), "p_neg_base": float(np.sum(p0[neg])),
               "mass_in_top256_others": float(pT.sum()), "arms": {}}
        for Gt in GAINS:
            # KL floor if only the positive / negative logits moved (best split of the contrast between them)
            best = None
            for frac in np.linspace(0.0, 1.0, 21):
                z1 = z0.astype(np.float64).copy()
                z1[pos] += frac * Gt; z1[neg] -= (1.0 - frac) * Gt
                kl = float(np.sum(p0 * (lp0 - log_softmax(z1))))
                best = kl if best is None or kl < best else best
            rec[f"kl_floor_only_pos_neg_G{Gt:g}"] = best
        for name in arms:
            dv = dirs[name]
            g1 = float(g @ dv)
            rs = {"cos_with_g": float(g1 / (np.linalg.norm(g) * np.linalg.norm(dv) + 1e-300))}
            for Gt in GAINS:
                if g1 <= 1e-12:
                    rs[f"G{Gt:g}"] = None
                    continue
                dh = dv * (Gt / g1)
                h1 = h + dh
                par = (dh @ h) / (h @ h)
                dperp = dh - par * h
                z1 = logits(h1)
                lp1 = log_softmax(z1)
                rs[f"G{Gt:g}"] = {"eta": float(dperp @ dperp / R0sq), "dh_norm": float(np.linalg.norm(dh)),
                                  "gain_exact": float((np.mean(z1[pos]) - np.mean(z1[neg])) - (np.mean(z0[pos]) - np.mean(z0[neg]))),
                                  "kl": float(np.sum(p0 * (lp0 - lp1))), "dlogp_pos": float(np.mean(lp1[pos] - lp0[pos])),
                                  "top1_changed": bool(int(np.argmax(z1)) != int(np.argmax(z0)))}
            rec["arms"][name] = rs
        per_state.append(rec)
        if (n_done + 1) % 20 == 0:
            print(f"  {n_done+1}/{len(sel)} states {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for Gt in GAINS:
        key = f"G{Gt:g}"
        tab = {}
        for name in arms:
            rows = [r["arms"][name][key] for r in per_state if r["arms"][name][key] is not None]
            col = lambda f: np.array([x[f] for x in rows], dtype=np.float64)
            tab[name] = {"n": len(rows), "eta_median": float(np.median(col("eta"))), "eta_mean": float(col("eta").mean()),
                         "kl_median": float(np.median(col("kl"))), "kl_mean": float(col("kl").mean()),
                         "gain_exact_median": float(np.median(col("gain_exact"))), "dlogp_pos_median": float(np.median(col("dlogp_pos"))),
                         "dlogp_pos_mean": float(col("dlogp_pos").mean()), "dh_norm_median": float(np.median(col("dh_norm"))),
                         "top1_changed_frac": float(col("top1_changed").mean()),
                         "cos_with_g_median": float(np.median([r["arms"][name]["cos_with_g"] for r in per_state]))}
        fl = np.array([r[f"kl_floor_only_pos_neg_{key}"] for r in per_state])
        summary[key] = {"arms": tab, "kl_floor_only_pos_neg_median": float(np.median(fl)), "kl_floor_only_pos_neg_mean": float(fl.mean())}
    out = {"schema": "geminus-kl-target-exp/v1", "layer": L, "n_states": len(sel), "support_k_fisher": KF,
           "p_pos_base_median": float(np.median([r["p_pos_base"] for r in per_state])),
           "mass_in_top256_others_median": float(np.median([r["mass_in_top256_others"] for r in per_state])),
           "summary": summary, "per_state": per_state, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"))
    print(f"p_pos_base median {out['p_pos_base_median']:.4f}; mass in top-256 others median {out['mass_in_top256_others_median']:.4f}")
    for key, blk in summary.items():
        print(f"\n== equal first-order contrast gain {key}: medians over {len(sel)} states; KL floor if only pos/neg logits moved: median {blk['kl_floor_only_pos_neg_median']:.5f} mean {blk['kl_floor_only_pos_neg_mean']:.5f} ==")
        print(f"{'arm':14s} {'eta':>10s} {'KL med':>10s} {'KL mean':>10s} {'gain':>7s} {'dlogp+ med':>10s} {'dlogp+ mean':>11s} {'|dh|':>7s} {'cos(g)':>7s} {'top1chg':>7s}")
        for name, v in blk["arms"].items():
            print(f"{name:14s} {v['eta_median']:10.6f} {v['kl_median']:10.5f} {v['kl_mean']:10.5f} {v['gain_exact_median']:7.3f} {v['dlogp_pos_median']:10.4f} {v['dlogp_pos_mean']:11.4f} "
                  f"{v['dh_norm_median']:7.3f} {v['cos_with_g_median']:7.3f} {v['top1_changed_frac']:7.3f}")


if __name__ == "__main__":
    main()
