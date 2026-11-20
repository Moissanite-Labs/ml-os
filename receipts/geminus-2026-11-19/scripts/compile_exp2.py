"""Experiment C2: distribution-space targets through the real compiler (rank 8, P2-B recipe), blk.31.

Same protocol as compile_exp.py (220 train states of 64 prompts, 112 held-out states of 32 prompts,
exact float64 evaluation) but (a) KL is taken over the full vocabulary, (b) off-domain prompts
(corpus, design) are evaluated as well as the anchors, and (c) the arms differ in the desired
final-residual direction v_i for each training state:
    plain        v = M g                      realised by du = s * W_down^T g
    precond      v = g                        realised by du = s * W_down^T (M + lam I)^-1 g
    null64       v = P_64 g                   realised by du = s * W_down^T (M + lam I)^-1 v
    fisher       v = (F + mu I)^-1 g          realised by du = s * W_down^T (M + lam I)^-1 v
lam = mean eigenvalue of M (matrix-free conjugate gradients). P_64 removes the first-order change in
the relative logits of the 64 most likely other tokens; F is the top-256 Fisher metric, mu = 0.1 tr(F)/d.
Per-state normalisation, step weights and output gain follow the team recipe.
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
ARMS = ["plain", "precond", "null64", "fisher"]
KF, MU = 256, 0.1


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
    colsq = np.sum(Wd * Wd, axis=0)

    fv = [i for i, s in enumerate(st) if s["group"] == "fv"]
    ids = sorted({st[i]["id"] for i in fv})
    held_ids = set(ids[2::3])
    train = [i for i in fv if st[i]["id"] not in held_ids]
    held = [i for i in fv if st[i]["id"] in held_ids]
    anchors = [i for i, s in enumerate(st) if s["group"] == "anchor"]
    offdom = [i for i, s in enumerate(st) if s["group"] in ("corpus", "eido")]
    print(f"train {len(train)} held-out {len(held)} anchors {len(anchors)} off-domain {len(offdom)}", flush=True)

    def normed(hv):
        return g_out * hv * (sqd / np.sqrt(hv @ hv + d * EPS))

    def base(i):
        h = Z["s_l_out"][i, LI].astype(np.float64)
        s = silu(Z["s_gate_pre"][i, LI].astype(np.float64))
        x = Z["s_key"][i, LI].astype(np.float64)
        return h, s, x

    def tang(rows, h, R0sq):
        u = rows.astype(np.float64) * g_out[None, :]
        return (sqd / np.sqrt(R0sq)) * (u - np.outer(u @ h, h) / R0sq)

    def cg(matvec, b, tol=1e-7, maxit=300):
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

    K = np.zeros((d, len(train)))
    U = {a: np.zeros((dff, len(train))) for a in ARMS}
    for j, i in enumerate(train):
        h, s, x = base(i)
        K[:, j] = x
        R0sq = h @ h + d * EPS
        pos, neg = st[i]["pos_toks"], st[i]["neg_toks"]
        g = tang(E32[pos], h, R0sq).mean(axis=0) - tang(E32[neg], h, R0sq).mean(axis=0)
        s2 = s * s
        mubar = float(np.sum(s2 * colsq) / d)
        z0 = E32 @ normed(h).astype(np.float32)
        p0 = np.exp(log_softmax(z0))
        order = np.argsort(-z0)
        others = [int(t) for t in order[:KF + len(pos) + len(neg)] if int(t) not in pos and int(t) not in neg][:KF]
        T = tang(E32[others], h, R0sq)
        pT = p0[others]
        Q, _ = np.linalg.qr((T[1:64] - T[0:1]).T)
        v_null = g - Q @ (Q.T @ g)
        Tc = T - (pT / max(pT.sum(), 1e-300)) @ T
        F = (Tc * pT[:, None]).T @ Tc
        v_fis = np.linalg.solve(F + (MU * float(np.trace(F)) / d + 1e-12) * np.eye(d), g)
        solve = lambda v: cg(lambda q: Wd @ (s2 * (Wd.T @ q)) + mubar * q, v)
        zs = {"plain": g, "precond": solve(g), "null64": solve(v_null), "fisher": solve(v_fis)}
        fo = float(np.linalg.norm(Z["s_ffn_out"][i, LI].astype(np.float64)))
        tn = max(0.05, fo) * REL_SCALE
        w_step = STEP0_W if st[i]["rel_step"] == 0 else 1.0
        for a in ARMS:
            du = s * (Wd.T @ zs[a])
            induced = Wd @ (s * du)
            U[a][:, j] = du * (tn / np.linalg.norm(induced)) * OUT_GAIN * w_step
        if (j + 1) % 40 == 0:
            print(f"  targets {j+1}/{len(train)} {time.time()-t0:.0f}s", flush=True)

    K0 = Z["n_key"][:, LI, :].astype(np.float64).T
    K0 = K0 / np.sqrt(K0.shape[1])
    anc_keys = np.stack([Z["s_key"][i, LI].astype(np.float64) for i in anchors], axis=1)
    cfg = BetaEditConfig(d_in=d, d_out=dff, max_rank=8, max_slots=8, lambda_leakage=250.0, lambda_ridge=4.0,
                         nullspace_threshold=1e9, threshold_mode="absolute", radix_scale_exp2=24, quant_dtype="f16")
    proj = BetaEditNullspaceProjector(np.zeros((dff, d)), cfg)
    proj.initialize_anchor_covariance(K0)

    cache = {}
    for i in held + anchors + offdom:
        h, s, x = base(i)
        R0sq = h @ h + d * EPS
        z0 = E32 @ normed(h).astype(np.float32)
        lp0 = log_softmax(z0)
        g = None
        if st[i]["pos_toks"] and st[i]["neg_toks"] and sorted(st[i]["pos_toks"]) != sorted(st[i]["neg_toks"]):
            g = tang(E32[st[i]["pos_toks"]], h, R0sq).mean(axis=0) - tang(E32[st[i]["neg_toks"]], h, R0sq).mean(axis=0)
        cache[i] = (h, s, x, R0sq, z0, lp0, g)

    def evaluate(Bm, Am, idx, scale):
        rows = []
        for i in idx:
            h, s, x, R0sq, z0, lp0, g = cache[i]
            dh = scale * (Wd @ (s * (Bm @ (Am @ x))))
            h1 = h + dh
            par = (dh @ h) / (h @ h)
            dperp = dh - par * h
            z1 = E32 @ normed(h1).astype(np.float32)
            lp1 = log_softmax(z1)
            r = {"eta": float(dperp @ dperp / R0sq), "dh_over_R0": float(np.linalg.norm(dh) / np.sqrt(R0sq)),
                 "kl": float(np.sum(np.exp(lp0) * (lp0 - lp1))), "top1_changed": bool(int(np.argmax(z1)) != int(np.argmax(z0)))}
            if g is not None:
                pos, neg = st[i]["pos_toks"], st[i]["neg_toks"]
                r["gain_first_order"] = float(g @ dh)
                r["gain_exact"] = float((np.mean(z1[pos]) - np.mean(z1[neg])) - (np.mean(z0[pos]) - np.mean(z0[neg])))
                r["dlogp_pos"] = float(np.mean(lp1[pos] - lp0[pos]))
            rows.append(r)
        return rows

    def agg(rows):
        keys = [k for k in rows[0] if k != "top1_changed"]
        o = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
        o["top1_changed_frac"] = float(np.mean([r["top1_changed"] for r in rows]))
        return o

    results, compiled = {}, {}
    for a in ARMS:
        patch = proj.stage_edit(K, U[a], is_residual=True, rank=8, preservation_keys=anc_keys, preservation_weight=2.0, slot_name=a)
        Bm, Am = np.asarray(patch.factor_b, np.float64), np.asarray(patch.factor_a, np.float64)
        compiled[a] = (Bm, Am)
        results[a] = {"fro_norm": float(np.sqrt(np.sum((Bm.T @ Bm) * (Am @ Am.T)))), "truncation_rel_error": float(patch.truncation_rel_error),
                      "k0_noise_leak_fro": float(np.linalg.norm(Bm @ (Am @ K0))), "native": {"held_out": agg(evaluate(Bm, Am, held, 1.0))}}
        print(f"compiled {a}: fro {results[a]['fro_norm']:.3f} native held-out gain {results[a]['native']['held_out']['gain_exact']:.3f} {time.time()-t0:.0f}s", flush=True)
    for target in ("half", "full"):
        tg = results["plain"]["native"]["held_out"]["gain_first_order"] * (0.5 if target == "half" else 1.0)
        for a in ARMS:
            Bm, Am = compiled[a]
            sc = tg / results[a]["native"]["held_out"]["gain_first_order"]
            results[a][f"equal_gain_{target}"] = {"scale": float(sc), "fro_norm": float(sc * results[a]["fro_norm"]), "k0_noise_leak_fro": float(sc * results[a]["k0_noise_leak_fro"]),
                                                  "held_out": agg(evaluate(Bm, Am, held, sc)), "anchors": agg(evaluate(Bm, Am, anchors, sc)), "off_domain": agg(evaluate(Bm, Am, offdom, sc))}
    out = {"schema": "geminus-compile-exp2/v1", "layer": L, "n_train": len(train), "n_held_out": len(held), "n_anchors": len(anchors), "n_off_domain": len(offdom),
           "arms": ARMS, "results": results, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    np.savez(OUT_JSON.replace(".json", ".factors.npz"), **{f"{n}_B": compiled[n][0] for n in compiled}, **{f"{n}_A": compiled[n][1] for n in compiled})
    for target in ("half", "full"):
        print(f"\n== rescaled to equal mean held-out first-order gain ({target} of plain native) ==")
        print(f"{'arm':9s} {'scale':>6s} {'|dW|_F':>7s} {'K0leak':>7s} | held-out {'gain':>6s} {'dlogp+':>7s} {'eta':>9s} {'KL':>8s} {'top1':>5s} | off-domain {'dh/R0':>7s} {'KL':>8s} {'top1':>5s} | anchors {'dh/R0':>8s} {'KL':>9s}")
        for a in ARMS:
            e = results[a][f"equal_gain_{target}"]
            print(f"{a:9s} {e['scale']:6.3f} {e['fro_norm']:7.3f} {e['k0_noise_leak_fro']:7.4f} | {'':9s}{e['held_out']['gain_exact']:6.3f} {e['held_out']['dlogp_pos']:7.4f} {e['held_out']['eta']:9.6f} {e['held_out']['kl']:8.4f} {e['held_out']['top1_changed_frac']:5.2f} | "
                  f"{'':11s}{e['off_domain']['dh_over_R0']:7.4f} {e['off_domain']['kl']:8.4f} {e['off_domain']['top1_changed_frac']:5.2f} | {'':8s}{e['anchors']['dh_over_R0']:8.5f} {e['anchors']['kl']:9.6f}")


if __name__ == "__main__":
    main()
