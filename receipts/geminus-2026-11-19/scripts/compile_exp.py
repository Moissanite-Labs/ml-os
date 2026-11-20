"""Experiment C: does target-residual preconditioning survive the real compiler?

Uses the team's own BetaEdit projector (weft `BetaEditNullspaceProjector.stage_edit`) with the
mcproof P2-B recipe (rank 8, lambda_leakage 250, lambda_ridge 4, soft full-space projector,
K0 = noise keys / sqrt(n), 24 y=0 anchors as preservation keys with weight 2, F16 factors,
0.20 weight on rel_step-0 states, output gain 1.65, per-state induced post-W_down norm
0.35 * ||ffn_out||) on geminus-harvested blk.31 states.

Only the TARGET RESIDUAL differs between arms:
    du_i(lambda) = s_i * W_down^T (M_i + lambda * mean_eig(M_i) I)^{-1} g_i     (lambda = inf: plain)
Train on 64 formal-verification prompts, evaluate on the 32 held-out prompts and on anchors,
exactly in float64 (a blk.31 W_up edit changes the final residual by W_down (s * (B A x))).
Arms are compared (a) at native scale and (b) rescaled to equal mean held-out contrast gain.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

SCRIPTS, WEFT_SRC, MODEL, HARVEST_PREFIX, OUT_JSON = sys.argv[1:6]
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, WEFT_SRC)
from ggufw import Weights  # noqa: E402
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector  # noqa: E402

L, LI, EPS = 31, 3, 1e-6
ARMS = [("plain", float("inf")), ("lam10", 10.0), ("lam1", 1.0), ("lam0.1", 0.1), ("minnorm", 0.0)]
REL_SCALE, OUT_GAIN, STEP0_W = 0.35, 1.65, 0.20


def silu(z):
    return z / (1.0 + np.exp(-z))


def log_softmax(v):
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
    Wu = W.get(f"blk.{L}.ffn_up.weight")
    sqd = np.sqrt(d)

    fv = [i for i, s in enumerate(st) if s["group"] == "fv"]
    ids = sorted({st[i]["id"] for i in fv})
    held_ids = set(ids[2::3])                              # every third prompt id held out (32 of 96)
    train = [i for i in fv if st[i]["id"] not in held_ids]
    held = [i for i in fv if st[i]["id"] in held_ids]
    anchors = [i for i, s in enumerate(st) if s["group"] == "anchor"]
    print(f"train states {len(train)} ({len(ids)-len(held_ids)} prompts) | held-out {len(held)} ({len(held_ids)} prompts) | anchors {len(anchors)}", flush=True)

    top_ids = Z["s_top_ids"]
    need = sorted({int(t) for i in train + held + anchors for t in list(top_ids[i]) + st[i]["pos_toks"] + st[i]["neg_toks"]})
    trow = {t: k for k, t in enumerate(need)}
    E = W.get_rows("token_embd.weight", need)

    def state(i):
        h = Z["s_l_out"][i, LI].astype(np.float64)
        s = silu(Z["s_gate_pre"][i, LI].astype(np.float64))
        x = Z["s_key"][i, LI].astype(np.float64)
        R0sq = h @ h + d * EPS
        g = c = None
        if st[i]["pos_toks"]:
            c = E[[trow[t] for t in st[i]["pos_toks"]]].mean(axis=0) - E[[trow[t] for t in st[i]["neg_toks"]]].mean(axis=0)
            u = g_out * c
            g = (sqd / np.sqrt(R0sq)) * (u - (u @ h) / R0sq * h)
        return h, s, x, R0sq, g, c

    def normed(hv):
        return g_out * hv * (sqd / np.sqrt(hv @ hv + d * EPS))

    # ---- targets for the training states, all arms ------------------------------------------------
    K = np.zeros((d, len(train)))
    U = {name: np.zeros((dff, len(train))) for name, _ in ARMS}
    tgt_stats = {name: {"gain_per_state": [], "du_norm": []} for name, _ in ARMS}
    for j, i in enumerate(train):
        h, s, x, R0sq, g, c = state(i)
        K[:, j] = x
        Ws = Wd * s[None, :]
        M = Ws @ Ws.T
        mubar = np.trace(M) / d
        fo_norm = float(np.linalg.norm(Z["s_ffn_out"][i, LI].astype(np.float64)))
        target_norm = max(0.05, fo_norm) * REL_SCALE
        w_step = STEP0_W if st[i]["rel_step"] == 0 else 1.0
        for name, lam in ARMS:
            if np.isinf(lam):
                z = g
            else:
                A_ = M + (lam * mubar + 1e-10 * mubar) * np.eye(d)
                z = np.linalg.solve(A_, g)
            du = s * (Wd.T @ z)
            induced = Wd @ (s * du)
            du = du * (target_norm / np.linalg.norm(induced))          # the team's per-state normalisation
            U[name][:, j] = du * OUT_GAIN * w_step
            tgt_stats[name]["gain_per_state"].append(float(g @ (Wd @ (s * du))))
            tgt_stats[name]["du_norm"].append(float(np.linalg.norm(du)))
        if (j + 1) % 40 == 0:
            print(f"  targets {j+1}/{len(train)}  {time.time()-t0:.0f}s", flush=True)

    # ---- compile each arm with the team's projector -------------------------------------------------
    K0 = Z["n_key"][:, LI, :].astype(np.float64).T
    K0 = K0 / np.sqrt(K0.shape[1])
    anc_keys = np.stack([Z["s_key"][i, LI].astype(np.float64) for i in anchors], axis=1)
    cfg = BetaEditConfig(d_in=d, d_out=dff, max_rank=8, max_slots=8, lambda_leakage=250.0, lambda_ridge=4.0,
                         nullspace_threshold=1e9, threshold_mode="absolute", radix_scale_exp2=24, quant_dtype="f16")
    proj = BetaEditNullspaceProjector(Wu, cfg)
    proj.initialize_anchor_covariance(K0)
    print(f"projector ready, soft null-space dim {proj.nullspace_dim}  {time.time()-t0:.0f}s", flush=True)

    held_state = {i: state(i) for i in held}
    anc_state = {i: state(i) for i in anchors}
    train_state_g = None

    def evaluate(Bm, Am, idx, cache, scale):
        rows = []
        for i in idx:
            h, s, x, R0sq, g, c = cache[i]
            dh = scale * (Wd @ (s * (Bm @ (Am @ x))))
            h1 = h + dh
            par = (dh @ h) / (h @ h)
            dperp = dh - par * h
            supp = sorted(set(int(t) for t in top_ids[i]) | set(st[i]["pos_toks"]) | set(st[i]["neg_toks"]))
            Es = E[[trow[t] for t in supp]]
            b_lp, e_lp = log_softmax(Es @ normed(h)), log_softmax(Es @ normed(h1))
            r = {"eta": float(dperp @ dperp / R0sq), "dh_over_R0": float(np.linalg.norm(dh) / np.sqrt(R0sq)),
                 "kl": float(np.sum(np.exp(b_lp) * (b_lp - e_lp))), "top1_changed": bool(np.argmax(b_lp) != np.argmax(e_lp))}
            if g is not None:
                pi = [supp.index(t) for t in st[i]["pos_toks"]]
                r["gain_first_order"] = float(g @ dh)
                r["gain_exact"] = float(c @ normed(h1) - c @ normed(h))
                r["dlogp_pos"] = float(np.mean(e_lp[pi] - b_lp[pi]))
                r["cos_dh_g"] = float((g @ dh) / (np.linalg.norm(g) * np.linalg.norm(dh) + 1e-300))
            rows.append(r)
        return rows

    def agg(rows, keys):
        return {k: {"mean": float(np.mean([r[k] for r in rows])), "median": float(np.median([r[k] for r in rows]))} for k in keys if k in rows[0]}

    results = {}
    compiled = {}
    for name, lam in ARMS:
        patch = proj.stage_edit(K, U[name], is_residual=True, rank=8, preservation_keys=anc_keys, preservation_weight=2.0, slot_name=name)
        Bm, Am = np.asarray(patch.factor_b, dtype=np.float64), np.asarray(patch.factor_a, dtype=np.float64)
        compiled[name] = (Bm, Am)
        dW_fro = float(np.sqrt(np.sum((Bm.T @ Bm) * (Am @ Am.T))))
        sv = np.linalg.svd(np.linalg.qr(Bm)[1] @ np.linalg.qr(Am.T)[1].T, compute_uv=False)
        k0_leak = float(np.linalg.norm(Bm @ (Am @ K0)))
        nat_h = evaluate(Bm, Am, held, held_state, 1.0)
        nat_a = evaluate(Bm, Am, anchors, anc_state, 1.0)
        results[name] = {
            "lambda_rel": None if np.isinf(lam) else lam,
            "target_mean_first_order_gain_per_train_state": float(np.mean(tgt_stats[name]["gain_per_state"])),
            "target_mean_du_norm": float(np.mean(tgt_stats[name]["du_norm"])),
            "adapter": {"fro_norm": dW_fro, "spectral_norm": float(sv[0]), "effective_rank": int(patch.effective_rank),
                        "truncation_rel_error": float(patch.truncation_rel_error), "k0_noise_leak_fro": k0_leak},
            "native_scale": {"held_out": agg(nat_h, ["gain_exact", "gain_first_order", "dlogp_pos", "eta", "kl", "cos_dh_g", "dh_over_R0"]),
                             "held_out_top1_changed_frac": float(np.mean([r["top1_changed"] for r in nat_h])),
                             "anchors": agg(nat_a, ["eta", "kl", "dh_over_R0"]),
                             "anchors_top1_changed_frac": float(np.mean([r["top1_changed"] for r in nat_a]))},
        }
        print(f"compiled {name}: fro {dW_fro:.4f} spec {sv[0]:.4f} trunc {patch.truncation_rel_error:.3f}  {time.time()-t0:.0f}s", flush=True)

    # ---- equal-benefit comparison: rescale every arm to the plain arm's mean held-out exact gain -----
    target_gain = results["plain"]["native_scale"]["held_out"]["gain_first_order"]["mean"]
    for name, _ in ARMS:
        Bm, Am = compiled[name]
        sc = target_gain / results[name]["native_scale"]["held_out"]["gain_first_order"]["mean"]
        eh = evaluate(Bm, Am, held, held_state, sc)
        ea = evaluate(Bm, Am, anchors, anc_state, sc)
        results[name]["equal_gain"] = {
            "scale_applied": float(sc), "scaled_fro_norm": float(sc * results[name]["adapter"]["fro_norm"]),
            "scaled_k0_noise_leak_fro": float(sc * results[name]["adapter"]["k0_noise_leak_fro"]),
            "held_out": agg(eh, ["gain_exact", "gain_first_order", "dlogp_pos", "eta", "kl", "cos_dh_g", "dh_over_R0"]),
            "held_out_top1_changed_frac": float(np.mean([r["top1_changed"] for r in eh])),
            "anchors": agg(ea, ["eta", "kl", "dh_over_R0"]),
            "anchors_top1_changed_frac": float(np.mean([r["top1_changed"] for r in ea])),
        }
    out = {"schema": "geminus-compile-exp/v1", "layer": L, "n_train": len(train), "n_held_out": len(held), "n_anchors": len(anchors),
           "recipe": {"rank": 8, "lambda_leakage": 250.0, "lambda_ridge": 4.0, "preservation_weight": 2.0, "quant": "f16",
                      "residual_rel_scale": REL_SCALE, "output_gain": OUT_GAIN, "rel_step0_weight": STEP0_W,
                      "note": "safety directions not registered (not available on geminus)"},
           "equal_gain_target_mean_first_order_gain": target_gain, "results": results, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    np.savez(OUT_JSON.replace(".json", ".factors.npz"), **{f"{n}_B": compiled[n][0] for n in compiled}, **{f"{n}_A": compiled[n][1] for n in compiled})

    print(f"\n{'arm':8s} {'|dW|_F':>8s} {'spec':>7s} {'K0leak':>8s} | native held-out: {'gain':>7s} {'dlogp+':>7s} {'eta':>9s} {'KL':>9s} {'cos':>6s} | anchors {'KL':>9s} {'top1chg':>7s}")
    for name, _ in ARMS:
        r = results[name]; n = r["native_scale"]
        print(f"{name:8s} {r['adapter']['fro_norm']:8.4f} {r['adapter']['spectral_norm']:7.4f} {r['adapter']['k0_noise_leak_fro']:8.4f} | "
              f"{'':17s}{n['held_out']['gain_exact']['mean']:7.4f} {n['held_out']['dlogp_pos']['mean']:7.4f} {n['held_out']['eta']['mean']:9.6f} {n['held_out']['kl']['mean']:9.6f} {n['held_out']['cos_dh_g']['mean']:6.3f} | "
              f"{'':8s}{n['anchors']['kl']['mean']:9.6f} {n['anchors_top1_changed_frac']:7.3f}")
    print(f"\nrescaled to equal mean held-out first-order gain = {target_gain:.4f}")
    print(f"{'arm':8s} {'scale':>7s} {'|dW|_F':>8s} {'K0leak':>8s} | held-out: {'gain':>7s} {'dlogp+':>7s} {'eta':>9s} {'KL':>9s} | anchors {'eta':>9s} {'KL':>9s} {'top1chg':>7s}")
    for name, _ in ARMS:
        e = results[name]["equal_gain"]
        print(f"{name:8s} {e['scale_applied']:7.3f} {e['scaled_fro_norm']:8.4f} {e['scaled_k0_noise_leak_fro']:8.4f} | "
              f"{'':10s}{e['held_out']['gain_exact']['mean']:7.4f} {e['held_out']['dlogp_pos']['mean']:7.4f} {e['held_out']['eta']['mean']:9.6f} {e['held_out']['kl']['mean']:9.6f} | "
              f"{'':8s}{e['anchors']['eta']['mean']:9.6f} {e['anchors']['kl']['mean']:9.6f} {e['anchors_top1_changed_frac']:7.3f}")


if __name__ == "__main__":
    main()
