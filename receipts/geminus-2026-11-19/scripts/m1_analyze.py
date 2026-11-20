"""M1 analysis on the geminus harvest: residual norms, gate-metric sensitivities, cross-layer
coherence, and the allocation law derived from first principles.

Model (exact for an edit of W_up at one layer, given that layer's key x and gate s):
    delta_l = W_down^l (s_l * du_l),   du_l = dW_up^l x_l          (FFN is linear in du)
Final-norm contrast gradient (tangent-projected), g, satisfies d(logit contrast) ~= <g, dh>.
Pull-back to the up-projection output:  a_l = s_l * (W_down^l^T g).
The plain pulled-back residual du = t a_l (the team harvester's choice) gives
    dh = t M_l g,   M_l = W_down^l diag(s_l^2) W_down^l^T   (the "gate metric", d x d PSD)
    gain = t g^T M_l g,   ||dh||^2 = t^2 g^T M_l^2 g.
Under residual pass-through (P_l = I, exact for l = last layer) the tangent energy per unit
gain^2 is  kappa_l / ||g||^2  with  kappa_l = (g^ M^2 g^) / (g^ M g^)^2 >= 1  (g^ = g/||g||),
and the minimum over ALL du is 1/||g||^2, attained by du = s * W_down^T M^{-1} g (dh parallel g).
"""
from __future__ import annotations

import json
import sys

import numpy as np

SCRIPTS, MODEL, HARVEST_PREFIX, OUT_JSON = sys.argv[1:5]
sys.path.insert(0, SCRIPTS)
from ggufw import Weights  # noqa: E402

LAYERS = [28, 29, 30, 31]
EPS = 1e-6
MAC = {"r_norms": [62.740422, 65.498958, 64.250502, 58.851399], "final_pre_rmsnorm_mean_r0": 62.703345,
       "gammas": [6.805068, 6.186092, 4.709198, 2.073155], "rhos": [17.24481, 26.633308, 21.047025, 37.780445]}


def silu(z):
    return z / (1.0 + np.exp(-z))


def bootstrap_ci(x, fn=np.mean, n=2000, seed=20261119):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = np.array([fn(x[i]) for i in idx])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def nonneg_min_quadratic(Gm):
    """argmin a^T G a  s.t. sum a = 1, a >= 0  (tiny problem: enumerate supports)."""
    L = Gm.shape[0]
    best, best_a = None, None
    for mask in range(1, 1 << L):
        idx = [i for i in range(L) if mask >> i & 1]
        sub = Gm[np.ix_(idx, idx)]
        try:
            w = np.linalg.solve(sub + 1e-15 * np.eye(len(idx)), np.ones(len(idx)))
        except np.linalg.LinAlgError:
            continue
        if np.any(w <= 0):
            continue
        a = np.zeros(L)
        a[idx] = w / w.sum()
        val = float(a @ Gm @ a)
        if best is None or val < best:
            best, best_a = val, a
    return best_a, best


def main():
    meta = json.load(open(HARVEST_PREFIX + ".meta.json", encoding="utf-8"))
    Z = np.load(HARVEST_PREFIX + ".npz")
    states = meta["states"]
    S = len(states)
    d, dff = meta["d_model"], meta["d_ff"]
    W = Weights(MODEL)
    g_out = W.get("output_norm.weight")
    Wd = {l: W.get(f"blk.{l}.ffn_down.weight") for l in LAYERS}        # (d, d_ff)

    h_mid = Z["s_h_mid"].astype(np.float64)     # (S, 4, d)
    l_out = Z["s_l_out"].astype(np.float64)
    l_out27 = Z["s_l_out27"].astype(np.float64)
    ffn_out = Z["s_ffn_out"].astype(np.float64)
    s_gate = silu(Z["s_gate_pre"].astype(np.float64))   # (S, 4, d_ff)
    up0 = Z["s_up"].astype(np.float64)
    groups = np.array([s["group"] for s in states])
    out: dict = {"schema": "geminus-m1-analysis/v1", "layers": LAYERS, "n_states": S,
                 "group_counts": {g: int(np.sum(groups == g)) for g in sorted(set(groups))}}

    # ---- A. cross-implementation validation against the team's bf16 PyTorch harvest ---------
    ref = np.array([s["ref_ffn_out_norm"] if s["ref_ffn_out_norm"] is not None else np.nan for s in states])
    mine = np.linalg.norm(ffn_out[:, 3, :], axis=1)
    ok = ~np.isnan(ref)
    ratio = mine[ok] / ref[ok]
    out["validation_vs_mac_bf16_ffn_out_norm_blk31"] = {
        "n": int(ok.sum()), "ratio_mean": float(ratio.mean()), "ratio_std": float(ratio.std()),
        "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
        "max_abs_rel_dev": float(np.max(np.abs(ratio - 1.0))),
        "pearson_r": float(np.corrcoef(mine[ok], ref[ok])[0, 1]),
    }
    # FFN linearity check of the harvested tensors against dequantised weights (runtime numerics)
    recon = np.einsum("sf,df->sd", s_gate[:, 3, :] * up0[:, 3, :], Wd[31])
    out["runtime_vs_exact_ffn_down_relerr_blk31"] = float(np.linalg.norm(recon - ffn_out[:, 3, :]) / np.linalg.norm(ffn_out[:, 3, :]))

    # ---- B. residual norms ------------------------------------------------------------------
    def norms_block(mask):
        r = {"l_out": [float(np.linalg.norm(l_out27[mask], axis=1).mean())] +
                      [float(np.linalg.norm(l_out[mask, li], axis=1).mean()) for li in range(4)],
             "attn_residual": [float(np.linalg.norm(h_mid[mask, li], axis=1).mean()) for li in range(4)],
             "ffn_out": [float(np.linalg.norm(ffn_out[mask, li], axis=1).mean()) for li in range(4)]}
        return r
    out["residual_norms_decision_states"] = {"index_note": "l_out lists layers 27..31; attn_residual/ffn_out list 28..31",
                                             "all": norms_block(np.ones(S, bool))}
    for g in sorted(set(groups)):
        out["residual_norms_decision_states"][g] = norms_block(groups == g)
    out["residual_norms_noise_last_pos"] = {
        "l_out": Z["n_l_out_norm"].mean(axis=0).tolist(), "attn_residual": Z["n_h_mid_norm"].mean(axis=0).tolist(),
        "ffn_out": Z["n_ffn_out_norm"].mean(axis=0).tolist(), "n": int(Z["n_l_out_norm"].shape[0])}
    prof = meta["norm_profiles"]
    if prof:
        allpos = {"l_out": [], "attn_residual": []}
        keys_lo = ["l_out-27"] + [f"l_out-{l}" for l in LAYERS]
        allpos["l_out"] = [float(np.mean(np.concatenate([np.asarray(p[k]) for p in prof]))) for k in keys_lo]
        allpos["attn_residual"] = [float(np.mean(np.concatenate([np.asarray(p[f"attn_residual-{l}"]) for p in prof]))) for l in LAYERS]
        allpos["n_tokens_total"] = int(sum(p["n_tokens"] for p in prof))
        # first-position (BOS-like) outlier check
        allpos["l_out_first_position_mean"] = [float(np.mean([p[k][0] for p in prof])) for k in keys_lo]
        allpos["l_out_excluding_first_position"] = [float(np.mean(np.concatenate([np.asarray(p[k][1:]) for p in prof]))) for k in keys_lo]
        out["residual_norms_all_positions_chat_prompts"] = allpos

    # ---- C. contrast gradients and gate-metric quantities -------------------------------------
    cmask = np.array([bool(s["pos_toks"]) and bool(s["neg_toks"]) for s in states])
    cidx = np.where(cmask)[0]
    tok_ids = sorted({t for i in cidx for t in states[i]["pos_toks"] + states[i]["neg_toks"]})
    tok_row = {t: k for k, t in enumerate(tok_ids)}
    E = W.get_rows("token_embd.weight", tok_ids)
    h = l_out[:, 3, :]
    R0sq = np.sum(h * h, axis=1) + d * EPS
    G = np.zeros((S, d))
    for i in cidx:
        c = E[[tok_row[t] for t in states[i]["pos_toks"]]].mean(axis=0) - E[[tok_row[t] for t in states[i]["neg_toks"]]].mean(axis=0)
        u = g_out * c
        G[i] = (np.sqrt(d) / np.sqrt(R0sq[i])) * (u - (u @ h[i]) / R0sq[i] * h[i])
    gn_all = np.linalg.norm(G[cidx], axis=1)
    degenerate = [states[i]["id"] for i, n in zip(cidx, gn_all) if n <= 1e-9]
    out["zero_contrast_states_excluded"] = {"n": len(degenerate), "ids": degenerate,
        "note": "positive and negative token sets identical -> contrast direction and target residual are exactly zero"}
    cidx = cidx[gn_all > 1e-9]
    gn = np.linalg.norm(G[cidx], axis=1)
    Gh = G[cidx] / gn[:, None]                     # unit contrast gradients (n_c, d)
    hc = h[cidx]
    hhat = hc / np.linalg.norm(hc, axis=1, keepdims=True)
    n_c = len(cidx)
    out["n_contrast_states"] = int(n_c)
    out["contrast_gradient_norm"] = {"mean": float(gn.mean()), "median": float(np.median(gn)), "min": float(gn.min()), "max": float(gn.max())}

    sig = np.zeros((n_c, 4)); q2 = np.zeros((n_c, 4)); V = np.zeros((n_c, 4, d))
    sig_p = np.zeros((n_c, 4)); Vp = np.zeros((n_c, 4, d))
    gate_stats = []
    for li, l in enumerate(LAYERS):
        s = s_gate[cidx, li, :]                    # (n_c, d_ff)
        wg = Gh @ Wd[l]                            # W_down^T g^   (n_c, d_ff)
        a = s * wg
        v = (s * a) @ Wd[l].T                      # M g^           (n_c, d)
        sig[:, li] = np.sum(a * a, axis=1)         # g^T M g^
        q2[:, li] = np.sum(v * v, axis=1)          # g^T M^2 g^
        V[:, li] = v
        # weft radial-null projection: remove from u the component along s*(W_down^T h_pre)
        wh = hc @ Wd[l]
        qh = s * wh
        qn = np.linalg.norm(qh, axis=1, keepdims=True)
        qhat = qh / qn
        a_p = a - np.sum(a * qhat, axis=1, keepdims=True) * qhat
        vp = (s * a_p) @ Wd[l].T
        Vp[:, li] = vp
        sig_p[:, li] = np.sum(vp * Gh, axis=1)
        gate_stats.append({"layer": l, "gate_rms": float(np.sqrt(np.mean(s * s))), "gate_max_abs_mean": float(np.mean(np.max(np.abs(s), axis=1))),
                           "gate_frac_negative": float(np.mean(s < 0)), "trace_M_over_d_mean": float(np.mean(np.sum((s * s) * np.sum(Wd[l] * Wd[l], axis=0)[None, :], axis=1)) / d)})
    out["gate_statistics"] = gate_stats
    kappa = q2 / sig ** 2
    gamma_a = np.sqrt(q2 / sig)                    # ||M g^|| / ||a||  : post-W_down norm per unit up-residual norm
    rho_a = 1.0 / sig                              # 1 / (g^ M g^)
    cos_gain = 1.0 / np.sqrt(kappa)

    def summ(x):
        return {"mean": x.mean(axis=0).tolist(), "median": np.median(x, axis=0).tolist(),
                "ci95_mean": [bootstrap_ci(x[:, j]) for j in range(x.shape[1])]}
    out["gate_metric_pass_through"] = {
        "definitions": {
            "sigma": "g^T M_l g^  (contrast gain per unit plain-pullback step; g^ unit contrast gradient at the final norm)",
            "gamma": "sqrt(g^T M_l^2 g^ / g^T M_l g^) = ||W_down(s*a)|| / ||a||, a = s*(W_down^T g^): post-W_down perturbation norm per unit up-residual norm",
            "rho": "1 / (g^T M_l g^): inverse gate-metric energy along the contrast gradient (gate alignment)",
            "kappa": "gamma^2 * rho = g^T M^2 g^ / (g^T M g^)^2 >= 1: perturbation energy per unit gain^2, relative to the minimum 1 (dh parallel g)",
        },
        "sigma": summ(sig), "gamma": summ(gamma_a), "rho": summ(rho_a), "kappa": summ(kappa), "cos_gain": summ(cos_gain),
        "gamma_rms_over_states": np.sqrt(np.mean(gamma_a ** 2, axis=0)).tolist(),
        "sqrt_sigma (=||a||, alt gamma)": summ(np.sqrt(sig)), "sqrt_q2 (=||M g^||, alt gamma)": summ(np.sqrt(q2)),
    }
    by_group = {}
    cg = groups[cidx]
    for g in sorted(set(cg)):
        m = cg == g
        by_group[g] = {"n": int(m.sum()), "kappa_mean": kappa[m].mean(axis=0).tolist(), "kappa_median": np.median(kappa[m], axis=0).tolist(),
                       "gamma_mean": gamma_a[m].mean(axis=0).tolist(), "rho_mean": rho_a[m].mean(axis=0).tolist()}
    out["gate_metric_by_group"] = by_group

    # ---- D. cross-layer coherence and allocation ------------------------------------------------
    # unit-gain perturbations (tangent to the final residual h): y_l = v_l / sigma_l
    def tangent(Vm):
        return Vm - np.sum(Vm * hhat[:, None, :], axis=2, keepdims=True) * hhat[:, None, :]

    def alloc_block(Vm, sg, label):
        Y = tangent(Vm) / sg[:, :, None]
        Gm = np.einsum("sld,smd->slm", Y, Y)                     # per-state Gram (n_c, 4, 4), units 1/gain^2
        dg = np.sqrt(np.einsum("sll->sl", Gm))
        Cos = Gm / (dg[:, :, None] * dg[:, None, :])
        Gbar = Gm.mean(axis=0)
        res = {"label": label, "mean_cosine_matrix": Cos.mean(axis=0).tolist(),
               "median_cosine_matrix": np.median(Cos, axis=0).tolist(), "mean_gram": Gbar.tolist()}
        L = 4
        uni = np.full(L, 1.0 / L)
        diag = (1.0 / np.diag(Gbar)); diag /= diag.sum()
        opt, _ = nonneg_min_quadratic(Gbar)
        last = np.array([0, 0, 0, 1.0])
        floor = 1.0  # per unit gain^2 in g^ units: ||g^||^-2 = 1
        costs = {}
        for name, a in (("uniform_1_over_L", uni), ("diagonal_law_1_over_Gll", diag), ("gram_optimal_nonneg", opt), ("single_layer_31", last)):
            per_state = np.einsum("l,slm,m->s", a, Gm, a)
            costs[name] = {"alpha": a.tolist(), "mean_energy_per_unit_gain_sq": float(per_state.mean()),
                           "median": float(np.median(per_state)), "ci95_mean": bootstrap_ci(per_state)}
        # what the diagonal (incoherent) model PREDICTS for each allocation vs what the Gram gives
        pred_uniform_incoherent = float(np.mean(np.einsum("l,sl->s", uni ** 2, np.einsum("sll->sl", Gm))))
        res["allocations"] = costs
        res["floor_min_norm_energy_per_unit_gain_sq"] = floor
        res["uniform_cost_if_layers_were_incoherent"] = pred_uniform_incoherent
        res["coherence_inflation_uniform"] = costs["uniform_1_over_L"]["mean_energy_per_unit_gain_sq"] / pred_uniform_incoherent
        res["uniform_vs_single_layer_31_ratio"] = costs["uniform_1_over_L"]["mean_energy_per_unit_gain_sq"] / costs["single_layer_31"]["mean_energy_per_unit_gain_sq"]
        res["ideal_incoherent_ratio_would_be"] = "sum_l (1/L^2) G_ll / G_33"
        return res
    out["allocation_plain_pullback"] = alloc_block(V, sig, "plain pulled-back residual a = s*(W_down^T g)")
    out["allocation_plain_pullback_radial_null"] = alloc_block(Vp, sig_p, "plain residual after weft radial-null projection")

    # shares in the team's product form, using this harvest's per-layer means
    r_l = np.array(out["residual_norms_decision_states"]["all"]["l_out"][1:])
    gm, rm = gamma_a.mean(axis=0), rho_a.mean(axis=0)
    share = r_l ** 2 / (gm ** 2 * rm); share /= share.sum()
    mac_share = np.array(MAC["r_norms"]) ** 2 / (np.array(MAC["gammas"]) ** 2 * np.array(MAC["rhos"])); mac_share /= mac_share.sum()
    out["product_law_shares"] = {"geminus": share.tolist(), "mac_e8_receipt": mac_share.tolist()}

    # ---- E. K0 soft-null-space leverage per layer (TangentBetaEdit convention: trace-normalised noise keys) ----
    K0 = Z["n_key"].astype(np.float64)            # (N, 4, d)
    keys = Z["s_key"].astype(np.float64)
    lev = []
    for li, l in enumerate(LAYERS):
        k0 = K0[:, li, :].T                        # (d, N)
        k0 = k0 * (np.sqrt(d) / np.linalg.norm(k0))          # trace-normalise: ||K0||_F^2 = d
        ev = np.linalg.eigvalsh(k0 @ k0.T)
        kx = keys[cidx, li, :]
        # fraction of contrast-state key energy outside the span of the top noise-key directions
        evals, evecs = np.linalg.eigh(k0 @ k0.T)
        top = evecs[:, -64:]
        frac_out = 1.0 - np.sum((kx @ top) ** 2, axis=1) / np.sum(kx * kx, axis=1)
        lev.append({"layer": l, "key_norm_mean": float(np.linalg.norm(kx, axis=1).mean()),
                    "noise_key_norm_mean": float(np.linalg.norm(K0[:, li, :], axis=1).mean()),
                    "k0_eig_top5_frac": (np.sort(ev)[::-1][:5] / ev.sum()).tolist(),
                    "k0_participation_ratio": float(ev.sum() ** 2 / np.sum(ev ** 2)),
                    "contrast_key_energy_outside_top64_noise_dirs_mean": float(frac_out.mean())})
    out["k0_noise_key_geometry"] = lev

    # ---- F. comparison with the Mac E8 spectra summary -----------------------------------------
    out["mac_reference"] = MAC
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)

    # ---- readable print ---------------------------------------------------------------------
    f4 = lambda xs: " ".join(f"{x:9.4f}" for x in xs)
    v = out["validation_vs_mac_bf16_ffn_out_norm_blk31"]
    print(f"[A] ||ffn_out-31|| vs Mac bf16 over {v['n']} states: ratio mean {v['ratio_mean']:.5f} std {v['ratio_std']:.5f} "
          f"range [{v['ratio_min']:.4f},{v['ratio_max']:.4f}] pearson {v['pearson_r']:.6f}")
    print(f"    runtime vs exact W_down matmul relerr (blk31): {out['runtime_vs_exact_ffn_down_relerr_blk31']:.5f}")
    print("[B] mean residual norms            l27       l28       l29       l30       l31")
    for g, blk in out["residual_norms_decision_states"].items():
        if isinstance(blk, dict):
            print(f"    decision/{g:8s} l_out     {f4(blk['l_out'])}")
            print(f"    decision/{g:8s} attn_res            {f4(blk['attn_residual'])}")
    nz = out["residual_norms_noise_last_pos"]
    print(f"    noise last pos    l_out     {f4(nz['l_out'])}")
    print(f"    noise last pos    attn_res            {f4(nz['attn_residual'])}")
    if "residual_norms_all_positions_chat_prompts" in out:
        ap = out["residual_norms_all_positions_chat_prompts"]
        print(f"    all positions     l_out     {f4(ap['l_out'])}   (first position: {f4(ap['l_out_first_position_mean'])})")
        print(f"    all pos excl. 1st l_out     {f4(ap['l_out_excluding_first_position'])}")
    print(f"    MAC r_norms (28..31)                {f4(MAC['r_norms'])}   final R0 {MAC['final_pre_rmsnorm_mean_r0']:.4f}")
    gmx = out["gate_metric_pass_through"]
    print("[C] gate metric (pass-through)        l28       l29       l30       l31")
    for k in ("sigma", "gamma", "rho", "kappa", "cos_gain"):
        print(f"    {k:9s} mean                 {f4(gmx[k]['mean'])}     median {f4(gmx[k]['median'])}")
    print(f"    sqrt(g'Mg) (alt gamma) mean      {f4(gmx['sqrt_sigma (=||a||, alt gamma)']['mean'])}")
    print(f"    ||M g^||   (alt gamma) mean      {f4(gmx['sqrt_q2 (=||M g^||, alt gamma)']['mean'])}")
    print(f"    MAC gammas                       {f4(MAC['gammas'])}")
    print(f"    MAC rhos                         {f4(MAC['rhos'])}")
    for g, b in by_group.items():
        print(f"    kappa mean [{g:6s} n={b['n']:3d}]        {f4(b['kappa_mean'])}")
    for key in ("allocation_plain_pullback", "allocation_plain_pullback_radial_null"):
        ab = out[key]
        print(f"[D] {ab['label']}")
        print("    mean cross-layer cosine of unit-gain tangent perturbations:")
        for row in ab["mean_cosine_matrix"]:
            print("       " + f4(row))
        for name, c in ab["allocations"].items():
            print(f"    {name:26s} alpha=[{f4(c['alpha'])}]  energy/gain^2 = {c['mean_energy_per_unit_gain_sq']:9.4f}  ci95 [{c['ci95_mean'][0]:.3f},{c['ci95_mean'][1]:.3f}]")
        print(f"    floor (dh || g): 1.0000 ; uniform if incoherent: {ab['uniform_cost_if_layers_were_incoherent']:.4f} ; coherence inflation: {ab['coherence_inflation_uniform']:.3f}")
    ps = out["product_law_shares"]
    print(f"[E] product-law shares geminus: {f4(ps['geminus'])}   mac: {f4(ps['mac_e8_receipt'])}")
    for r in lev:
        print(f"    K0 blk.{r['layer']}: key norm {r['key_norm_mean']:.2f} noise-key norm {r['noise_key_norm_mean']:.2f} "
              f"participation ratio {r['k0_participation_ratio']:.1f} top5 eig frac {[round(x,3) for x in r['k0_eig_top5_frac']]} "
              f"contrast-key energy outside top-64 noise dirs {r['contrast_key_energy_outside_top64_noise_dirs_mean']:.3f}")


if __name__ == "__main__":
    main()
