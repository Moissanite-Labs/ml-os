"""Digest the runtime probe: noise floor, per-row LoRA delivery, shipped-adapter effect, propagation."""
from __future__ import annotations

import json
import sys

import numpy as np

SCRIPTS, MODEL, HARVEST_PREFIX, PROBE_PREFIX, LORA_DIR, OUT_JSON = sys.argv[1:7]
sys.path.insert(0, SCRIPTS)
from ggufw import Weights  # noqa: E402

LAYERS = [28, 29, 30, 31]
EPS = 1e-6
ADAPTERS = ["corpus", "weft-eido", "formal-verification"]


def silu(z):
    return z / (1.0 + np.exp(-z))


def nonneg_min_quadratic(Gm):
    L = Gm.shape[0]
    best, best_a = None, None
    for mask in range(1, 1 << L):
        idx = [i for i in range(L) if mask >> i & 1]
        try:
            w = np.linalg.solve(Gm[np.ix_(idx, idx)], np.ones(len(idx)))
        except np.linalg.LinAlgError:
            continue
        if np.any(w <= 0):
            continue
        a = np.zeros(L); a[idx] = w / w.sum()
        val = float(a @ Gm @ a)
        if best is None or val < best:
            best, best_a = val, a
    return best_a, best


def main():
    P = json.load(open(PROBE_PREFIX + ".json", encoding="utf-8"))
    Zp = np.load(PROBE_PREFIX + ".npz")
    hm = json.load(open(HARVEST_PREFIX + ".meta.json", encoding="utf-8"))
    Zh = np.load(HARVEST_PREFIX + ".npz")
    st = hm["states"]
    d = hm["d_model"]
    S = len(st)
    groups = np.array([s["group"] for s in st])
    out: dict = {"schema": "geminus-runtime-probe-analysis/v1"}

    # ---------------- P1 / P2 ----------------
    p1 = P["p1_noise_floor"]
    rows = P["p2_lora_rows"]
    out["p1"] = p1
    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)
    p2 = {}
    for v, rs in by_var.items():
        p2[v] = {
            "n_rows": len(rs),
            "w0_fork_margin_min_max": [min(r["w0_fork_margin"] for r in rs), max(r["w0_fork_margin"] for r in rs)],
            "w0_noise_vs_ref_max_abs_logit": max(r["w0_noise_vs_ref_max_abs"] for r in rs),
            "w0_noise_vs_ref_l2_max": max(r["w0_noise_vs_ref_l2"] for r in rs),
            "triple_fork_margin_min_max": [min(r["triple_fork_margin"] for r in rs), max(r["triple_fork_margin"] for r in rs)],
            "triple_delta_corr_min": min(r["triple_delta_corr_with_ref_delta"] for r in rs),
            "triple_delta_projection_min_max": [min(r["triple_delta_projection_on_ref"] for r in rs), max(r["triple_delta_projection_on_ref"] for r in rs)],
            "fv_solo_delta_corr_min": min(r["fv_solo_delta_corr_with_ref_delta"] for r in rs),
            "w0_top1_set": sorted({r["w0_top1"] for r in rs}), "triple_top1_set": sorted({r["triple_top1"] for r in rs}),
            "rows_where_triple_top1_equals_w0_top1": [r["seq"] for r in rs if r["triple_top1"] == r["w0_top1"]],
        }
    out["p2_by_variant"] = p2

    # ---------------- P3: shipped adapters, live vs exact ----------------
    W = Weights(MODEL)
    g_out = W.get("output_norm.weight")
    Wd = {l: W.get(f"blk.{l}.ffn_down.weight") for l in LAYERS}
    H = Zp["H"].astype(np.float64)                      # (5, S, d): base, corpus, eido, fv, triple
    contrast = Zp["contrast"]
    top1 = Zp["top1"]
    conds = P["p3_conditions"]
    x31 = Zh["s_key"][:, 3, :].astype(np.float64)
    s31 = silu(Zh["s_gate_pre"][:, 3, :].astype(np.float64))
    h_base_live = H[0]
    h_base_harv = Zh["s_l_out"][:, 3, :].astype(np.float64)
    out["p3_base_live_vs_harvest_relerr"] = float(np.linalg.norm(h_base_live - h_base_harv) / np.linalg.norm(h_base_harv))
    out["p3_base_repeat_max_abs_diff"] = P["p3_base_repeat_max_abs_diff_l_out31"]
    R0 = np.sqrt(np.sum(h_base_live ** 2, axis=1) + d * EPS)
    hhat = h_base_live / np.linalg.norm(h_base_live, axis=1, keepdims=True)
    factors = {}
    for n in ADAPTERS:
        Wl = Weights(f"{LORA_DIR}/{n}.f16.gguf")
        A = Wl.get("blk.31.ffn_up.weight.lora_a")    # (8, 2560)
        B = Wl.get("blk.31.ffn_up.weight.lora_b")    # (9216, 8)
        factors[n] = (B, A)
    pred = {n: ((s31 * (x31 @ factors[n][1].T @ factors[n][0].T)) @ Wd[31].T) for n in ADAPTERS}     # (S, d)
    pred["triple"] = sum(pred[n] for n in ADAPTERS)
    p3 = {}
    for ci, cname in enumerate(conds):
        if cname == "base":
            continue
        dh = H[ci] - h_base_live
        pr = pred[cname]
        par = np.sum(dh * hhat, axis=1)
        dperp = dh - par[:, None] * hhat
        eta = np.sum(dperp ** 2, axis=1) / R0 ** 2
        rec = {"live_vs_exact_pred_relerr_all_states": float(np.linalg.norm(dh - pr) / np.linalg.norm(pr)),
               "live_vs_exact_pred_cos_mean": float(np.mean(np.sum(dh * pr, axis=1) / (np.linalg.norm(dh, axis=1) * np.linalg.norm(pr, axis=1) + 1e-300))),
               "by_group": {}}
        for g in ("fv", "corpus", "eido", "anchor"):
            m = groups == g
            dc = contrast[ci, m] - contrast[0, m]
            ok = ~np.isnan(dc)
            rec["by_group"][g] = {
                "n": int(m.sum()),
                "dh_norm_mean": float(np.linalg.norm(dh[m], axis=1).mean()),
                "pred_dh_norm_mean": float(np.linalg.norm(pr[m], axis=1).mean()),
                "dh_over_R0_mean": float((np.linalg.norm(dh[m], axis=1) / R0[m]).mean()),
                "eta_mean": float(eta[m].mean()), "eta_max": float(eta[m].max()),
                "radial_coord_mean": float((par[m] / np.linalg.norm(h_base_live[m], axis=1)).mean()),
                "contrast_gain_mean": float(dc[ok].mean()) if ok.any() else None,
                "contrast_gain_median": float(np.median(dc[ok])) if ok.any() else None,
                "contrast_gain_frac_positive": float(np.mean(dc[ok] > 0)) if ok.any() else None,
                "top1_changed_frac": float(np.mean(top1[ci, m] != top1[0, m])),
            }
        p3[cname] = rec
    out["p3_shipped_adapters"] = p3

    # ---------------- P4: propagation of injected perturbations ----------------
    probes = P["p4_probes"]
    p4 = {}
    for l in LAYERS:
        ps = [p for p in probes if p["layer"] == l]
        rec = {"n": len(ps)}
        for key in ("plain_eps0.03", "plain_eps0.10", "random_eps0.10"):
            vals = [p[key] for p in ps]
            rec[key] = {f: {"mean": float(np.mean([v[f] for v in vals if v[f] is not None])), "median": float(np.median([v[f] for v in vals if v[f] is not None]))}
                        for f in ("amp", "cos_with_injected", "tangent_energy_ratio", "gain_real_over_pred", "even_over_odd_norm", "radial_frac_of_odd")}
        p4[str(l)] = rec
    out["p4_propagation_by_layer"] = p4

    # allocation with MEASURED propagation: unit-realised-gain propagated perturbations
    sel = Zp["sel"]
    DH = Zp["DH"].astype(np.float64)                   # (n_sel, 4, d) odd response per unit injected norm
    DIRS = Zp["DIRS"].astype(np.float64)
    tok_ids = sorted({t for i in sel for t in st[i]["pos_toks"] + st[i]["neg_toks"]})
    E = W.get_rows("token_embd.weight", tok_ids)
    trow = {t: k for k, t in enumerate(tok_ids)}
    Gm_prop, Gm_pass = [], []
    gain_ratio = []
    for k, si in enumerate(sel):
        h0 = h_base_live[si]
        R = R0[si]
        c = E[[trow[t] for t in st[si]["pos_toks"]]].mean(axis=0) - E[[trow[t] for t in st[si]["neg_toks"]]].mean(axis=0)
        u = g_out * c
        g = (np.sqrt(d) / R) * (u - (u @ h0) / (R * R) * h0)
        gh = g / np.linalg.norm(g)
        hh = hhat[si]
        Yp, Ys = [], []
        ok = True
        for li in range(4):
            gp, gs = gh @ DH[k, li], gh @ DIRS[k, li]
            if gp <= 1e-6 or gs <= 1e-6:
                ok = False
                break
            yp = DH[k, li] / gp; yp = yp - (yp @ hh) * hh
            ys = DIRS[k, li] / gs; ys = ys - (ys @ hh) * hh
            Yp.append(yp); Ys.append(ys)
        if not ok:
            continue
        Yp, Ys = np.array(Yp), np.array(Ys)
        Gm_prop.append(Yp @ Yp.T); Gm_pass.append(Ys @ Ys.T)
        gain_ratio.append([(gh @ DH[k, li]) / (gh @ DIRS[k, li]) for li in range(4)])
    Gm_prop, Gm_pass = np.array(Gm_prop), np.array(Gm_pass)

    def alloc(Gm):
        Gbar = Gm.mean(axis=0)
        dg = np.sqrt(np.einsum("sll->sl", Gm))
        cos = (Gm / (dg[:, :, None] * dg[:, None, :])).mean(axis=0)
        uni = np.full(4, 0.25)
        diag = 1.0 / np.diag(Gbar); diag /= diag.sum()
        opt, _ = nonneg_min_quadratic(Gbar)
        last = np.array([0, 0, 0, 1.0])
        res = {"mean_cosine_matrix": cos.tolist(), "mean_diag": np.diag(Gbar).tolist()}
        for name, a in (("uniform", uni), ("diagonal_law", diag), ("gram_optimal_nonneg", opt), ("single_layer_31", last)):
            res[name] = {"alpha": a.tolist(), "energy_per_unit_gain_sq": float(np.mean(np.einsum("l,slm,m->s", a, Gm, a)))}
        return res
    out["p4_allocation"] = {"n_states_used": int(len(Gm_prop)), "measured_propagation": alloc(Gm_prop), "pass_through_same_states": alloc(Gm_pass),
                            "realised_over_passthrough_gain_mean_by_layer": np.mean(gain_ratio, axis=0).tolist(),
                            "note": "energy is tangent energy at the final residual per unit realised contrast gain^2, in units where the min-norm floor is 1"}
    out["thermal"] = P["thermal"]
    out["elapsed_s"] = P["elapsed_s"]
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)

    # ---------------- print ----------------
    print("== P1 noise floor ==")
    print(" repeat identical decode, max |dlogit|:", p1["repeat_single_whole_w0_max_abs_logit_diff"])
    print(" W0 second-token top5:", p1["reference_w0"]["second_token_top5"], "fork margin (fence - think):", round(p1["reference_w0"]["fork_margin_fence_minus_think"], 4))
    print(" triple second-token top5:", p1["reference_triple"]["second_token_top5"], "fork margin:", round(p1["reference_triple"]["fork_margin_fence_minus_think"], 4))
    print(" reference LoRA delta at fork:", json.dumps(p1["reference_lora_delta_second_token"]))
    print("== P2 per-variant ==")
    for v, r in p2.items():
        print(f" {v:20s} rows {r['n_rows']} | W0 margin [{r['w0_fork_margin_min_max'][0]:+.4f},{r['w0_fork_margin_min_max'][1]:+.4f}] noise max|dlogit| {r['w0_noise_vs_ref_max_abs_logit']:.4f}"
              f" | triple margin [{r['triple_fork_margin_min_max'][0]:+.4f},{r['triple_fork_margin_min_max'][1]:+.4f}] delta-corr min {r['triple_delta_corr_min']:.4f}"
              f" proj [{r['triple_delta_projection_min_max'][0]:.3f},{r['triple_delta_projection_min_max'][1]:.3f}] | top1 W0 {r['w0_top1_set']} triple {r['triple_top1_set']} same-as-W0 rows {r['rows_where_triple_top1_equals_w0_top1']}")
    print("== P3 shipped adapters (live GPU) ==")
    print(f" base live vs harvest relerr {out['p3_base_live_vs_harvest_relerr']:.2e}; base repeat max abs diff {out['p3_base_repeat_max_abs_diff']:.3e}")
    for cname, rec in p3.items():
        print(f" [{cname}] live vs exact float64 prediction: relerr {rec['live_vs_exact_pred_relerr_all_states']:.4f}, cos {rec['live_vs_exact_pred_cos_mean']:.4f}")
        for g, b in rec["by_group"].items():
            cg = "   n/a" if b["contrast_gain_mean"] is None else f"{b['contrast_gain_mean']:+.4f} (median {b['contrast_gain_median']:+.4f}, {100*b['contrast_gain_frac_positive']:.0f}% >0)"
            print(f"    {g:7s} n={b['n']:3d} |dh| {b['dh_norm_mean']:.4f} (pred {b['pred_dh_norm_mean']:.4f}) |dh|/R0 {b['dh_over_R0_mean']:.5f} eta mean {b['eta_mean']:.2e} max {b['eta_max']:.2e} contrast gain {cg} top1 changed {100*b['top1_changed_frac']:.1f}%")
    print("== P4 propagation (odd response to +/- injection after layer l, all positions) ==")
    for l, rec in p4.items():
        for key in ("plain_eps0.03", "plain_eps0.10", "random_eps0.10"):
            r = rec[key]
            print(f" l{l} {key:15s} amp {r['amp']['mean']:.3f} cos {r['cos_with_injected']['mean']:.3f} tangent-energy x{r['tangent_energy_ratio']['mean']:.3f} gain real/pred {r['gain_real_over_pred']['mean']:.3f} (median {r['gain_real_over_pred']['median']:.3f}) even/odd {r['even_over_odd_norm']['mean']:.3f}")
    pa = out["p4_allocation"]
    for tag in ("measured_propagation", "pass_through_same_states"):
        a = pa[tag]
        print(f" allocation [{tag}] n={pa['n_states_used']}: " + " | ".join(f"{k} {a[k]['energy_per_unit_gain_sq']:.3f} a={[round(x,3) for x in a[k]['alpha']]}" for k in ("uniform", "diagonal_law", "gram_optimal_nonneg", "single_layer_31")))
        print("    cosines: " + "; ".join(" ".join(f"{x:.3f}" for x in row) for row in a["mean_cosine_matrix"]))
    print(" realised/pass-through gain by layer:", [round(x, 3) for x in pa["realised_over_passthrough_gain_mean_by_layer"]])
    print(" thermal:", P["thermal"], "elapsed", P["elapsed_s"])


if __name__ == "__main__":
    main()
