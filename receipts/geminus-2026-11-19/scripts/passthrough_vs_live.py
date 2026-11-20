"""Supporting numbers for packets 005/011 (saved after review round 1; replaces an ad-hoc snippet).

1. Pass-through FIRST-ORDER predicted gain of each single-layer adapter on the held-out states, against
   the live gain, plus the EXACT float64 gain at blk.31 (where the edit-to-residual map is exact).
2. Float64 anchor movement for the blk.31 adapter at the scale used in the live allocation test, as the
   baseline that the live anchor figure (which sits at the runtime noise floor) should be compared with.
3. How independent the "held-out" states are: prompt-id overlap, string similarity to the nearest
   training prompt, and reuse of (positive token, negative set) pairs.
4. How often the base model already ranks the positive token first.
"""
from __future__ import annotations

import difflib
import json
import sys

import numpy as np

SCRIPTS, MODEL, HARVEST_PREFIX, ADAPTER_DIR, LIVE_JSON, COMPILE_JSONL, OUT_JSON = sys.argv[1:8]
sys.path.insert(0, SCRIPTS)
from ggufw import Weights  # noqa: E402

LAYERS = [28, 29, 30, 31]
EPS = 1e-6


def silu(z):
    return z / (1.0 + np.exp(-z))


def main():
    meta = json.load(open(HARVEST_PREFIX + ".meta.json", encoding="utf-8"))
    Z = np.load(HARVEST_PREFIX + ".npz")
    man = json.load(open(ADAPTER_DIR + "/manifest.json", encoding="utf-8"))
    live = json.load(open(LIVE_JSON, encoding="utf-8"))
    st = meta["states"]
    d = meta["d_model"]
    W = Weights(MODEL)
    g_out = W.get("output_norm.weight")
    sqd = np.sqrt(d)
    held = set(man["held_out_ids"])
    fv = [i for i, s in enumerate(st) if s["group"] == "fv"]
    hidx = [i for i in fv if st[i]["id"] in held]
    tidx = [i for i in fv if st[i]["id"] not in held]
    anchors = [i for i, s in enumerate(st) if s["group"] == "anchor"]
    need = sorted({t for i in hidx for t in st[i]["pos_toks"] + st[i]["neg_toks"]})
    trow = {t: k for k, t in enumerate(need)}
    E = W.get_rows("token_embd.weight", need)

    def normed(hv):
        return g_out * hv * (sqd / np.sqrt(hv @ hv + d * EPS))
    G, C = np.zeros((len(hidx), d)), np.zeros((len(hidx), d))
    for j, i in enumerate(hidx):
        h = Z["s_l_out"][i, 3].astype(np.float64)
        R0sq = h @ h + d * EPS
        c = E[[trow[t] for t in st[i]["pos_toks"]]].mean(0) - E[[trow[t] for t in st[i]["neg_toks"]]].mean(0)
        u = g_out * c
        G[j] = (sqd / np.sqrt(R0sq)) * (u - (u @ h) / R0sq * h)
        C[j] = c
    out = {"schema": "geminus-passthrough-vs-live/v2", "n_held_out_states": len(hidx), "by_layer": {}}
    for li, l in enumerate(LAYERS):
        Wl = Weights(f"{ADAPTER_DIR}/plain_blk{l}.gguf")
        A, B = Wl.get(f"blk.{l}.ffn_up.weight.lora_a"), Wl.get(f"blk.{l}.ffn_up.weight.lora_b")
        Wd = W.get(f"blk.{l}.ffn_down.weight")
        x = Z["s_key"][hidx, li, :].astype(np.float64)
        s = silu(Z["s_gate_pre"][hidx, li, :].astype(np.float64))
        dh = (s * (x @ A.T @ B.T)) @ Wd.T
        pred = float(np.mean(np.sum(G * dh, axis=1)))
        lv = live["pass1_scale1"][f"plain_blk{l}"]["held_out_fv"]
        rec = {"passthrough_first_order_pred_gain_heldout": pred, "live_gain_heldout": lv["gain_mean"], "live_over_first_order_pred": lv["gain_mean"] / pred,
               "pred_dh_norm_mean": float(np.linalg.norm(dh, axis=1).mean()), "live_eta_mean_scale1": lv["eta_mean"],
               "live_cost_eta_over_gain_sq_scale1": lv["cost_mean_eta_over_mean_gain_sq"]}
        if l == 31:
            hb = Z["s_l_out"][hidx, 3, :].astype(np.float64)
            ex = [float(C[j] @ normed(hb[j] + dh[j]) - C[j] @ normed(hb[j])) for j in range(len(hidx))]
            rec["exact_float64_gain_heldout"] = float(np.mean(ex))
            rec["live_over_exact_float64"] = lv["gain_mean"] / float(np.mean(ex))
            # anchors, float64, at the live allocation scale for only31
            sc = live["pass2"]["only31"]["scales"]["plain_blk31"]
            xa = Z["s_key"][anchors, 3, :].astype(np.float64)
            sa = silu(Z["s_gate_pre"][anchors, 3, :].astype(np.float64))
            ha = Z["s_l_out"][anchors, 3, :].astype(np.float64)
            dha = sc * ((sa * (xa @ A.T @ B.T)) @ Wd.T)
            rec["anchors_float64_dh_over_R0_at_only31_scale"] = float((np.linalg.norm(dha, axis=1) / np.linalg.norm(ha, axis=1)).mean())
            rec["anchors_live_dh_over_R0_at_only31_scale"] = live["pass2"]["only31"]["anchor"]["dh_over_R0_mean"]
            rec["only31_scale"] = sc
            rec["anchors_float64_dh_norm_mean"] = float(np.linalg.norm(dha, axis=1).mean())
        out["by_layer"][str(l)] = rec
    c1 = [out["by_layer"][str(l)]["live_cost_eta_over_gain_sq_scale1"] for l in LAYERS]
    out["uniform_cost_from_scale1_costs_added_incoherently"] = float(sum(c1) / 16.0)
    out["uniform_over_only31_from_scale1_costs"] = float(sum(c1) / 16.0 / c1[3])

    # independence of the held-out split
    rows = {r["id"]: r for r in (json.loads(x) for x in open(COMPILE_JSONL, encoding="utf-8") if x.strip())}
    tr_ids = sorted({st[i]["id"] for i in tidx}); ho_ids = sorted({st[i]["id"] for i in hidx})
    sims = []
    for hid in ho_ids:
        sims.append(max(difflib.SequenceMatcher(None, rows[hid]["prompt"], rows[t]["prompt"], autojunk=False).ratio() for t in tr_ids))
    tr_pairs = {(tuple(st[i]["pos_toks"]), tuple(sorted(st[i]["neg_toks"]))) for i in tidx}
    reuse = np.mean([(tuple(st[i]["pos_toks"]), tuple(sorted(st[i]["neg_toks"]))) in tr_pairs for i in hidx])
    fam = lambda i: st[i]["family"]
    out["held_out_independence"] = {
        "prompt_id_overlap": len(set(tr_ids) & set(ho_ids)), "n_train_prompts": len(tr_ids), "n_held_out_prompts": len(ho_ids),
        "families_train": sorted({fam(i) for i in tidx}), "families_held_out": sorted({fam(i) for i in hidx}),
        "nearest_train_prompt_similarity_median": float(np.median(sims)), "nearest_train_prompt_similarity_frac_above_0.8": float(np.mean(np.array(sims) > 0.8)),
        "held_out_states_reusing_a_training_pos_neg_token_pair_frac": float(reuse),
        "split_rule": "every third prompt id (sorted) held out",
        "anchors_note": "the 24 anchors evaluated are the same anchors used as preservation keys at compile time",
    }
    top1 = Z["s_top_ids"][:, 0]
    fvs = [i for i in fv if st[i]["pos_toks"]]
    marg = [min(st[i]["logit_pos"]) - max(st[i]["logit_neg"]) for i in fvs]
    out["base_model_on_formal_verification_states"] = {
        "n": len(fvs), "positive_token_is_top1_frac": float(np.mean([int(top1[i]) in st[i]["pos_toks"] for i in fvs])),
        "positive_beats_every_listed_negative_frac": float(np.mean(np.array(marg) > 0)), "median_margin_vs_listed_negatives": float(np.median(marg))}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "by_layer"}, indent=1))
    for l in LAYERS:
        print(l, json.dumps(out["by_layer"][str(l)]))


if __name__ == "__main__":
    main()
