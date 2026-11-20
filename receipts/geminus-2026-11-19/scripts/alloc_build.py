"""Compile one rank-8 W_up LoRA per layer (blk.28..31) with the mcproof P2-B recipe and plain
pulled-back targets, and write each as its own GGUF adapter. Loading all four and setting their
scales then realises any cross-layer allocation on the live model.

Per-layer target (pass-through pull-back of the final-norm contrast gradient g):
    du_l = s_l * (W_down^l^T g), normalised per state so ||W_down^l (s_l * du_l)|| = 0.35 ||ffn_out_l||,
    times output gain 1.65 and the 0.20 rel_step-0 weight (identical to the team recipe at blk.31).
Also writes blk.31 adapters for the preconditioned arms produced by compile_exp.py, if present.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

SCRIPTS, WEFT_SRC, MODEL, HARVEST_PREFIX, COMPILE_FACTORS_NPZ, OUT_DIR = sys.argv[1:7]
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, WEFT_SRC)
from ggufw import Weights  # noqa: E402
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector  # noqa: E402
from weft.gguf_lora import write_gguf_lora  # noqa: E402

LAYERS = [28, 29, 30, 31]
EPS = 1e-6
REL_SCALE, OUT_GAIN, STEP0_W = 0.35, 1.65, 0.20


def silu(z):
    return z / (1.0 + np.exp(-z))


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    meta = json.load(open(HARVEST_PREFIX + ".meta.json", encoding="utf-8"))
    Z = np.load(HARVEST_PREFIX + ".npz")
    st = meta["states"]
    d, dff = meta["d_model"], meta["d_ff"]
    W = Weights(MODEL)
    g_out = W.get("output_norm.weight")
    sqd = np.sqrt(d)
    fv = [i for i, s in enumerate(st) if s["group"] == "fv"]
    ids = sorted({st[i]["id"] for i in fv})
    held_ids = set(ids[2::3])
    train = [i for i in fv if st[i]["id"] not in held_ids]
    anchors = [i for i, s in enumerate(st) if s["group"] == "anchor"]
    need = sorted({int(t) for i in train for t in st[i]["pos_toks"] + st[i]["neg_toks"]})
    trow = {t: k for k, t in enumerate(need)}
    E = W.get_rows("token_embd.weight", need)

    # final-norm contrast gradients for the training states
    G = np.zeros((len(train), d))
    for j, i in enumerate(train):
        h = Z["s_l_out"][i, 3].astype(np.float64)
        R0sq = h @ h + d * EPS
        c = E[[trow[t] for t in st[i]["pos_toks"]]].mean(axis=0) - E[[trow[t] for t in st[i]["neg_toks"]]].mean(axis=0)
        u = g_out * c
        G[j] = (sqd / np.sqrt(R0sq)) * (u - (u @ h) / R0sq * h)
    w_step = np.array([STEP0_W if st[i]["rel_step"] == 0 else 1.0 for i in train])

    manifest = {"schema": "geminus-alloc-adapters/v1", "held_out_ids": sorted(held_ids), "n_train": len(train), "adapters": {}}
    cfg = BetaEditConfig(d_in=d, d_out=dff, max_rank=8, max_slots=8, lambda_leakage=250.0, lambda_ridge=4.0,
                         nullspace_threshold=1e9, threshold_mode="absolute", radix_scale_exp2=24, quant_dtype="f16")
    zero_w0 = np.zeros((dff, d))
    for li, l in enumerate(LAYERS):
        Wd = W.get(f"blk.{l}.ffn_down.weight")
        S_ = silu(Z["s_gate_pre"][train, li, :].astype(np.float64))        # (n, d_ff)
        K = Z["s_key"][train, li, :].astype(np.float64).T                  # (d, n)
        A_ = S_ * (G @ Wd)                                                 # plain pull-back (n, d_ff)
        induced = (S_ * A_) @ Wd.T                                         # (n, d)
        fo = np.linalg.norm(Z["s_ffn_out"][train, li, :].astype(np.float64), axis=1)
        scale = np.maximum(0.05, fo) * REL_SCALE / np.linalg.norm(induced, axis=1)
        U = (A_ * (scale * OUT_GAIN * w_step)[:, None]).T                  # (d_ff, n)
        pt_gain = np.sum(G * (induced * scale[:, None]), axis=1)           # pass-through first-order gain of the per-state target
        K0 = Z["n_key"][:, li, :].astype(np.float64).T
        K0 = K0 / np.sqrt(K0.shape[1])
        anc = np.stack([Z["s_key"][i, li].astype(np.float64) for i in anchors], axis=1)
        proj = BetaEditNullspaceProjector(zero_w0, cfg)
        proj.initialize_anchor_covariance(K0)
        patch = proj.stage_edit(K, U, is_residual=True, rank=8, preservation_keys=anc, preservation_weight=2.0, slot_name=f"plain.blk{l}")
        Bm, Am = np.asarray(patch.factor_b, np.float64), np.asarray(patch.factor_a, np.float64)
        path, sha = write_gguf_lora(f"{OUT_DIR}/plain_blk{l}.gguf", architecture="qwen35", lora_alpha=8.0,
                                    layer_factors={f"blk.{l}.ffn_up.weight": (Bm, Am)}, extra_metadata={"geminus.arm": "plain", "geminus.layer": l})
        # fitted pass-through gain on the training states (what the compiled factors deliver under P_l = I)
        fit = (S_ * (K.T @ Am.T @ Bm.T)) @ Wd.T
        manifest["adapters"][f"plain_blk{l}"] = {
            "file": f"plain_blk{l}.gguf", "sha256": sha, "layer": l, "arm": "plain", "rank": int(Bm.shape[1]),
            "fro_norm": float(np.sqrt(np.sum((Bm.T @ Bm) * (Am @ Am.T)))), "truncation_rel_error": float(patch.truncation_rel_error),
            "target_passthrough_gain_mean": float(pt_gain.mean()), "fitted_passthrough_gain_mean_train": float(np.mean(np.sum(G * fit, axis=1))),
            "fitted_dh_norm_mean_train": float(np.linalg.norm(fit, axis=1).mean()), "ffn_out_norm_mean_train": float(fo.mean())}
        print(f"blk.{l}: fro {manifest['adapters'][f'plain_blk{l}']['fro_norm']:.4f} fitted pass-through gain {manifest['adapters'][f'plain_blk{l}']['fitted_passthrough_gain_mean_train']:.4f} "
              f"|dh| {manifest['adapters'][f'plain_blk{l}']['fitted_dh_norm_mean_train']:.4f}  {time.time()-t0:.0f}s", flush=True)

    if os.path.exists(COMPILE_FACTORS_NPZ):
        F = np.load(COMPILE_FACTORS_NPZ)
        for arm in ("plain", "lam10", "lam1", "lam0.1", "minnorm"):
            if f"{arm}_B" not in F:
                continue
            Bm, Am = F[f"{arm}_B"], F[f"{arm}_A"]
            path, sha = write_gguf_lora(f"{OUT_DIR}/c31_{arm}.gguf", architecture="qwen35", lora_alpha=8.0,
                                        layer_factors={"blk.31.ffn_up.weight": (Bm, Am)}, extra_metadata={"geminus.arm": arm, "geminus.layer": 31})
            manifest["adapters"][f"c31_{arm}"] = {"file": f"c31_{arm}.gguf", "sha256": sha, "layer": 31, "arm": arm, "rank": int(Bm.shape[1]),
                                                  "fro_norm": float(np.sqrt(np.sum((Bm.T @ Bm) * (Am @ Am.T))))}
            print(f"wrote c31_{arm}: fro {manifest['adapters'][f'c31_{arm}']['fro_norm']:.4f}", flush=True)
    else:
        print("compile_exp factors not present; skipping preconditioned blk.31 adapters", flush=True)
    json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w", encoding="utf-8"), indent=1)
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
