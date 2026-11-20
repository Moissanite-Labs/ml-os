"""Does 8-bit activation quantisation explain the gap between the runtime's quantised-weight matmuls
and exact float64 arithmetic on the same dequantised weights?

For each FFN matmul of blk.28 and blk.31, reconstruct the runtime output from the harvested input in
three ways: exact float64; input rounded to float16; input quantised per block of 32 to int8 with
scale max|x|/127 (the ggml Q8 activation format). Report relative L2 error against the runtime tensor.
"""
from __future__ import annotations

import json
import sys

import numpy as np

SCRIPTS, MODEL, HARVEST_PREFIX, OUT_JSON = sys.argv[1:5]
sys.path.insert(0, SCRIPTS)
from ggufw import Weights  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def q8_blocks(x, block=32):
    y = np.empty_like(x)
    for b0 in range(0, x.shape[-1], block):
        blk = x[..., b0:b0 + block]
        dsc = np.max(np.abs(blk), axis=-1, keepdims=True) / 127.0
        inv = np.where(dsc > 0, 1.0 / np.where(dsc > 0, dsc, 1.0), 0.0)
        y[..., b0:b0 + block] = np.rint(blk * inv) * dsc
    return y


def f16(a):
    return a.astype(np.float16).astype(np.float64)


def main():
    Z = np.load(HARVEST_PREFIX + ".npz")
    W = Weights(MODEL)
    out = {"schema": "geminus-activation-quantisation-check/v1", "n_states": int(Z["s_key"].shape[0]), "layers": {}}
    for li, l in ((0, 28), (3, 31)):
        Wu, Wg, Wd = W.get(f"blk.{l}.ffn_up.weight"), W.get(f"blk.{l}.ffn_gate.weight"), W.get(f"blk.{l}.ffn_down.weight")
        x = Z["s_key"][:, li, :].astype(np.float64)
        up = Z["s_up"][:, li, :].astype(np.float64)
        gp = Z["s_gate_pre"][:, li, :].astype(np.float64)
        fo = Z["s_ffn_out"][:, li, :].astype(np.float64)
        sw = (gp / (1.0 + np.exp(-gp))) * up
        xq, swq = q8_blocks(x), q8_blocks(sw)
        rec = {
            "ffn_up": {"exact_f64": rel(x @ Wu.T, up), "f16_input": rel(f16(x) @ Wu.T, up), "q8_input": rel(xq @ Wu.T, up)},
            "ffn_gate": {"exact_f64": rel(x @ Wg.T, gp), "f16_input": rel(f16(x) @ Wg.T, gp), "q8_input": rel(xq @ Wg.T, gp)},
            "ffn_down": {"exact_f64": rel(sw @ Wd.T, fo), "f16_input": rel(f16(sw) @ Wd.T, fo), "q8_input": rel(swq @ Wd.T, fo)},
            "q8_input_error_itself": {"key": rel(xq, x), "gated_up": rel(swq, sw)},
            "ffn_out_norm_mean": float(np.linalg.norm(fo, axis=1).mean()),
        }
        for k in ("ffn_up", "ffn_gate", "ffn_down"):
            rec[k]["improvement_factor_q8_over_exact"] = rec[k]["exact_f64"] / rec[k]["q8_input"]
        # is block size 32 specific? (ggml QK8_1 = 32)
        rec["ffn_up_by_block_size"] = {str(b): rel(q8_blocks(x, b) @ Wu.T, up) for b in (16, 32, 64)}
        rec["ffn_down_by_block_size"] = {str(b): rel(q8_blocks(sw, b) @ Wd.T, fo) for b in (16, 32, 64)}
        rec["implied_abs_noise_in_ffn_out_norm"] = rec["ffn_down"]["exact_f64"] * rec["ffn_out_norm_mean"]
        out["layers"][str(l)] = rec
        print(f"blk.{l}: " + " | ".join(f"{k} exact {rec[k]['exact_f64']:.2e} f16 {rec[k]['f16_input']:.2e} q8 {rec[k]['q8_input']:.2e} (x{rec[k]['improvement_factor_q8_over_exact']:.0f})"
                                         for k in ("ffn_up", "ffn_gate", "ffn_down")))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
