"""Weight-only candidate statistics for the M1 'gamma_l' / 'rho_l' definitions (identification test)."""
import sys, json, math
import numpy as np
sys.path.insert(0, sys.argv[2])
from ggufw import Weights

W = Weights(sys.argv[1])
layers = [28, 29, 30, 31]
mac = {"gammas": [6.805068, 6.186092, 4.709198, 2.073155],
       "rhos": [17.24481, 26.633308, 21.047025, 37.780445],
       "r_norms": [62.740422, 65.498958, 64.250502, 58.851399]}
g_out = W.get("output_norm.weight")
d = g_out.shape[0]
rows = {}
def add(k, v): rows.setdefault(k, []).append(float(v))

for l in layers:
    Wd = W.get(f"blk.{l}.ffn_down.weight")     # (d, d_ff)
    Wu = W.get(f"blk.{l}.ffn_up.weight")       # (d_ff, d)
    Wg = W.get(f"blk.{l}.ffn_gate.weight")     # (d_ff, d)
    gn = W.get(f"blk.{l}.post_attention_norm.weight")
    if not (Wd.shape == (d, Wu.shape[0]) and Wu.shape == Wg.shape == (Wd.shape[1], d)):
        raise AssertionError(f"FFN weight shape mismatch: {(Wd.shape, Wu.shape)}")
    dff = Wd.shape[1]
    sd = np.linalg.svd(Wd, compute_uv=False)
    su = np.linalg.svd(Wu, compute_uv=False)
    sg = np.linalg.svd(Wg, compute_uv=False)
    fro = math.sqrt(float(np.sum(Wd * Wd)))
    col = np.linalg.norm(Wd, axis=0); row = np.linalg.norm(Wd, axis=1)
    add("down.sigma_max", sd[0]); add("down.sigma_2", sd[1]); add("down.sigma_min", sd[-1])
    add("down.fro", fro); add("down.fro/sqrt(d)", fro / math.sqrt(d)); add("down.fro/sqrt(dff)", fro / math.sqrt(dff))
    add("down.rms_sigma", math.sqrt(float(np.mean(sd * sd))))
    add("down.mean_sigma", float(np.mean(sd))); add("down.median_sigma", float(np.median(sd)))
    add("down.stable_rank", fro * fro / (sd[0] ** 2))
    add("down.col_norm_mean", col.mean()); add("down.col_norm_max", col.max()); add("down.col_norm_rms", math.sqrt(float(np.mean(col * col))))
    add("down.row_norm_mean", row.mean()); add("down.row_norm_max", row.max())
    sdo = np.linalg.svd(g_out[:, None] * Wd, compute_uv=False)
    add("gout*down.sigma_max", sdo[0]); add("gout*down.fro/sqrt(d)", math.sqrt(float(np.sum(sdo * sdo))) / math.sqrt(d))
    add("up.sigma_max", su[0]); add("up.fro", math.sqrt(float(np.sum(su * su)))); add("up.fro/sqrt(dff)", math.sqrt(float(np.sum(su * su)) / dff))
    add("gate.sigma_max", sg[0]); add("gate.fro", math.sqrt(float(np.sum(sg * sg))))
    add("ffn_norm_gain.rms", math.sqrt(float(np.mean(gn * gn)))); add("ffn_norm_gain.l2", float(np.linalg.norm(gn)))
    add("down.sigma_max*up.sigma_max", sd[0] * su[0])
    add("(sigma_max/rms_sigma)^2 down", (sd[0] ** 2) / float(np.mean(sd * sd)))
    add("down.sigma_max^2", sd[0] ** 2)
    add("d/stable_rank", d / (fro * fro / (sd[0] ** 2)))
add("out_norm_gain.rms", math.sqrt(float(np.mean(g_out * g_out)))); add("out_norm_gain.l2", float(np.linalg.norm(g_out)))

def rel(a, b): return max(abs(x - y) / abs(y) for x, y in zip(a, b))
print(f"{'statistic':34s} " + " ".join(f"{'blk.'+str(l):>12s}" for l in layers) + "   relerr_vs_gamma relerr_vs_rho")
for k, v in rows.items():
    if len(v) != 4:
        print(f"{k:34s} {v[0]:12.6f}"); continue
    print(f"{k:34s} " + " ".join(f"{x:12.6f}" for x in v) + f"   {rel(v, mac['gammas']):10.4f} {rel(v, mac['rhos']):10.4f}")
json.dump({"layers": layers, "stats": rows, "mac_reference": mac}, open(sys.argv[3], "w"), indent=1)
