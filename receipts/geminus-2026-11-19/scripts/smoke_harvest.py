"""Smoke test: layout check, tensor-name discovery, and one numeric cross-check against the
team's PyTorch-bf16 harvest metadata (layer-31 ffn_out norm at a recorded decision state)."""
import sys, json, time
import numpy as np
SCRIPTS, BIN, MODEL, DATASETS, META = sys.argv[1:6]
sys.path.insert(0, SCRIPTS)
from llama_ctypes import Llama
from render import ChatTemplate

tpl = ChatTemplate(MODEL)
print("template chars:", len(tpl.source))
demo = tpl.render_user("PROMPT_BODY")
print("rendered demo:", repr(demo))

ll = Llama(BIN)
print("layout:", json.dumps(ll.check_layout()))
t0 = time.time()
ll.load(MODEL, n_gpu_layers=99, n_ctx=2048, n_batch=2048)
print(f"loaded in {time.time()-t0:.1f}s  n_embd={ll.n_embd} n_layer={ll.n_layer} n_vocab={ll.n_vocab}"); print("gpu_offload:", json.dumps(ll.gpu_offload))

rows = [json.loads(l) for l in open(DATASETS + "/compile.jsonl", encoding="utf-8") if l.strip()]
meta = json.load(open(META, encoding="utf-8"))
rec = next(r for r in rows if r["id"] == "PUB-A-C01")
steps = [s for s in meta["step_metadata"] if s["id"] == "PUB-A-C01"]
prompt_ids = ll.tokenize(tpl.render_user(rec["prompt"]))
pos_ids = ll.tokenize(rec["positive_completion"])
neg_ids = ll.tokenize(rec["negative_completion"])
print("prompt tokens:", len(prompt_ids), "tail pieces:", [ll.piece(t) for t in prompt_ids[-8:]])
print("pos first tok:", pos_ids[0], repr(ll.piece(pos_ids[0])), "| meta pos_toks step0:", steps[0]["pos_toks"])
print("neg first tok:", neg_ids[0], repr(ll.piece(neg_ids[0])), "| meta neg_toks step0:", steps[0]["neg_toks"])
max_step = max(s["rel_step"] for s in steps)
full = prompt_ids + pos_ids[: max_step + 1]
base = len(prompt_ids) - 1
pos_list = sorted(base + s["rel_step"] for s in steps)
print("full len", len(full), "decision positions", pos_list)

L = 31
out_rows = list(range(len(pos_list)))          # layer-31 FFN tensors are cropped to output rows
want = {f"ffn_out-{L}": out_rows, f"l_out-{L}": out_rows, f"attn_residual-{L}": out_rows,
        f"attn_post_norm-{L}": out_rows, f"ffn_gate-{L}": out_rows, f"ffn_up-{L}": out_rows, f"ffn_swiglu-{L}": out_rows,
        "ffn_out-28": pos_list, "l_out-28": pos_list, "attn_residual-28": pos_list, "l_out-27": pos_list,
        "result_norm": out_rows, "__logits__": True}
t0 = time.time()
res = ll.decode(full, want, pos_list, record_names=True)
print(f"decode {time.time()-t0:.2f}s")
names = ll._seen_names
uniq = []
for n in names:
    base_n = n.rsplit("-", 1)[0] if n.rsplit("-", 1)[-1].isdigit() else n
    if base_n not in uniq:
        uniq.append(base_n)
print("distinct node name stems seen (%d nodes):" % len(names), uniq)
print("shapes:", {k: ll._shapes[k] for k in ll._shapes})
for j, s in enumerate(sorted(steps, key=lambda s: s["rel_step"])):
    mine = float(np.linalg.norm(res[f"ffn_out-{L}"][j].astype(np.float64)))
    print(f"rel_step {s['rel_step']:4d}: ||ffn_out-31|| mine={mine:.4f}  mac_bf16={s['ffn_out_norm']:.4f}  ratio={mine/s['ffn_out_norm']:.4f}"
          f"  | R(l_out-31)={np.linalg.norm(res['l_out-31'][j]):.3f} R(attn_residual-31)={np.linalg.norm(res['attn_residual-31'][j]):.3f}"
          f" R(l_out-28)={np.linalg.norm(res['l_out-28'][j]):.3f}")
lg = res["__logits__"]
for j in range(len(pos_list)):
    top = np.argsort(-lg[j])[:4]
    print("pos", pos_list[j], "top4:", [(int(t), ll.piece(int(t)), round(float(lg[j][t]), 2)) for t in top],
          "| next actual:", repr(ll.piece(full[pos_list[j] + 1])) if pos_list[j] + 1 < len(full) else "(end)")
ll.close()

# ---- internal consistency: reproduce the runtime's FFN from dequantised GGUF weights (float64) ----
sys.path.insert(0, SCRIPTS)
from ggufw import Weights
W = Weights(MODEL)
Wd = W.get("blk.31.ffn_down.weight"); Wu = W.get("blk.31.ffn_up.weight"); Wg = W.get("blk.31.ffn_gate.weight")
gn = W.get("blk.31.post_attention_norm.weight"); go = W.get("output_norm.weight")
x = res["attn_post_norm-31"].astype(np.float64); zg = res["ffn_gate-31"].astype(np.float64)
zu = res["ffn_up-31"].astype(np.float64); sw = res["ffn_swiglu-31"].astype(np.float64)
fo = res["ffn_out-31"].astype(np.float64); hm = res["attn_residual-31"].astype(np.float64)
lo = res["l_out-31"].astype(np.float64); rn = res["result_norm"].astype(np.float64)
silu = lambda z: z / (1.0 + np.exp(-z))
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
d = x.shape[1]; eps = 1e-6
rms = lambda h, g: g * h / np.sqrt(np.mean(h * h, axis=1, keepdims=True) + eps)
print("relerr  W_up x  vs ffn_up     :", rel(x @ Wu.T, zu))
print("relerr  W_gate x vs ffn_gate  :", rel(x @ Wg.T, zg))
print("relerr  silu(gate)*up vs swiglu:", rel(silu(zg) * zu, sw))
print("relerr  W_down swiglu vs ffn_out:", rel(sw @ Wd.T, fo))
print("relerr  rmsnorm(attn_residual)*g vs attn_post_norm:", rel(rms(hm, gn), x))
print("relerr  attn_residual+ffn_out vs l_out:", rel(hm + fo, lo))
print("relerr  rmsnorm(l_out)*g_out vs result_norm:", rel(rms(lo, go), rn))
E = W.get("token_embd.weight")
print("token_embd shape", E.shape, "| relerr E result_norm vs logits (tied head?):", rel(rn @ E.T, lg.astype(np.float64)))
