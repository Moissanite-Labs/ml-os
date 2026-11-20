"""Rollback without a cache flush (v2): does switching an adapter off restore base behaviour mid-sequence?

For each sequence and each single-layer adapter (blk.28..31, scale 1):
  1. decode the prefix with the adapter ON (fresh memory), REQUESTING AN OUTPUT ROW FOR EVERY PREFIX TOKEN;
  2. switch the adapter OFF and decode the remaining k tokens WITHOUT clearing memory;
  3. compare the logits at the last token with a clean run (no adapter at any point, same split and shapes).

Why step 1 requests outputs: this model crops the last layer to output rows before its FFN (qwen35.cpp),
so without them a blk.31 adapter would never execute on the prefix and an exact match would be vacuous
(v1 of this script had that flaw; a reviewer caught it). The script also records how much each adapter
changed the logits at the END OF THE PREFIX while it was on, as proof that it executed there.

Control: after a contaminated run, clearing memory and decoding the whole sequence must match exactly.
Library-level only. llama-server clears a slot's cached tokens when its LoRA configuration changes.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

SCRIPTS, BIN, MODEL, DATASETS, META, ADAPTER_DIR, OUT_JSON = sys.argv[1:8]
sys.path.insert(0, SCRIPTS)
from jobs import build_jobs  # noqa: E402
from llama_ctypes import Llama  # noqa: E402
from render import ChatTemplate  # noqa: E402

KS = [1, 4, 16]
LAYERS = [28, 29, 30, 31]


def log_softmax(v):
    v = v.astype(np.float64)
    m = v.max()
    return v - m - np.log(np.sum(np.exp(v - m)))


def main():
    t0 = time.time()
    man = json.load(open(ADAPTER_DIR + "/manifest.json", encoding="utf-8"))
    held = set(man["held_out_ids"])
    tpl = ChatTemplate(MODEL)
    ll = Llama(BIN)
    ll.check_layout()
    ll.load(MODEL, n_gpu_layers=99, n_ctx=2048, n_batch=2048)
    idx = {l: ll.load_lora(f"{ADAPTER_DIR}/plain_blk{l}.gguf") for l in LAYERS}
    jobs = build_jobs(ll, tpl, DATASETS, META)
    fvj = [j for j in jobs if j["group"] == "fv" and j["id"] in held][:12]
    seqs = [(j["id"], j["tokens"][: max(s["pos"] for s in j["states"]) + 1]) for j in fvj] + \
           [(j["id"], j["tokens"]) for j in jobs if j["group"] in ("corpus", "eido")]
    seqs = [(i, t) for i, t in seqs if 16 + 8 < len(t) <= 400]
    print(f"{len(seqs)} sequences", flush=True)

    def split_run(toks, k, on_layer):
        n = len(toks)
        ll.set_loras({idx[on_layer]: 1.0} if on_layer is not None else {})
        pre = ll.decode_multi([(0, toks[: n - k], 0, "all")], clear=True)[0]      # outputs on every prefix token
        ll.set_loras({})
        out = ll.decode_multi([(0, toks[n - k:], n - k, True)], clear=False)[0]
        return pre, out

    rows = []
    for sid, toks in seqs:
        for k in KS:
            pre_ref, ref = split_run(toks, k, None)
            _, ref2 = split_run(toks, k, None)
            lp_ref = log_softmax(ref)
            rec = {"id": sid, "n_tokens": len(toks), "k_tokens_after_rollback": k, "clean_repeat_max_abs_diff": float(np.max(np.abs(ref2 - ref)))}
            for l in LAYERS:
                pre, lg = split_run(toks, k, l)
                rec[f"blk{l}"] = {"adapter_effect_at_prefix_end_max_abs_logit_diff": float(np.max(np.abs(pre - pre_ref))),
                                  "max_abs_logit_diff": float(np.max(np.abs(lg - ref))),
                                  "kl_clean_to_rolled_back": float(np.sum(np.exp(lp_ref) * (lp_ref - log_softmax(lg)))),
                                  "top1_changed": bool(int(np.argmax(lg)) != int(np.argmax(ref)))}
            ll.set_loras({idx[28]: 1.0})
            ll.decode_multi([(0, toks[: len(toks) - k], 0, "all")], clear=True)
            ll.set_loras({})
            whole = ll.decode_multi([(0, toks, 0, True)], clear=True)[0]
            clean_whole = ll.decode_multi([(0, toks, 0, True)], clear=True)[0]
            rec["flush_control_max_abs_diff"] = float(np.max(np.abs(whole - clean_whole)))
            rows.append(rec)
    ll.close()
    summary = {}
    for k in KS:
        rk = [r for r in rows if r["k_tokens_after_rollback"] == k]
        summary[str(k)] = {"n_sequences": len(rk), "clean_repeat_max_abs_diff": max(r["clean_repeat_max_abs_diff"] for r in rk),
                           "flush_control_max_abs_diff": max(r["flush_control_max_abs_diff"] for r in rk)}
        for l in LAYERS:
            summary[str(k)][f"blk{l}"] = {
                "adapter_effect_at_prefix_end_median": float(np.median([r[f"blk{l}"]["adapter_effect_at_prefix_end_max_abs_logit_diff"] for r in rk])),
                "adapter_effect_at_prefix_end_min": min(r[f"blk{l}"]["adapter_effect_at_prefix_end_max_abs_logit_diff"] for r in rk),
                "max_abs_logit_diff_max": max(r[f"blk{l}"]["max_abs_logit_diff"] for r in rk),
                "max_abs_logit_diff_median": float(np.median([r[f"blk{l}"]["max_abs_logit_diff"] for r in rk])),
                "kl_mean": float(np.mean([r[f"blk{l}"]["kl_clean_to_rolled_back"] for r in rk])),
                "kl_max": max(r[f"blk{l}"]["kl_clean_to_rolled_back"] for r in rk),
                "top1_changed_frac": float(np.mean([r[f"blk{l}"]["top1_changed"] for r in rk])),
                "exact_match_frac": float(np.mean([r[f"blk{l}"]["max_abs_logit_diff"] == 0.0 for r in rk]))}
    out = {"schema": "geminus-rollback-probe/v2", "prefix_outputs": "all prefix tokens",
           "adapters": {f"blk{l}": man["adapters"][f"plain_blk{l}"]["sha256"] for l in LAYERS},
           "summary_by_k": summary, "rows": rows, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    for k, s in summary.items():
        print(f"k={k:>2s} tokens after rollback (n={s['n_sequences']}): clean repeat {s['clean_repeat_max_abs_diff']}, flush control {s['flush_control_max_abs_diff']}")
        for l in LAYERS:
            b = s[f"blk{l}"]
            print(f"    adapter at blk.{l}: effect at prefix end while ON median {b['adapter_effect_at_prefix_end_median']:.3f} (min {b['adapter_effect_at_prefix_end_min']:.3f}) | after rollback max|dlogit| median {b['max_abs_logit_diff_median']:.4f} max {b['max_abs_logit_diff_max']:.4f} | KL mean {b['kl_mean']:.2e} | exact match {b['exact_match_frac']:.2f}")
    print("elapsed", out["elapsed_s"])


if __name__ == "__main__":
    main()
