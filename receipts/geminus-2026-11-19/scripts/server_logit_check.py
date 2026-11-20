"""Server-level logit check: does llama-server's PER-REQUEST lora path deliver the adapters to every slot?

The library probe (probe.py) set adapters context-wide. This script exercises the path the anomaly was
reported on: POST /completion with a per-request "lora" list, 8 slots, concurrent requests. It reads
per-token log-probabilities (n_probs) instead of greedy text, at the second generated token, where
the base model prefers <think> to ``` by about 0.94 logits and the three adapters shift that by +0.84.

margin = logprob(```) - logprob(<think>) at the fork token (a logit difference, since both come from the
same softmax). Decision rule, fixed before the run: a slot counts as "adapters applied" when its margin is
closer to the triple-LoRA cluster than to the W0 cluster, i.e. margin > the midpoint of the two
single-request reference margins measured in this same server session.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLAMA_SERVER, MODEL, LORA_DIR, OUT_JSON, SERVER_LOG = sys.argv[1:6]
PROMPT = (
    "You are a formal verification assistant. Explain, step by step, how to prove in Lean 4 that "
    "the sum of the first n odd natural numbers equals n squared, including the induction step, "
    "the key rewriting lemmas, and how `simp` and `ring` close the remaining goals."
)
LORAS = ["corpus.f16.gguf", "weft-eido.f16.gguf", "formal-verification.f16.gguf"]
THREE = [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.0}, {"id": 2, "scale": 1.0}]
W0 = [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 0.0}]
N_PROBS = 40


def http(base, method, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def completion(base, lora, slot):
    t0 = time.time()
    r = http(base, "POST", "/completion", {"prompt": PROMPT, "lora": lora, "temperature": 0.0, "top_p": 1.0, "seed": 42, "n_predict": 3,
                                          "id_slot": slot, "cache_prompt": False, "n_probs": N_PROBS, "post_sampling_probs": False})
    probs = r.get("completion_probabilities") or []
    rec = {"slot_requested": slot, "slot_served": r.get("id_slot"), "content": r.get("content"), "sent_at": round(t0, 3), "dt": round(time.time() - t0, 3),
           "content_sha256": hashlib.sha256((r.get("content") or "").encode()).hexdigest()[:16], "n_prob_tokens": len(probs),
           "lora_echo": (r.get("generation_settings") or {}).get("lora")}
    for ti, p in enumerate(probs[:2]):
        tops = p.get("top_logprobs") or p.get("top_probs") or []
        d = {}
        for c in tops:
            key = c.get("token")
            val = c.get("logprob")
            if val is None and c.get("prob") is not None:
                import math
                val = math.log(max(c["prob"], 1e-300))
            d[key] = val
        rec[f"tok{ti}"] = {"chosen": p.get("token"), "chosen_logprob": p.get("logprob"), "top": d}
    t1 = rec.get("tok1", {}).get("top", {})
    rec["fence_logprob"], rec["think_logprob"] = t1.get("```"), t1.get("<think>")
    rec["margin"] = (rec["fence_logprob"] - rec["think_logprob"]) if (rec["fence_logprob"] is not None and rec["think_logprob"] is not None) else None
    return rec


def main():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    base = f"http://127.0.0.1:{port}"
    cmd = [LLAMA_SERVER, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "-c", "65536", "-ngl", "99", "-np", "8",
           "-ctk", "f16", "-ctv", "f16", "--slots", "--no-cache-idle-slots", "--lora-scaled", ",".join(f"{n}:0.0" for n in LORAS), "--lora-init-without-apply"]
    logf = open(SERVER_LOG, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=LORA_DIR)
    out = {"schema": "geminus-server-logit-check/v1", "server_flags": [c for c in cmd[1:] if not (":\\" in c or c.startswith("/"))], "n_probs": N_PROBS, "errors": []}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early rc={proc.returncode}")
            try:
                if http(base, "GET", "/health", timeout=2).get("status") == "ok":
                    break
            except Exception:
                pass
            if time.time() - t0 > 300:
                raise RuntimeError("timeout waiting for /health")
            time.sleep(0.5)
        out["build_info"] = http(base, "GET", "/props", timeout=10).get("build_info")
        out["lora_adapters_global"] = [{"id": x.get("id"), "scale": x.get("scale")} for x in http(base, "GET", "/lora-adapters")]
        ref_w0 = completion(base, W0, 0)
        ref_tr = completion(base, THREE, 0)
        out["reference_single"] = {"w0": ref_w0, "triple": ref_tr}
        if ref_w0["margin"] is None or ref_tr["margin"] is None:
            raise RuntimeError(f"fork tokens not both within top-{N_PROBS}: w0 {ref_w0.get('tok1')}, triple {ref_tr.get('tok1')}")
        mid = 0.5 * (ref_w0["margin"] + ref_tr["margin"])
        out["decision_threshold_margin_midpoint"] = mid
        rounds = []

        def run_round(name, loras_by_slot, stagger=None):
            def one(sl):
                if stagger and sl in stagger:
                    time.sleep(stagger[sl])
                return completion(base, loras_by_slot[sl], sl)
            with cf.ThreadPoolExecutor(8) as ex:
                res = list(ex.map(one, sorted(loras_by_slot)))
            for r_ in res:
                r_["requested"] = "triple" if loras_by_slot[r_["slot_requested"]] is THREE else "w0"
                r_["classified"] = None if r_["margin"] is None else ("triple" if r_["margin"] > mid else "w0")
            rounds.append({"name": name, "results": res})

        for k in range(6):
            run_round(f"triple_x8_r{k}", {s_: THREE for s_ in range(8)})
        for k in range(3):
            run_round(f"w0_x8_r{k}", {s_: W0 for s_ in range(8)})
        for k in range(3):
            run_round(f"mixed_4w0_4triple_r{k}", {s_: (THREE if s_ % 2 else W0) for s_ in range(8)})
        for k in range(3):      # staggered arrivals, as seen in the server log of the strict re-run
            run_round(f"triple_x8_staggered_r{k}", {s_: THREE for s_ in range(8)}, stagger={1: 0.09, 2: 0.09, 3: 0.09, 4: 0.09, 5: 0.44, 6: 0.44, 7: 0.44})
        out["rounds"] = rounds
        # Which adapters can this margin see? Single-request references for each adapter alone (added after review round 2).
        singles = {}
        for name, sc in (("formal_verification_only", (0.0, 0.0, 1.0)), ("corpus_only", (1.0, 0.0, 0.0)), ("weft_eido_only", (0.0, 1.0, 0.0)),
                         ("corpus_and_weft_eido", (1.0, 1.0, 0.0)), ("w0_again", (0.0, 0.0, 0.0)), ("three_again", (1.0, 1.0, 1.0))):
            rr = completion(base, [{"id": i, "scale": sc[i]} for i in range(3)], 0)
            singles[name] = {"margin": rr["margin"], "second_token": (rr.get("tok1") or {}).get("chosen"), "lora_echo": rr["lora_echo"]}
        out["single_request_references_by_adapter_subset"] = singles
        flat = [r_ for rd in rounds for r_ in rd["results"]]
        req_tr = [r_ for r_ in flat if r_["requested"] == "triple"]
        req_w0 = [r_ for r_ in flat if r_["requested"] == "w0"]
        mt = [r_["margin"] for r_ in req_tr if r_["margin"] is not None]
        mw = [r_["margin"] for r_ in req_w0 if r_["margin"] is not None]
        out["summary"] = {
            "reference_margin_w0": ref_w0["margin"], "reference_margin_triple": ref_tr["margin"],
            "triple_requests": len(req_tr), "triple_requests_with_margin": len(mt), "triple_classified_triple": sum(1 for r_ in req_tr if r_["classified"] == "triple"),
            "triple_margin_min": min(mt) if mt else None, "triple_margin_max": max(mt) if mt else None,
            "w0_requests": len(req_w0), "w0_requests_with_margin": len(mw), "w0_classified_w0": sum(1 for r_ in req_w0 if r_["classified"] == "w0"),
            "w0_margin_min": min(mw) if mw else None, "w0_margin_max": max(mw) if mw else None,
            "gap_between_clusters": (min(mt) - max(mw)) if (mt and mw) else None,
            "triple_requests_whose_second_token_is_think": sum(1 for r_ in req_tr if (r_.get("tok1") or {}).get("chosen") == "<think>"),
            "slot_served_equals_requested": all(r_["slot_served"] == r_["slot_requested"] for r_ in flat),
            "distinct_triple_margins": len({round(x, 6) for x in mt}), "distinct_w0_margins": len({round(x, 6) for x in mw}),
            "lora_echo_matches_request": all((r_["lora_echo"] is None) or all(abs(e.get("scale", 0.0) - (1.0 if r_["requested"] == "triple" else 0.0)) < 1e-9 for e in r_["lora_echo"]) for r_ in flat),
        }
        mixed = [r_ for rd in rounds if rd["name"].startswith("mixed") for r_ in rd["results"]]
        out["summary"]["mixed_rounds_note"] = {
            "n": len(mixed), "distinct_margins": sorted({round(r_["margin"], 6) for r_ in mixed}),
            "equal_to_single_request_references": all(any(abs(r_["margin"] - ref) < 1e-9 for ref in (ref_w0["margin"], ref_tr["margin"])) for r_ in mixed)}
    except Exception as exc:
        out["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        for _ in range(30):
            if not subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip():
                break
            time.sleep(1)
    Path(OUT_JSON).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out.get("summary", out["errors"]), indent=1))
    if "reference_single" in out:
        print("ref w0 tok1 chosen:", out["reference_single"]["w0"].get("tok1", {}).get("chosen"), "| ref triple tok1 chosen:", out["reference_single"]["triple"].get("tok1", {}).get("chosen"))
    return 0 if not out["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
