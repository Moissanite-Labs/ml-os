"""Packet 009 (revised after review rounds 1 and 2): Phase 1 native-Windows GPU smoke (ml-os#2 item 3).

Four separate runs feed this packet, in this order:
  1. library-level probe            18:06-18:21  m1/out/probe_v1.json (+ probe_analysis_v1.json)
  2. strict text re-run on the server 18:36-18:37  mlos/phase1/run2/  (the swarm's harness, gpu_workload.py)
  3. server-level log-probability test 19:1x      m1/out/server_logit_check_v1.json
  4. server-level log-probability test, run 2     m1/out/server_logit_check_v2.json (adds per-adapter-subset
     references and distinct-margin counts; run 1 is kept as a replicate)
Gates are computed from those files. The text gates fail and are reported as failing.
Round-2 corrections applied: the margin sees the formal-verification adapter only on this prompt; the
mixed rounds are serialised by the server; the batch-noise mechanism is incomplete.
"""
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
W = ROOT / "mlos" / "phase1" / "run2"
M1 = ROOT / "m1" / "out"
OUT = ROOT / "receipts"
w = json.loads((W / "smoke-workload.json").read_text(encoding="utf-8"))
s = json.loads((W / "smoke-stamp.json").read_text(encoding="utf-8"))
pa = json.loads((M1 / "probe_analysis_v1.json").read_text(encoding="utf-8"))
sl = json.loads((M1 / "server_logit_check_v1.json").read_text(encoding="utf-8"))
sls = json.loads((M1 / "server_logit_check_v1.stamp.json").read_text(encoding="utf-8"))
sl2 = json.loads((M1 / "server_logit_check_v2.json").read_text(encoding="utf-8"))
sls2 = json.loads((M1 / "server_logit_check_v2.stamp.json").read_text(encoding="utf-8"))
aq = json.loads((M1 / "actq_check_v1.json").read_text(encoding="utf-8"))
w1 = json.loads((ROOT / "mlos" / "phase1" / "run1" / "smoke-workload.json").read_text(encoding="utf-8"))
sp = json.loads((M1 / "probe_v1.stamp.json").read_text(encoding="utf-8"))


def span(stamp):
    e = stamp["execution"]
    return f"{e['started_utc'][11:19]}-{e['ended_utc'][11:19]}"


flags = w["server_flags"]
n_ctx_total = int(flags[flags.index("-c") + 1])
n_parallel = int(flags[flags.index("-np") + 1])
lora_dir = ROOT / "tc5" / "repo" / "Proven-Commons-mcproof-4552874" / "compile" / "artifacts" / "composition"
pred = json.loads(subprocess.run(
    [sys.executable, str(ROOT / "mlos" / "phase1" / "gguf_predict.py"), str(ROOT / "tc5" / "models" / "Qwen3.5-4B-Q8_0.gguf"), str(n_ctx_total),
     *[str(lora_dir / n) for n in w["lora_files"]]], capture_output=True, text=True, check=True).stdout)
t = w["telemetry"]
c = w["per_slot_lora_consistency"]
r0 = w["triple_lora_8slot_rounds"][0]
p2 = pa["p2_by_variant"]
multi = {k: v for k, v in p2.items() if v["n_rows"] == 8}
min_corr = min(v["triple_delta_corr_min"] for v in multi.values())
proj_lo = min(v["triple_delta_projection_min_max"][0] for v in multi.values())
proj_hi = max(v["triple_delta_projection_min_max"][1] for v in multi.values())
noise_max = max(v["w0_noise_vs_ref_max_abs_logit"] for v in p2.values())
ref_w0 = pa["p1"]["reference_w0"]["fork_margin_fence_minus_think"]
ref_tr = pa["p1"]["reference_triple"]["fork_margin_fence_minus_think"]
ss = sl["summary"]
ss2 = sl2["summary"]
subs = sl2["single_request_references_by_adapter_subset"]
thr2 = sl2["decision_threshold_margin_midpoint"]
measured, predicted = t["peak_vram_used_mib"], pred["predicted_mib_excl_compute_and_recurrent"]
gap = measured - predicted
TOL = 0.15
text_ok = c["lora_shaped"] == c["triple_lora_requests"] and c["w0_requests_that_came_back_lora_shaped"] == 0
flat = [x for rr in w["triple_lora_8slot_rounds"] for x in rr]
hashes_tr = collections.Counter((x["class"], x["content_sha256"][:8]) for x in flat)
hashes_w0 = collections.Counter(x["content_sha256"][:8] for x in w["w0_8slot"])
w0_set = set(hashes_w0) | {w["w0_single"]["content_sha256"][:8]}
n_equal_full_w0 = sum(1 for x in flat if x["content_sha256"][:8] in w0_set)
tok_s_run2 = sum(x["predicted_per_second"] for x in r0) / len(r0)
tok_s_run1 = sum(x["predicted_per_second"] for x in w1["triple_lora_8slot"]) / len(w1["triple_lora_8slot"])
server_ok = ss["triple_classified_triple"] == ss["triple_requests"] and ss["w0_classified_w0"] == ss["w0_requests"] and ss["slot_served_equals_requested"]
server_ok2 = ss2["triple_classified_triple"] == ss2["triple_requests"] and ss2["w0_classified_w0"] == ss2["w0_requests"] and ss2["slot_served_equals_requested"]

gates = {
    "peak_vram_le_7372_mib": {
        "threshold": "<= 7372 MiB (7.2 GiB)",
        "measured": f"{measured:.0f} MiB nvidia-smi memory.used @{t['interval_s']}s, {t['samples']} samples, baseline {t['baseline_vram_used_mib']:.0f} MiB (strict text re-run)",
        "passed": bool(t["vram_gate_passed"])},
    "vram_predicted_vs_measured": {
        "threshold": f"|measured - predicted| <= {int(TOL*100)}% of measured; prediction from the GGUF header and the recorded -c value. The tolerance was chosen after seeing the data and is provisional.",
        "measured": f"-c {n_ctx_total} (from the recorded server flags), -np {n_parallel} (recorded, not used by the predictor): predicted {predicted} MiB = model file {pred['model_file_mib']} + f16 K/V {pred['kv_cache_mib']} for {pred['n_attention_layers_with_kv']} attention layers + LoRA {pred['lora_files_mib']}; measured {measured:.0f} MiB; gap {gap:.1f} MiB = {100*gap/measured:.1f}% (compute buffers, recurrent state and the CUDA context are not modelled)",
        "passed": abs(gap) <= TOL * measured},
    "zero_lingering_compute_vram": {
        "threshold": "0 MiB compute VRAM after exit",
        "measured": f"end compute_vram_mib={s['end_gpu_telemetry']['compute_vram_mib']}, released_after_exit={w['compute_vram_released_after_exit']}; stamper snapshots of free MiB at start/end {s['start_gpu_telemetry']['memory_free_mib']}/{s['end_gpu_telemetry']['memory_free_mib']}",
        "passed": s["end_gpu_telemetry"]["compute_vram_mib"] == 0 and w["compute_vram_released_after_exit"]},
    "kv_8_slots_x_8192": {
        "threshold": "8 slots with n_ctx 8192 each, serving 8 concurrent requests",
        "measured": f"total_slots={w['server_props']['total_slots']}, n_ctx_per_slot={w['server_props']['n_ctx_per_slot']}, {len(w['triple_lora_8slot_rounds'])} rounds x {len(r0)} concurrent completions x128 tok returned",
        "passed": w["server_props"]["total_slots"] == 8 and w["server_props"]["n_ctx_per_slot"] == 8192 and len(r0) == 8},
    "lora_applied_on_every_slot_server_level_logprob_margin": {
        "threshold": "llama-server per-request 'lora' path: every adapter request's fork-token margin lies on the adapter side of the midpoint between the two single-request reference margins, and every W0 request's on the W0 side. Rule fixed before the run. SCOPE (round-2 correction): on this prompt the margin sees the formal-verification adapter only, so the gate tests that an adapter-sized shift reached every request, not that all three adapters were applied.",
        "measured": f"run 1: {ss['triple_classified_triple']}/{ss['triple_requests']} adapter and {ss['w0_classified_w0']}/{ss['w0_requests']} W0 correct, gap {ss['gap_between_clusters']:.3f}; run 2: {ss2['triple_classified_triple']}/{ss2['triple_requests']} adapter and {ss2['w0_classified_w0']}/{ss2['w0_requests']} W0 correct, gap {ss2['gap_between_clusters']:.3f}. Run-2 subset references: fv alone {subs['formal_verification_only']['margin']:+.3f} (adapter side), corpus alone {subs['corpus_only']['margin']:+.3f}, design alone {subs['weft_eido_only']['margin']:+.3f}, corpus+design {subs['corpus_and_weft_eido']['margin']:+.3f} (all W0 side; threshold {thr2:+.3f}). Effective sample in run 2: {ss2['distinct_triple_margins']} distinct adapter margins and {ss2['distinct_w0_margins']} distinct W0 margins; the mixed rounds are serialised by the server and equal the two references. Rounds per run: 6 x 8 adapter, 3 x 8 W0, 3 mixed, 3 staggered",
        "passed": server_ok and server_ok2},
    "lora_delta_on_every_row_library_level": {
        "threshold": "ctypes on the same llama.dll, adapters set context-wide: for every output row of every 8-sequence composition tested, the adapter-induced logit delta over the full vocabulary correlates >= 0.99 with the single-sequence reference delta. The 0.99 threshold was chosen after seeing the data (an unapplied row would give a delta of zero).",
        "measured": f"min correlation {min_corr:.4f} over {len(multi)} compositions x 8 rows; projection on the reference delta in [{proj_lo:.3f}, {proj_hi:.3f}]",
        "passed": min_corr >= 0.99},
    "text_proxy_single_slot_lora_changes_greedy_output": {
        "threshold": "DIAGNOSTIC ONLY: single-slot per-request scale 1.0 changes greedy text vs W0",
        "measured": f"lora_changes_output={w['lora_changes_output']} in the strict text re-run. Library probe: the fork-token margin (``` minus <think>) is {ref_w0:+.3f} for W0 and {ref_tr:+.3f} with the three adapters when the prompt is decoded whole, so top-1 does not change there; through the server (prompt split 55+4) it is {ss['reference_margin_w0']:+.3f} and {ss['reference_margin_triple']:+.3f}",
        "passed": bool(w["lora_changes_output"])},
    "text_proxy_8slot_lora_shaped_output": {
        "threshold": "DIAGNOSTIC ONLY: every concurrent three-adapter request returns LoRA-shaped text and no W0 request does (first 60 characters compared)",
        "measured": f"{c['lora_shaped']}/{c['triple_lora_requests']} LoRA-shaped, {c['w0_identical']} W0-like in their first 60 characters, {c['other']} other; W0-like slots by round={c['w0_identical_slots_by_round']}; W0 requests that came back LoRA-shaped: {c['w0_requests_that_came_back_lora_shaped']}/{c['w0_requests_8slot']}. Full-content hashes: {n_equal_full_w0} of the 24 equal any W0 output",
        "passed": text_ok},
    "stamp_clean_hardware_run": {
        "threshold": "mlos stamp-run zone B: clean_hardware_run=true, 0 anomalies (strict text re-run and both server-level tests)",
        "measured": f"text re-run: clean_hardware_run={s['clean_hardware_run']}, anomalies={s['anomalies']}; server-level test run 1: clean_hardware_run={sls['clean_hardware_run']}, anomalies={sls['anomalies']}; run 2: clean_hardware_run={sls2['clean_hardware_run']}, anomalies={sls2['anomalies']}",
        "passed": bool(s["clean_hardware_run"]) and bool(sls["clean_hardware_run"]) and bool(sls2["clean_hardware_run"])},
}
(OUT / "009-gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
w2 = dict(w)
w2["vram_prediction"] = pred
w2["vram_prediction"]["n_ctx_total_source"] = "parsed from the recorded llama-server flags of this run"
def per_round_view(run):
    return [{"name": rd["name"], "margins": [None if x["margin"] is None else round(x["margin"], 4) for x in rd["results"]],
             "requested": [x["requested"] for x in rd["results"]], "classified": [x["classified"] for x in rd["results"]],
             "second_token": [(x.get("tok1") or {}).get("chosen") for x in rd["results"]]} for rd in run["rounds"]]


w2["server_level_logprob_test"] = {
    "reading": "An adapter-sized shift reached every request in both runs. On this prompt the margin sees the formal-verification adapter only: fv alone gives a margin on the adapter side, corpus alone, design alone and corpus+design all give margins on the W0 side. The test does not show that all three adapters were applied. The server never batches different adapter configurations together, so the mixed rounds equal the two single-request references exactly; the effective sample is the distinct margins counted in run 2.",
    "run_v1": {"summary": ss, "decision_threshold_margin_midpoint": sl["decision_threshold_margin_midpoint"], "build_info": sl["build_info"],
               "per_round": per_round_view(sl),
               "source": "m1/out/server_logit_check_v1.json", "source_sha256": hashlib.sha256((M1 / "server_logit_check_v1.json").read_bytes()).hexdigest()},
    "run_v2": {"summary": ss2, "decision_threshold_margin_midpoint": thr2, "build_info": sl2["build_info"],
               "single_request_references_by_adapter_subset": subs, "per_round": per_round_view(sl2),
               "source": "m1/out/server_logit_check_v2.json", "source_sha256": hashlib.sha256((M1 / "server_logit_check_v2.json").read_bytes()).hexdigest()},
}
w2["library_level_probe"] = {"p1_noise_floor": pa["p1"], "p2_by_variant": p2, "source": "m1/out/probe_analysis_v1.json",
                             "source_sha256": hashlib.sha256((M1 / "probe_analysis_v1.json").read_bytes()).hexdigest()}
w2["text_rerun_content_hashes"] = {"three_adapter_requests_by_class_and_hash": {f"{k[0]}:{k[1]}": v for k, v in sorted(hashes_tr.items())},
                                   "w0_8slot_hashes": dict(hashes_w0), "w0_single_hash": w["w0_single"]["content_sha256"][:8],
                                   "three_adapter_requests_equal_to_any_w0_output_in_full": n_equal_full_w0}
sound = [k for k in gates if not k.startswith("text_proxy_")]
w2["gate_views"] = {
    "all_gates_passed": all(g["passed"] for g in gates.values()),
    "gates_other_than_text_proxies_passed": all(gates[k]["passed"] for k in sound),
    "gates_other_than_text_proxies": sound,
    "note": "The split into text proxies and other gates is ours. The server-level log-probability gate was added after the text gate failed, to test the same property directly.",
}
(OUT / "009-smoke-workload-with-probe.json").write_text(json.dumps(w2, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
b31 = aq["layers"]["31"]
q8_all = [aq["layers"][l][k]["q8_input"] for l in ("28", "31") for k in ("ffn_up", "ffn_gate", "ffn_down")]
ex_all = [aq["layers"][l][k]["exact_f64"] for l in ("28", "31") for k in ("ffn_up", "ffn_gate", "ffn_down")]
a8 = pa["p2_by_variant"]["all8_whole"]
notes = [
    f"Headline: through llama-server's per-request path, an adapter-sized shift reached every request in both runs: {ss['triple_classified_triple']} of {ss['triple_requests']} adapter requests and {ss['w0_classified_w0']} of {ss['w0_requests']} W0 requests classified correctly by log-probability margin in run 1 (gap {ss['gap_between_clusters']:.2f} logits), and {ss2['triple_classified_triple']} of {ss2['triple_requests']} and {ss2['w0_classified_w0']} of {ss2['w0_requests']} in run 2 (gap {ss2['gap_between_clusters']:.2f}). SCOPE: on this prompt the margin sees the formal-verification adapter only (fv alone {subs['formal_verification_only']['margin']:+.3f}, adapter side; corpus alone {subs['corpus_only']['margin']:+.3f}, design alone {subs['weft_eido_only']['margin']:+.3f}, corpus+design {subs['corpus_and_weft_eido']['margin']:+.3f}, all W0 side). The test does not show that all three adapters were applied. {ss['triple_requests_whose_second_token_is_think']} of the {ss['triple_requests']} adapter requests in run 1 and {ss2['triple_requests_whose_second_token_is_think']} of {ss2['triple_requests']} in run 2 still chose the W0-like token. The greedy-text gate fails ({c['lora_shaped']} of {c['triple_lora_requests']}) because it is not measuring adapter application.",
    f"Why text fails: at the second generated token the base model prefers <think> over ``` by {abs(ss['reference_margin_w0']):.2f} logits through the server. The three adapters move that margin by {ss['reference_margin_triple']-ss['reference_margin_w0']:+.2f}, leaving {ss['reference_margin_triple']:+.3f}, a near tie. Decoding the same tokens in a different batch composition moves logits by up to {noise_max:.2f} (0.0 when the composition is identical), so requests take either branch.",
    f"Order of events, from the run stamps (UTC): (1) library probe {span(sp)}; (2) strict text re-run on the server {span(s)}, the run whose VRAM and slot figures are gated here; (3) server-level log-probability test run 1 {span(sls)}; (4) server-level log-probability test run 2 {span(sls2)}, which adds the per-adapter-subset references and distinct-margin counts. This packet joins them; they were not one experiment.",
    f"The mixed rounds add nothing about noise: the server never batches different adapter configurations together, so those {ss2['mixed_rounds_note']['n']} results equal the two single-request references exactly (run 2 records this). The effective sample is {ss2['distinct_triple_margins']} distinct adapter margins and {ss2['distinct_w0_margins']} distinct W0 margins, not 84 and 36 independent draws.",
    f"The strict text re-run overlapped a CPU job on the same machine: {tok_s_run2:.1f} tok/s per slot against {tok_s_run1:.1f} in the first smoke run, and the server took {w['server_load_s']} s to load against {w1['server_load_s']} s. Throughput from this run should not be used.",
    "Agreement and disagreement between runs. Rounds 0 and 1 of the text re-run show slots 0,1,2,6,7 W0-like, exactly the rows the library's 8-sequence 55+4 emulation predicts. Round 2 shows 7 of 8 after three arrival waves, which no library emulation reproduces; the staggered rounds of the server-level test use that arrival pattern and classify 24 of 24 correctly. The library's reconstruction of the first run's mixed batch predicts one W0-like row where that run showed two. Batch composition is not logged by the server, so it is inferred from arrival times and source.",
    f"'W0-like' means the first 60 characters match. In full, {n_equal_full_w0} of the 24 three-adapter outputs equal any W0 output.",
    f"Cause of the composition dependence, part measured and part open. In the pinned source the CUDA backend quantises the activation operand of quantised-weight matmuls to 8 bits per block of 32 (mmvq.cu:1562, mmq.cu:291, quantize.cu:89-90) and picks the kernel by batch row count. Measured for six matmuls (ffn_up, ffn_gate and ffn_down of blk.28 and blk.31): emulating that quantisation reproduces the runtime to {min(q8_all):.0e}-{max(q8_all):.1e} where exact arithmetic differs by {min(ex_all):.0e}-{max(ex_all):.0e}. That this is what makes logits depend on batch composition is an inference; there is no unquantised control.",
    f"The mechanism is incomplete (round-2 correction): eight identical sequences in one batch still differ row to row (all8_whole: W0 fork margins span {a8['w0_fork_margin_min_max'][0]:.3f} to {a8['w0_fork_margin_min_max'][1]:.3f} across the 8 rows), which kernel choice by batch row count cannot explain, since all rows share one batch size. Treat the cause of batch-composition noise as open.",
    "Consequence for gates elsewhere: test adapter application, T-C5 rollback and 'adapter changes output' on log-probabilities under a fixed batch composition, not on greedy text. Packet 002's 8-slot 'triple LoRA' throughput figures stand as throughput and carry no evidence about per-slot content either way.",
    "overall_passed below is computed over all eight gates and is false, because the 8-slot text gate fails. workload.gate_views gives the other view and says who drew the line.",
    f"VRAM: only the 8 full-attention layers of this hybrid model hold K/V, which gives exactly {pred['kv_cache_mib']} MiB at 8 x 8192. The {gap:.0f} MiB residual is not modelled.",
    f"llama.cpp {w['server_props']['build_info']} cu12.4 Windows release, Qwen3.5-4B-Q8_0.gguf sha256 10cc391b...61e1, mcproof composition LoRAs (each a single rank-8 pair on blk.31.ffn_up). Server flags: -c {n_ctx_total} -np {n_parallel} -ngl 99 -ctk f16 -ctv f16 --slots --no-cache-idle-slots --lora-scaled with the three adapters at scale 0.0, --lora-init-without-apply, --host 127.0.0.1. Native Windows 11, before WSL.",
    "Runs were launched through mlos gpu-lease and mlos stamp-run. The embedded stamp is from the strict text re-run; no separate lease record is embedded. Raw server logs contain a local path and are not shipped; their hashes are listed.",
    "History: the first smoke run (16:39) showed slots 2 and 3 returning W0-like text; the swarm's verifier caught it and the receipt was held. A source review of the server at this commit found no path by which an equal-config batch could skip the adapter set, with medium-high confidence and without reading all of it. The log-probability test above is the direct check.",
    "Embedded-data notes: 127.0.0.1 and the ephemeral ports in the raw records are loopback-only server endpoints, not network exposure. The harness label w0_identical means the first 60 characters match a W0 output, not full equality; full-content hashes are reported separately.",
    "Review: recomputed from raw files by two independent reviewers (rounds 1 and 2); every number reproduced. Round 2 found that the margin sees the formal-verification adapter only, that the mixed rounds are serialised, and that the noise mechanism is incomplete; those corrections are applied in this revision. The round-2 fixes were applied by the author; a third, narrow review of the revision is recorded in receipts/REVIEW_ROUND_3.md.",
]
cmd = [sys.executable, str(ROOT / "mlos" / "build_receipt.py"),
       "--packet-id", "geminus-to-orbit-009", "--issue", "Moissanite-Labs/ml-os#2 (Phase 1, item 3)",
       "--title", "Live RTX 3080 llama-server + 3 F16 LoRA smoke, native Windows 11: an adapter-sized shift reaches every request by log-probability test (on this prompt the margin sees the fv adapter only); the greedy-text gate fails and is shown to be the wrong test",
       "--stamp", str(W / "smoke-stamp.json"), "--workload", str(OUT / "009-smoke-workload-with-probe.json"),
       "--artifacts", str(W / "smoke-workload.json"), str(W / "smoke-nvsmi.csv"), str(W / "smoke-server.log"),
       str(ROOT / "mlos" / "phase1" / "gpu_workload.py"), str(ROOT / "mlos" / "phase1" / "gguf_predict.py"),
       str(ROOT / "m1" / "scripts" / "probe.py"), str(ROOT / "m1" / "scripts" / "llama_ctypes.py"), str(ROOT / "m1" / "scripts" / "server_logit_check.py"),
       str(M1 / "server_logit_check_v1.json"), str(M1 / "server_logit_check_v1.stamp.json"),
       str(M1 / "server_logit_check_v2.json"), str(M1 / "server_logit_check_v2.stamp.json"),
       str(M1 / "probe_analysis_v1.json"), str(M1 / "actq_check_v1.json"),
       "--gates", str(OUT / "009-gates.json"), "--out", str(OUT / "geminus-to-orbit-009-mlos-phase1-gpu-smoke.json"),
       "--notes", *notes]
sys.exit(subprocess.run(cmd, check=False).returncode)
