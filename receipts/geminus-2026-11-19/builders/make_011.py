"""Packet 011 (revised after review round 1): runtime noise floor, per-slot LoRA delivery, rollback,
what the shipped adapters do, zero-contrast design pairs, target-residual preconditioning.

Not requested; offered. All numbers are read from m1/out result files.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
M1 = ROOT / "m1" / "out"
OUT = ROOT / "receipts"


def load(name):
    return json.loads((M1 / name).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r(x, n=6):
    return [round(float(v), n) for v in x] if isinstance(x, (list, tuple)) else round(float(x), n)


def sci(x):
    return float(f"{x:.3e}")


pa = load("probe_analysis_v1.json")
aq = load("actq_check_v1.json")
an = load("m1_analysis_v1.json")
tr = load("target_residual_exp_v1.json")
ce = load("compile_exp_v1.json")
live = load("alloc_live_v2.json")
hmeta = load("harvest_v1.meta.json")
sl = load("server_logit_check_v1.json")
rb = load("rollback_probe_v1.json")
pvl = load("alloc_passthrough_vs_live.json")
man = json.loads((M1 / "alloc_adapters" / "manifest.json").read_text(encoding="utf-8"))
stamp_p = load("probe_v1.stamp.json")
st = hmeta["states"]
ind = pvl["held_out_independence"]
held_caveat = (f"'Held-out' means held-out prompt ids ({ind['n_held_out_prompts']} of {ind['n_train_prompts'] + ind['n_held_out_prompts']}, 0 overlap) inside the same four template families; "
               f"{100*ind['held_out_states_reusing_a_training_pos_neg_token_pair_frac']:.0f}% of held-out states reuse a (positive token, negative set) pair seen in training. The anchors evaluated are the compile-time preservation keys.")

# ---- A. per-slot delivery ----------------------------------------------------------------------------
p1, p2 = pa["p1"], pa["p2_by_variant"]
multi = {k: v for k, v in p2.items() if v["n_rows"] == 8}
ss = sl["summary"]
A = {
    "claim": "Per-request adapters reach every slot of llama-server. The earlier 'W0-like text on some slots' was a near tie at one token plus batch-composition noise.",
    "server_level_logit_test": {
        "what": "llama-server b12549, 8 slots, per-request 'lora' field, n_probs log-probabilities at the second generated token of the packet-009 smoke prompt. margin = logprob(```) - logprob(<think>).",
        "decision_rule_fixed_before_the_run": "a request counts as 'adapters applied' when its margin is above the midpoint of the two single-request reference margins from the same server session",
        "reference_margin_w0": r(ss["reference_margin_w0"], 4), "reference_margin_three_adapters": r(ss["reference_margin_triple"], 4),
        "adapter_requests": ss["triple_requests"], "adapter_requests_classified_as_adapter": ss["triple_classified_triple"],
        "adapter_margin_range": [r(ss["triple_margin_min"], 4), r(ss["triple_margin_max"], 4)],
        "w0_requests": ss["w0_requests"], "w0_requests_classified_as_w0": ss["w0_classified_w0"],
        "w0_margin_range": [r(ss["w0_margin_min"], 4), r(ss["w0_margin_max"], 4)],
        "gap_between_the_two_clusters": r(ss["gap_between_clusters"], 4),
        "adapter_requests_whose_greedy_second_token_was_think": ss["triple_requests_whose_second_token_is_think"],
        "rounds": "6 rounds of 8 concurrent adapter requests, 3 of 8 concurrent W0, 3 mixed (4 W0 + 4 adapter, alternating slots), 3 with staggered arrivals in three waves",
        "slot_served_equals_slot_requested": ss["slot_served_equals_requested"],
    },
    "library_level_probe": {
        "what": "ctypes on the same llama.dll, adapters set context-wide, full logit vectors, three 8-sequence batch compositions. This does not exercise the server's per-request path; the test above does.",
        "identical_decode_repeated_max_abs_logit_diff": p1["repeat_single_whole_w0_max_abs_logit_diff"],
        "same_tokens_different_batch_composition_max_abs_logit_diff": {k: r(v["w0_noise_vs_ref_max_abs_logit"], 4) for k, v in p2.items()},
        "fork_margin_w0_whole_prompt": r(p1["reference_w0"]["fork_margin_fence_minus_think"], 4),
        "fork_margin_three_adapters_whole_prompt": r(p1["reference_triple"]["fork_margin_fence_minus_think"], 4),
        "min_correlation_of_adapter_logit_delta_with_reference_by_composition": {k: r(v["triple_delta_corr_min"], 4) for k, v in multi.items()},
        "projection_on_reference_delta_range": {k: r(v["triple_delta_projection_min_max"], 3) for k, v in multi.items()},
        "rows_of_8_whose_top1_with_adapters_equals_w0_top1": {k: len(v["rows_where_triple_top1_equals_w0_top1"]) for k, v in multi.items()},
    },
    "cross_checks_and_loose_ends": [
        f"The server's single-request reference margins ({ss['reference_margin_w0']:.4f}, {ss['reference_margin_triple']:.4f}) equal the library emulation of a prompt split 55+4 ({[x for x in pa['p2_lora_rows_single_split']] if 'p2_lora_rows_single_split' in pa else 'see probe_analysis single_split_55_4'}).",
        "In the strict text re-run (packet 009) rounds 0 and 1 showed slots 0,1,2,6,7 taking the W0-like branch, which is exactly the row set of the library's 8-sequence 55+4 emulation.",
        "Round 2 of that re-run showed 7 of 8 W0-like after three arrival waves. No library emulation reproduces 7 of 8; the staggered rounds of the server-level test cover that arrival pattern and classify 24 of 24 correctly by margin.",
        "The library's reconstruction of the first run's mixed batch predicts one W0-like row where that run showed two (the second has margin +0.0096).",
        "All of this is one prompt and one token position.",
    ],
    "recommendation": "Test adapter application, rollback and 'adapter changes output' on logits or log-probabilities, not on greedy text. For an identical batch composition require an exact match; otherwise use a margin or delta-correlation test with a threshold set from reference requests.",
}
A["cross_checks_and_loose_ends"][0] = (f"The server's single-request reference margins ({ss['reference_margin_w0']:.4f} and {ss['reference_margin_triple']:.4f}) equal the library emulation of the prompt split 55+4 to four decimals "
                                       f"({p2['single_split_55_4']['w0_fork_margin_min_max'][0]:.4f} and {p2['single_split_55_4']['triple_fork_margin_min_max'][0]:.4f}). That split was inferred from source; this confirms it.")

# ---- B. activation quantisation ------------------------------------------------------------------------
B = {
    "claim": "With Q8_0 weights the CUDA backend quantises the activation operand of quantised-weight matmuls to 8 bits per block of 32.",
    "in_the_pinned_source": ["ggml/src/ggml-cuda/mmvq.cu:1562 quantize_row_q8_1_cuda(src1 ...)", "ggml/src/ggml-cuda/mmq.cu:291 quantize_mmq_q8_1_cuda(src1 ...)",
                             "ggml/src/ggml-cuda/quantize.cu:89-90 d = amax / 127.0f; q = roundf(xi / d), over QK8_1 = 32 elements",
                             "ggml/src/ggml-cuda/ggml-cuda.cu selects between these kernels by the number of rows in the batch"],
    "measured_for_six_matmuls": {
        "what": f"ffn_up, ffn_gate and ffn_down of blk.28 and blk.31, {aq['n_states']} single-sequence states: relative L2 error between a float64 reconstruction from the harvested input and the runtime's output",
        "by_layer": {l: {k: {"exact_float64": sci(v[k]["exact_f64"]), "float16_input": sci(v[k]["f16_input"]), "q8_block32_input": sci(v[k]["q8_input"]),
                             "improvement_factor": round(v[k]["improvement_factor_q8_over_exact"], 1)} for k in ("ffn_up", "ffn_gate", "ffn_down")} for l, v in aq["layers"].items()},
        "block_size_is_specific": {l: {"ffn_up": {b: sci(x) for b, x in v["ffn_up_by_block_size"].items()}, "ffn_down": {b: sci(x) for b, x in v["ffn_down_by_block_size"].items()}} for l, v in aq["layers"].items()},
        "size_of_the_activation_rounding_itself": {l: {k: sci(x) for k, x in v["q8_input_error_itself"].items()} for l, v in aq["layers"].items()},
    },
    "f16_lora_branch": "The LoRA factors are F16, so their matmuls take the float path. Adapter deltas are computed on unquantised activations while the base is not. (Read from the source; not measured separately.)",
    "link_to_batch_composition_noise": "INFERRED. Kernel choice depends on batch row count in the source, and the noise appears exactly when composition changes, but there is no unquantised control (an F16 GGUF of this model does not fit in 8 GiB).",
    "resolution_limit": {
        "final_residual": "about 0.09 in norm, 0.2% of the radius. On the 24 anchors the measured change under the corpus adapter is 0.116 where exact arithmetic predicts 0.069.",
        "logits": "0.0 for an identical batch; up to 0.26-0.46 across batch compositions on the example prompt",
        "meaning": "An adapter effect below these levels cannot be observed on this runtime. A leakage bound tighter than the floor is valid arithmetic with no observable consequence here.",
    },
}

# ---- rollback -----------------------------------------------------------------------------------------
rbs = rb["summary_by_k"]
ROLL = {
    "weight_level": {
        "statement": "After adapters are applied and removed, the base final residual is restored bit for bit when the batch composition is the same.",
        "max_abs_diff_after_adapter_sweeps": {"probe_run_5_conditions_20_sequences": pa["p3_base_repeat_max_abs_diff"],
                                              "live_allocation_run_24_adapter_configurations_152_states": live["base_repeat_max_abs_diff"]},
        "scope": "Each decode started from cleared memory. This is the scale-to-zero step of T-C5, not cache rollback.",
    },
    "mid_sequence_without_cache_flush": {
        "what": "Decode a prefix with one single-layer adapter on (scale 1), switch it off WITHOUT clearing memory, decode k more tokens, compare logits with a run that never had the adapter. 16 sequences.",
        "by_k": {k: {f"blk{l}": {"max_abs_logit_diff_median": r(v[f"blk{l}"]["max_abs_logit_diff_median"], 4), "max_abs_logit_diff_max": r(v[f"blk{l}"]["max_abs_logit_diff_max"], 4),
                                 "kl_mean": sci(v[f"blk{l}"]["kl_mean"]), "top1_changed_frac": v[f"blk{l}"]["top1_changed_frac"], "exact_match_frac": v[f"blk{l}"]["exact_match_frac"]}
                     for l in (28, 29, 30, 31)} | {"clean_repeat_max_abs_diff": v["clean_repeat_max_abs_diff"], "flush_then_whole_decode_max_abs_diff": v["flush_control_max_abs_diff"]}
                 for k, v in rbs.items()},
        "reading": "A blk.31 FFN adapter leaves no trace: nothing follows the last FFN, so it cannot write into KV or recurrent state, and rollback is exact with no flush. Adapters at blk.28..30 leave a trace in the caches that is still present 16 tokens later. At these strengths the trace is small (KL below 1e-3, no top-token change in 16 sequences).",
        "consequence": "The cache-flush step of the rollback contract is necessary for earlier-layer adapters and unnecessary for last-layer ones.",
    },
}

# ---- C. shipped adapters -------------------------------------------------------------------------------
ship = {}
for name, rec in pa["p3_shipped_adapters"].items():
    ship[name] = {"live_vs_exact_float64_relerr": r(rec["live_vs_exact_pred_relerr_all_states"], 4), "live_vs_exact_cosine": r(rec["live_vs_exact_pred_cos_mean"], 4),
                  "by_state_group": {g: {"n": b["n"], "dh_over_R0": r(b["dh_over_R0_mean"], 5), "eta_mean": sci(b["eta_mean"]), "eta_max": sci(b["eta_max"]),
                                         "contrast_gain_mean": None if b["contrast_gain_mean"] is None else r(b["contrast_gain_mean"], 4),
                                         "top1_changed_frac": r(b["top1_changed_frac"], 3)} for g, b in rec["by_group"].items()}}
fvr = pa["p3_shipped_adapters"]["formal-verification"]["by_group"]
C = {
    "what": "The three composition adapters at mcproof 4552874 (each one rank-8 pair on blk.31.ffn_up), scale 1, measured live on 372 states and compared with the exact float64 prediction W_down (s * (B A x)).",
    "adapters": ship,
    "selectivity_of_formal_verification_adapter": {
        "dh_over_R0_on_own_training_states": r(fvr["fv"]["dh_over_R0_mean"], 4), "on_corpus_prompts": r(fvr["corpus"]["dh_over_R0_mean"], 4),
        "on_design_prompts": r(fvr["eido"]["dh_over_R0_mean"], 4), "on_y0_anchors": r(fvr["anchor"]["dh_over_R0_mean"], 5),
        "reading": "It moves unrelated chat prompts about as much as its own training states and leaves the 24 anchors almost untouched. Preservation holds on the listed anchors and does not extend to their neighbours. n is 8 prompts per small pack.",
        "suggested_check": "Measure preservation on off-domain prompts that were never preservation keys.",
    },
    "eta_note": f"Mean eta on own training states: formal-verification {fvr['fv']['eta_mean']:.3f} (max {fvr['fv']['eta_max']:.3f}), corpus {pa['p3_shipped_adapters']['corpus']['by_group']['corpus']['eta_mean']:.3f}, design {pa['p3_shipped_adapters']['weft-eido']['by_group']['eido']['eta_mean']:.3f}. These single-layer adapters predate the 0.01 per-layer cap.",
}

# ---- D. design-pack contrast pairs ---------------------------------------------------------------------
def kind(s):
    p, n = s["pos_toks"], s["neg_toks"]
    if sorted(p) == sorted(n):
        return "zero: identical token sets"
    if set(p) & set(n):
        return "partial: first token shared"
    return "distinct"


eido = [{"id": s["id"], "pos_token_ids": s["pos_toks"], "neg_token_ids": s["neg_toks"], "contrast": kind(s)} for s in st if s["group"] == "eido"]
bm = pvl["base_model_on_formal_verification_states"]
D = {
    "statement": "In activation_harvester.py the corpus and design mini-packs take the first two tokens of the good and bad completions as the positive and negative token sets. For the design pack that makes 3 of 8 contrasts exactly zero, and 4 more depend only on the second token.",
    "pairs": eido,
    "counts": {k: sum(1 for e in eido if e["contrast"] == k) for k in ("zero: identical token sets", "partial: first token shared", "distinct")},
    "consequence": "exact_up_proj_swiglu_residual_f64 returns a zero residual when the contrast is zero. A zero target on a real key acts as a preservation constraint, so three design prompts are held at baseline. For the four partial pairs the target promotes the second token at the position that predicts the first.",
    "corpus_pack": "all 8 pairs distinct",
    "possible_link": "In e8_qwen35_4b_live_tc1_receipt.json weft-eido has the smallest solo shift in the single-layer arm (0.002) and the lowest triple retention in both arms. This could contribute. We did not recompile the pack, and in the 4-layer arm its solo shift is not the smallest.",
    "fix": "Take the first position at which the two completions differ, as the formal-verification path already does, or skip pairs with an empty contrast.",
    "related_observation": {"statement": "On the formal-verification decision states the base model often already agrees with the target.",
                            "positive_token_is_already_top1_frac": r(bm["positive_token_is_top1_frac"], 3), "positive_beats_every_listed_negative_frac": r(bm["positive_beats_every_listed_negative_frac"], 3),
                            "median_margin_over_listed_negatives_logits": r(bm["median_margin_vs_listed_negatives"], 2), "n": bm["n"],
                            "reading": "The listed negatives are rarely the real competition. Weighting states by how contested they are, and choosing negatives from what the base actually prefers, are options."},
}

# ---- E. preconditioning ---------------------------------------------------------------------------------
tsum = tr["summary"]["G1"]
per_state = {k: {"eta_relative_to_plain": r(tsum[k]["eta_median"] / tsum["inf"]["eta_median"], 3), "kl_top32_relative_to_plain": r(tsum[k]["kl_median"] / tsum["inf"]["kl_median"], 3),
                 "dlogp_positive": r(tsum[k]["dlogp_pos_median"], 4), "up_residual_energy_relative_to_plain": r(tsum[k]["du_norm_sq_median"] / tsum["inf"]["du_norm_sq_median"], 2)}
             for k in ("inf", "10", "1", "0.1", "0.01", "0")}
comp = {}
for arm, res in ce["results"].items():
    e = res["equal_gain"]
    comp[arm] = {"lambda_rel": res["lambda_rel"], "scale_to_equal_gain": r(e["scale_applied"], 3), "fro_norm_at_that_scale": r(e["scaled_fro_norm"], 3),
                 "k0_noise_leak_at_that_scale": r(e["scaled_k0_noise_leak_fro"], 4), "truncation_rel_error": r(res["adapter"]["truncation_rel_error"], 3),
                 "held_out_gain_mean": r(e["held_out"]["gain_exact"]["mean"], 4), "held_out_gain_median": r(e["held_out"]["gain_exact"]["median"], 4),
                 "held_out_eta_mean": r(e["held_out"]["eta"]["mean"], 6), "held_out_eta_median": r(e["held_out"]["eta"]["median"], 6),
                 "held_out_dlogp_positive_mean": r(e["held_out"]["dlogp_pos"]["mean"], 4), "held_out_kl_top32_mean": r(e["held_out"]["kl"]["mean"], 4),
                 "anchors_kl_top32_mean": sci(e["anchors"]["kl"]["mean"]), "anchors_top1_changed_frac": r(e["anchors_top1_changed_frac"], 3)}
lv = {}
for arm in ("plain", "lam10", "lam1", "lam0.1", "minnorm"):
    k = f"c31_{arm}"
    m = live["pass2"][k]
    h = m["held_out_fv"]
    n1 = live["pass1_scale1"][k]["held_out_fv"]
    lv[arm] = {"scale": r(list(m["scales"].values())[0], 4), "realised_gain": r(h["gain_mean"], 4), "eta": r(h["eta_mean"], 6), "eta_per_gain_sq": r(h["cost_mean_eta_over_mean_gain_sq"], 6),
               "eta_per_gain_sq_median_over_states": r(h["cost_median_per_state_eta_over_gain_sq"], 6), "dlogp_positive": r(h["dlogp_pos_mean"], 4),
               "kl_on_domain_full_vocab": r(h["kl_mean"], 4), "corpus_prompts_dh_over_R0": r(m["corpus"]["dh_over_R0_mean"], 4), "corpus_prompts_kl": r(m["corpus"]["kl_mean"], 4),
               "design_prompts_kl": r(m["eido"]["kl_mean"], 4), "anchors_dh_over_R0": r(m["anchor"]["dh_over_R0_mean"], 5), "anchors_top1_changed_frac": r(m["anchor"]["top1_changed_frac"], 3),
               "native_scale_gain_live": r(n1["gain_mean"], 4), "native_scale_gain_float64": r(ce["results"][arm]["native_scale"]["held_out"]["gain_exact"]["mean"], 4),
               "native_scale_eta_live": r(n1["eta_mean"], 6), "native_scale_eta_float64": r(ce["results"][arm]["native_scale"]["held_out"]["eta"]["mean"], 6)}
pl, l1, l01, mn, l10 = lv["plain"], lv["lam1"], lv["lam0.1"], lv["minnorm"], lv["lam10"]
E = {
    "statement": "The harvester builds the up-projection target as the contrast gradient pulled back by the transpose, du = s * (W_down^T g). That is the steepest-descent direction. Preconditioning with the gate metric M = W_down diag(s^2) W_down^T, du = s * W_down^T (M + lambda I)^-1 g, buys the same contrast gain with a smaller final-residual change.",
    "derivation": "For a W_up edit at the last layer, dh = W_down (s * du) exactly. The plain choice gives dh = M g, whose energy per unit gain squared exceeds the minimum (dh parallel to g) by kappa = (g'M^2 g)(g'g)/(g'M g)^2.",
    "kappa_measured_blk31": {"median": r(tr["M_spectrum"]["kappa_plain_median"], 2), "n_states": tr["n_states"], "by_layer_mean_28_to_31": r(an["gate_metric_pass_through"]["kappa"]["mean"], 2),
                             "gate_metric_participation_ratio_median": r(tr["M_spectrum"]["participation_ratio_median"], 1), "of_dimension": hmeta["d_model"]},
    "stage_1_per_state_exact": {"what": "single state, unconstrained du, exact float64, equal first-order gain of 1 logit; lambda in units of the mean eigenvalue of M; 'inf' is the plain target, '0' the minimum-norm one; KL over a top-32 support",
                                "by_lambda": per_state},
    "stage_2_compiled_rank8": {"what": "the team's BetaEdit projector with the P2-B recipe, 220 training states of 64 prompts, evaluated on 112 states of 32 held-out prompt ids in exact float64, every arm rescaled to the same mean gain; KL over a top-32 support",
                               "by_arm": comp},
    "stage_3_live_gpu": {"what": "the same compiled factors written as GGUF LoRAs and measured on the RTX 3080 at equal target gain; KL over the full vocabulary", "by_arm": lv},
    "what_improves_at_lambda_1": [
        f"Radial shrinkage per unit gain squared: {pl['eta_per_gain_sq']:.5f} to {l1['eta_per_gain_sq']:.5f} live ({l1['eta_per_gain_sq']/pl['eta_per_gain_sq']:.2f}x), and the per-state median {pl['eta_per_gain_sq_median_over_states']:.5f} to {l1['eta_per_gain_sq_median_over_states']:.5f}.",
        f"Log-probability of the positive token at equal contrast gain: {pl['dlogp_positive']:+.3f} to {l1['dlogp_positive']:+.3f} live.",
        f"Residual movement on unrelated corpus prompts: {pl['corpus_prompts_dh_over_R0']:.4f} to {l1['corpus_prompts_dh_over_R0']:.4f}.",
        "Live gain and eta match the float64 prediction to three or four digits at native scale.",
    ],
    "what_gets_worse_or_does_not_improve": [
        f"KL on unrelated corpus prompts RISES: {pl['corpus_prompts_kl']:.3f} to {l1['corpus_prompts_kl']:.3f} at lambda 1 and {l01['corpus_prompts_kl']:.3f} at lambda 0.1, although the residual moves less. A smaller residual change is not a smaller change in distribution.",
        f"On-domain KL at equal gain does not improve: {pl['kl_on_domain_full_vocab']:.3f} plain, {l1['kl_on_domain_full_vocab']:.3f} at lambda 1, {l01['kl_on_domain_full_vocab']:.3f} at lambda 0.1, {mn['kl_on_domain_full_vocab']:.3f} minimum-norm.",
        f"Weight norm and K0 leakage at equal gain rise by about half at lambda 1 ({comp['plain']['fro_norm_at_that_scale']} to {comp['lam1']['fro_norm_at_that_scale']}; {comp['plain']['k0_noise_leak_at_that_scale']} to {comp['lam1']['k0_noise_leak_at_that_scale']}), and the target is less low-rank (truncation error {comp['plain']['truncation_rel_error']} to {comp['lam1']['truncation_rel_error']}).",
        f"The unregularised minimum-norm target is worse once compiled: positive-token log-probability {mn['dlogp_positive']:+.3f} live against {pl['dlogp_positive']:+.3f} plain, at 8 times the weight norm.",
        f"lambda 10 improves the mean but not the per-state median of eta after compilation ({comp['plain']['held_out_eta_median']} plain, {comp['lam10']['held_out_eta_median']} at lambda 10): equal mean gain can hide redistribution across states (median gain {comp['plain']['held_out_gain_median']} plain, {comp['lam1']['held_out_gain_median']} at lambda 1).",
    ],
    "sizing": "This is the gradient versus Gauss-Newton distinction applied to an edit that sits upstream of a gate and a down-projection. The principle is standard. What is offered is the identification in this pipeline and a three-stage measurement, with a mixed result: less shrinkage and more of the objective, more distribution change off-domain.",
    "limits": ["blk.31 only, formal-verification states only, one split, safety directions not registered.", held_caveat,
               "Stages 1 and 2 measure KL over a top-32 support; stage 3 over the full vocabulary."],
    "reproducibility_note": {"statement": "Two scripts that call the same compiler with the same inputs wrote byte-identical GGUF files. This shows the compile is reproducible on this machine and nothing more.",
                             "sha256": man["adapters"]["plain_blk31"]["sha256"], "equal": man["adapters"]["plain_blk31"]["sha256"] == man["adapters"]["c31_plain"]["sha256"]},
}

receipt = {
    "schema": "geminus-to-orbit-runtime-and-pipeline-findings/v2",
    "packet_id": "geminus-to-orbit-011",
    "in_reply_to": "not requested; arises from orbit-to-geminus-005 action 1 and from the smoke anomaly that held the swarm's first packet",
    "title": "Per-slot LoRA delivery and rollback on the RTX 3080, the runtime noise floor, what the shipped adapters do, zero-contrast design pairs, and target-residual preconditioning",
    "summary": [
        f"A. Per-request adapters reach every slot: {ss['triple_classified_triple']} of {ss['triple_requests']} adapter requests and {ss['w0_classified_w0']} of {ss['w0_requests']} W0 requests classified correctly by log-probability margin at server level, while {ss['triple_requests_whose_second_token_is_think']} of those {ss['triple_requests']} adapter requests produced W0-like greedy text.",
        "B. Q8_0 matmuls quantise their input activations to 8 bits on this backend (in the source, and measured for six matmuls). We infer this is the source of batch-composition noise. It sets a floor below which adapter effects cannot be observed.",
        "Rollback. A last-layer adapter rolls back exactly with no cache flush (16 of 16). Adapters at blk.28..30 leave a trace in KV and recurrent state that is still present 16 tokens later.",
        "C. The shipped formal-verification adapter moves unrelated chat prompts as much as its own states, while the 24 anchors stay put.",
        "D. Three of the eight design-pack training pairs carry a zero target and four more a mistimed one.",
        "E. Preconditioning the target residual with the gate metric roughly halves radial shrinkage at equal gain after compilation and is confirmed live, but raises KL on unrelated prompts. A mixed result.",
    ],
    "A_per_slot_lora_delivery": A,
    "B_activation_quantisation_and_noise_floor": B,
    "rollback": ROLL,
    "C_shipped_adapters": C,
    "D_design_pack_contrast_pairs": D,
    "E_target_residual_preconditioning": E,
    "operational_note": {"statement": "Loading llama.dll from another process fell back to CPU without any error, because ggml loads its CUDA backend with a plain native LoadLibrary that could not resolve cuBLAS. The first harvest ran on CPU with 0 MiB of VRAM in use until this was noticed; it was discarded and repeated.",
                         "guard": "The harness now asserts a registered GPU device, 'offloaded N/N layers' in the load log and a drop of over 1 GiB in free VRAM. Any tool that embeds llama.cpp and must run on the GPU should assert the same."},
    "gpu_runs": {"probe": {"free_mib_start": stamp_p["start_gpu_telemetry"]["memory_free_mib"], "free_mib_end": stamp_p["end_gpu_telemetry"]["memory_free_mib"],
                           "clean_hardware_run": stamp_p["clean_hardware_run"], "duration_ms": stamp_p["execution"]["duration_ms"]},
                 "thermal": {"max_sampled_c": pa["thermal"]["max_temp_c_sampled"], "self_imposed_pauses_at_85c": len(pa["thermal"]["pauses"])},
                 "note": "Launched through mlos gpu-lease and mlos stamp-run; figures are the stamper's start and end snapshots."},
    "review": "Recomputed from raw files by an independent reviewer before sending; findings were raised and addressed, including one claim that was wrong (off-domain KL). The server-level test and the rollback probe were added after that review and checked by a second reviewer.",
    "artifact_sha256": {n: sha(M1 / n) for n in ("probe_analysis_v1.json", "probe_v1.json", "actq_check_v1.json", "m1_analysis_v1.json", "target_residual_exp_v1.json", "compile_exp_v1.json",
                                                 "alloc_live_v2.json", "server_logit_check_v1.json", "rollback_probe_v1.json", "alloc_passthrough_vs_live.json")}
                       | {f"adapter:{k}": v["sha256"] for k, v in man["adapters"].items() if k.startswith("c31_")},
    "scrubbed": "user paths and tailnet/LAN IPs replaced with placeholders; no keys or tokens. Raw logs are not shipped.",
}
SCRUB = [
    (re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\"/]+", re.I), "<USERHOME>"),
    (re.compile(r"/home/[^/\"]+"), "<USERHOME>"),
    (re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<TAILNET_IP>"),
    (re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"), "<LAN_IP>"),
    (re.compile(r"(?<!orbit-to-)geminus(?!-to-orbit)|<MAC_NODE>(-1)?|<TAILNET>", re.I), "<HOST>"),
]
text = json.dumps(receipt, indent=2, ensure_ascii=True)
for pat, rep in SCRUB:
    text = pat.sub(rep, text)
json.loads(text)
out = OUT / "geminus-to-orbit-011-runtime-and-pipeline-findings.json"
out.write_text(text + "\n", encoding="utf-8")
print("wrote", out.name, "sha256", hashlib.sha256((text + "\n").encode()).hexdigest(), "bytes", len(text))
for line in receipt["summary"] + E["what_improves_at_lambda_1"] + E["what_gets_worse_or_does_not_improve"]:
    print(" -", line)
