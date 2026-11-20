"""Packet 010 (revised after review rounds 1 and 2): Phase 3 Windows-side sustained-load thermal run (ml-os#4 item 3).

Built from the raw artifacts of the run that started 2026-11-19 16:39:59 local. The run ABORTED on its
own safety rule after 743 s. Gates and notes are computed from the data.
Round-2 corrections applied: brief power samples reach 96.6 W (the 95 W cap was touched, not sustained);
the plateau's CPU-load attribution is an inference, not a measurement; the stall monitor kept posting
after the excursion; "no second GPU process" was not recorded.
"""
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])            # geminus-work
W = ROOT / "mlos" / "phase3"
OUT = ROOT / "receipts"
w = json.loads((W / "thermal-workload.json").read_text(encoding="utf-8"))
s = json.loads((W / "thermal-stamp.json").read_text(encoding="utf-8"))
t, sus = w["telemetry"], w["sustained"]
rows = list(csv.DictReader((W / "thermal-nvsmi.csv").open(encoding="utf-8")))
load = [r for r in rows if r["phase"] == "sustained_load"]

# Times of the last activity of the agents that shared this machine, from the swarm's own records
# (system messages 21:50:33Z / 21:50:35Z for the two Opus workers; board post #43 at 16:50:58 local;
# the monitor session's last_post_time 2026-11-19 17:20:30, board seq 47).
OPUS_WORKERS_STOPPED_LOCAL = "2026/11/19 16:50:33"
LAST_WORKER_POST_LOCAL = "2026/11/19 16:50:58"
STALL_MONITOR_LAST_POST_LOCAL = "2026/11/19 17:20:30"


def f(r, k):
    return float(r[k])


def ev(r):
    return int(r["clocks_event_reasons.active"], 16)


def mean(v, k):
    return sum(f(x, k) for x in v) / len(v)


def ts(r):
    return dt.datetime.strptime(r["timestamp"].strip()[:19], "%Y/%m/%d %H:%M:%S")


t0 = f(load[0], "t_rel_s")
abs0 = ts(load[0])


def rel_of(stamp):
    return (dt.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S") - abs0).total_seconds()


def tr(r):
    return f(r, "t_rel_s") - t0


per_min, mins = [], {}
for r in load:
    mins.setdefault(int(tr(r) // 60), []).append(r)
for m, v in sorted(mins.items()):
    per_min.append({"minute": m, "samples": len(v), "temp_max_c": max(f(x, "temperature.gpu") for x in v),
                    "temp_mean_c": round(mean(v, "temperature.gpu"), 1), "power_mean_w": round(mean(v, "power.draw"), 1),
                    "sm_clock_mean_mhz": round(mean(v, "clocks.sm")), "util_mean_pct": round(mean(v, "utilization.gpu"), 1),
                    "sw_power_cap_samples": sum(1 for x in v if ev(x) & 0x4),
                    "sw_thermal_slowdown_samples": sum(1 for x in v if ev(x) & 0x20),
                    "hw_thermal_slowdown_samples": sum(1 for x in v if ev(x) & 0x40)})


def first(pred, after=0.0):
    for r in load:
        if tr(r) >= after and pred(r):
            return tr(r)
    return None


run87 = cur = 0
run_start = run_end = None
cur_start = None
for r in load:
    if f(r, "temperature.gpu") >= 87:
        if cur == 0:
            cur_start = tr(r)
        cur += 1
        if cur > run87:
            run87, run_start, run_end = cur, cur_start, tr(r)
    else:
        cur = 0
steady = [r for r in load if 120 <= tr(r) < 660]
t84 = first(lambda r: f(r, "temperature.gpu") >= 84, after=120.0)
t85 = first(lambda r: f(r, "temperature.gpu") >= 85)
t87 = first(lambda r: f(r, "temperature.gpu") >= 87)
tsw = first(lambda r: ev(r) & 0x20)
early84 = [r for r in load if tr(r) < 120 and f(r, "temperature.gpu") >= 84]
pre = [r for r in load if t84 - 60 <= tr(r) < t84]
post = [r for r in load if tr(r) >= t87 + 5]
post_a = [r for r in load if t87 + 5 <= tr(r) < t87 + 20]
post_b = [r for r in load if tr(r) >= tr(load[-1]) - 9]
half = len(steady) // 2
drift = (mean(steady[half:], "temperature.gpu") - mean(steady[:half], "temperature.gpu")) / ((tr(steady[-1]) - tr(steady[0])) / 2 / 60)
init = [r for r in rows if r["phase"] == "init"]
after = [r for r in rows if r["phase"] == "post"]
derived = {
    "steady_state_window_s": [120, 660], "steady_temp_mean_c": round(mean(steady, "temperature.gpu"), 1),
    "steady_temp_max_c": max(f(x, "temperature.gpu") for x in steady), "steady_power_mean_w": round(mean(steady, "power.draw"), 1),
    "steady_sm_clock_mean_mhz": round(mean(steady, "clocks.sm"), 1), "steady_util_mean_pct": round(mean(steady, "utilization.gpu"), 1),
    "steady_temp_drift_c_per_min": round(drift, 3),
    "warmup_s_to_80c": round(first(lambda r: f(r, "temperature.gpu") >= 80), 1), "warmup_s_to_83c": round(first(lambda r: f(r, "temperature.gpu") >= 83), 1),
    "early_samples_at_84c_before_t120": len(early84),
    "excursion_first_84c_s": round(t84, 1), "excursion_first_85c_s": round(t85, 1), "excursion_first_87c_s": round(t87, 1),
    "excursion_first_sw_thermal_slowdown_s": round(tsw, 1), "excursion_seconds_from_first_84c_to_87c": round(t87 - t84, 1),
    "last_60s_before_excursion": {"power_mean_w": round(mean(pre, "power.draw"), 1), "sm_clock_mean_mhz": round(mean(pre, "clocks.sm")),
                                  "util_mean_pct": round(mean(pre, "utilization.gpu"), 1), "memory_used_mib_max": max(f(x, "memory.used") for x in pre)},
    "after_excursion": {"power_mean_w": round(mean(post, "power.draw"), 1), "sm_clock_mean_mhz": round(mean(post, "clocks.sm"), 1),
                        "temp_mean_first_15s_c": round(mean(post_a, "temperature.gpu"), 2), "temp_mean_last_9s_c": round(mean(post_b, "temperature.gpu"), 2),
                        "power_mean_last_9s_w": round(mean(post_b, "power.draw"), 1)},
    "longest_run_ge_87c": {"samples": run87, "span_s": round(run_end - run_start, 1)},
    "headroom_c_steady_to_abort_threshold": round(87 - mean(steady, "temperature.gpu"), 1),
    "idle_temp_before_c": f(init[0], "temperature.gpu") if init else None, "idle_temp_after_unload_c": f(after[-1], "temperature.gpu") if after else None,
    "host_activity": {"opus_workers_stopped_at_t_s": round(rel_of(OPUS_WORKERS_STOPPED_LOCAL), 1), "last_worker_board_post_at_t_s": round(rel_of(LAST_WORKER_POST_LOCAL), 1),
                      "first_84c_after_last_post_s": round(t84 - rel_of(LAST_WORKER_POST_LOCAL), 1), "first_84c_after_opus_stop_s": round(t84 - rel_of(OPUS_WORKERS_STOPPED_LOCAL), 1),
                      "stall_monitor_last_post_at_t_s": round(rel_of(STALL_MONITOR_LAST_POST_LOCAL), 1),
                      "cpu_load_of_other_agents": "not recorded",
                      "source": "swarm system messages, board timestamps and the monitor session's last_post_time (board seq 47); whole-second resolution"},
}
ha = derived["host_activity"]
ae = derived["after_excursion"]
gates = {
    "sustained_15min_completed": {
        "threshold": ">= 900 s sustained 8-slot generation with no abort",
        "measured": f"actual {sus['actual_duration_s']} s of {sus['target_duration_s']} s, {sus['rounds']} rounds, {sus['generated_tokens']} tokens, abort_reason={sus['abort_reason']}",
        "passed": sus["abort_reason"] is None and sus["actual_duration_s"] >= 900},
    "no_thermal_slowdown": {
        "threshold": "0 samples with SwThermalSlowdown (0x20) or HwThermalSlowdown (0x40)",
        "measured": f"sw_thermal_samples={t['sw_thermal_samples']}, hw_thermal_samples={t['hw_thermal_samples']} of {t['samples']} (1 s); first SwThermalSlowdown at t+{tsw:.1f} s",
        "passed": t["sw_thermal_samples"] == 0 and t["hw_thermal_samples"] == 0},
    "temp_below_87c_abort_rule": {
        "threshold": "GPU temp never >= 87 C for 30 s (coordinator abort rule, board #27)",
        "measured": f"max {t['temp_max_c']} C; longest run >= 87 C: {run87} consecutive 1 s samples spanning {run_end - run_start:.1f} s; first 87 C at t+{t87:.1f} s",
        "passed": run87 < 30},
    "zero_lingering_compute_vram": {
        "threshold": "0 MiB compute VRAM after exit",
        "measured": f"end compute_vram_mib={s['end_gpu_telemetry']['compute_vram_mib']}, released_after_exit={w['compute_vram_released_after_exit']}; stamper snapshots of free MiB at start/end {s['start_gpu_telemetry']['memory_free_mib']}/{s['end_gpu_telemetry']['memory_free_mib']}",
        "passed": s["end_gpu_telemetry"]["compute_vram_mib"] == 0 and w["compute_vram_released_after_exit"]},
    "peak_vram_le_7372_mib": {
        "threshold": "<= 7372 MiB (7.2 GiB)",
        "measured": f"{t['peak_vram_used_mib']:.0f} MiB over {t['samples']} samples",
        "passed": bool(t["vram_gate_passed"])},
    "stamp_strict_anomalies_clean": {
        "threshold": "mlos stamp-run --zone B --strict-anomalies: clean_hardware_run=true",
        "measured": f"clean_hardware_run={s['clean_hardware_run']}, anomalies={s['anomalies']}, exit_code={s['execution']['exit_code']}. The stamper takes start and end snapshots only and did not see the slowdown during the run.",
        "passed": bool(s["clean_hardware_run"])},
}
(OUT / "010-gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
w2 = dict(w)
w2["per_minute_summary"] = per_min
w2["derived"] = derived
(OUT / "010-thermal-workload-with-minutes.json").write_text(json.dumps(w2, indent=2, sort_keys=True) + "\n", encoding="utf-8")
notes = [
    "Result: the run did NOT complete 15 minutes. It aborted itself at 743 s on the 87 C / 30 s rule and released the GPU cleanly. Three of six gates fail; this packet reports that as the finding.",
    f"Plateau (t+120 s to t+660 s): {derived['steady_temp_mean_c']} C mean, {derived['steady_temp_max_c']:.0f} C max, drifting up by {derived['steady_temp_drift_c_per_min']:.2f} C per minute, at {derived['steady_power_mean_w']} W and {derived['steady_sm_clock_mean_mhz']:.0f} MHz with SwPowerCap (0x4) active on nearly every sample. There were {derived['early_samples_at_84c_before_t120']} samples at 84 C during warm-up. Sustained power settled near {derived['steady_power_mean_w']} W; brief samples reach {t['power_max_w']:.1f} W ({sum(1 for r in rows if f(r, 'power.draw') > 95)} samples above 95 W), so the profile's dgpu_power_cap_w=95 W was touched but not sustained. The stamp records no power limit.",
    f"Excursion: the first 84 C sample after the plateau is at t+{t84:.1f} s and 87 C is reached {t87 - t84:.1f} s later, at t+{t87:.1f} s. SwThermalSlowdown begins at t+{tsw:.1f} s. Power then falls to {ae['power_mean_w']} W and SM clock to {ae['sm_clock_mean_mhz']:.0f} MHz, and the temperature keeps RISING, from {ae['temp_mean_first_15s_c']} C just after onset to {ae['temp_mean_last_9s_c']} C in the last 9 s at {ae['power_mean_last_9s_w']} W. It had not reached equilibrium when the run aborted.",
    f"Nothing recorded changes on the GPU in the 60 s before onset: {derived['last_60s_before_excursion']['power_mean_w']} W, {derived['last_60s_before_excursion']['sm_clock_mean_mhz']} MHz, {derived['last_60s_before_excursion']['util_mean_pct']}% utilisation, {derived['last_60s_before_excursion']['memory_used_mib_max']:.0f} MiB. Whether a second process shared the GPU was not recorded.",
    f"What is recorded on the host: the two Opus workers stopped at t+{ha['opus_workers_stopped_at_t_s']:.0f} s, and the last post from any working agent was at t+{ha['last_worker_board_post_at_t_s']:.0f} s, {ha['first_84c_after_last_post_s']:.0f} s before the first 84 C sample ({ha['first_84c_after_opus_stop_s']:.0f} s after the Opus stop). The stall monitor kept posting to the board until t+{ha['stall_monitor_last_post_at_t_s']:.0f} s, after the run aborted, so host agent activity did not cease at the excursion. The CPU load of those agents is not recorded. That the plateau was measured during their CPU work, and that the excursion follows the end of that work, are inferences from the timing, not measurements.",
    "Interpretation (inference, not measured): a step of several degrees within seconds at constant GPU power, with no recovery after a one-third power cut, is not GPU self-heating at constant cooling. It points to a loss of cooling capacity. A fan curve keyed to CPU temperature would fit the timing: if the workers' CPU load ended when they stopped, the fans may have slowed while the GPU was still at about 79 W. Coincidence fits too. This trace has no CPU temperature, fan speed or power-mode channel, so the cause is undetermined.",
    f"Idle GPU temperature was {derived['idle_temp_before_c']:.0f} C before the run and {derived['idle_temp_after_unload_c']:.0f} C in the last post-run sample, which shows how much heat the chassis was holding.",
    "Recommendation 1: add CPU package temperature, fan RPM, CPU load and the active ASUS/Windows power mode to the thermal sampler before repeating, and keep the machine otherwise idle, or record what else is running.",
    f"Recommendation 2: treat the margin on this chassis under Windows as about {derived['headroom_c_steady_to_abort_threshold']} C between plateau and the abort threshold. Later GPU jobs in this session paused themselves at 85 C repeatedly (20 pauses in one 15-minute probe). Sustained jobs should carry a temperature guard, and dgpu_power_cap_w should stay provisional until a Linux boot can set it with nvidia-smi -pl.",
    "Load: llama-server b12549 (cu12.4), Qwen3.5-4B-Q8_0 with 3 F16 LoRAs, 8 slots x 8192 ctx, 8 concurrent 256-token greedy completions per round alternating three-adapter and W0 requests. Harness gpu_workload_v1.py, the version that ran. Lid position and AC state were not recorded.",
    "Provenance: the worker that started this run stopped on model quota about ten minutes into it. The harness finished and wrote its artifacts with no agent watching. This packet was assembled afterwards from those raw files by the session that took the work over. The run was launched through mlos gpu-lease and mlos stamp-run; the embedded record is the stamp, and no separate lease record is embedded. The raw server log contains a local path and is not shipped; its hash is listed.",
    "Review: recomputed from raw files by two independent reviewers (rounds 1 and 2); every number reproduced. Round 2 corrected the 95 W wording, the CPU-load attribution and the unrecorded second-process claim; those corrections are applied in this revision. The round-2 fixes were applied by the author; a third, narrow review of the revision is recorded in receipts/REVIEW_ROUND_3.md.",
]
cmd = [sys.executable, str(ROOT / "mlos" / "build_receipt.py"),
       "--packet-id", "geminus-to-orbit-010", "--issue", "Moissanite-Labs/ml-os#4 (Phase 3, item 3, Windows-side)",
       "--title", "Sustained RTX 3080 load thermal run, native Windows 11: aborted at 743 s on the 87 C rule; 3 of 6 gates fail",
       "--stamp", str(W / "thermal-stamp.json"), "--workload", str(OUT / "010-thermal-workload-with-minutes.json"),
       "--artifacts", str(W / "thermal-workload.json"), str(W / "thermal-nvsmi.csv"), str(W / "thermal-server.log"),
       str(ROOT / "mlos" / "phase1" / "gpu_workload_v1.py"),
       "--gates", str(OUT / "010-gates.json"), "--out", str(OUT / "geminus-to-orbit-010-mlos-phase3-thermal-windows.json"),
       "--notes", *notes]
sys.exit(subprocess.run(cmd, check=False).returncode)
