"""Deterministic Zone A / Zone B environment enforcement and hardware provenance run stamper."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

from ml_os.adapt import ProfileValidationError, collect_profile_violations, load_profile, sha256_bytes
from ml_os.probe import (
    LINUX_KMD_BASELINE_VRAM_MAX_MIB,
    CommandRunner,
    compute_linux_kmd_baseline_mib,
    default_command_runner,
    extract_graphics_processes,
    probe_nvidia_gpu_processes,
)

STAMP_RECEIPT_SCHEMA = "mlos-run-receipt/v0.1"

ZONE_A_PINS: dict[str, str] = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

ZONE_B_PINS: dict[str, str] = {
    **ZONE_A_PINS,
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TORCH_USE_DETERMINISTIC_ALGORITHMS": "1",
}

# Exact NVML clocks_event_reasons bitmask flags.
NVML_CLOCKS_REASONS_BITS: tuple[tuple[int, str, bool], ...] = (
    (0x0000000000000001, "GpuIdle", False),
    (0x0000000000000002, "ApplicationsClocksSetting", False),
    (0x0000000000000004, "SwPowerCap", True),
    (0x0000000000000008, "HwSlowdown", True),
    (0x0000000000000010, "SyncBoost", False),
    (0x0000000000000020, "SwThermalSlowdown", True),
    (0x0000000000000040, "HwThermalSlowdown", True),
    (0x0000000000000080, "HwPowerBrakeSlowdown", True),
    (0x0000000000000100, "DisplayClockSetting", False),
)

THERMAL_REASON_NAMES: frozenset[str] = frozenset(
    {"SwThermalSlowdown", "HwThermalSlowdown"}
)
SLOWDOWN_REASON_NAMES: frozenset[str] = frozenset(
    {"SwPowerCap", "HwSlowdown", "SwThermalSlowdown", "HwThermalSlowdown", "HwPowerBrakeSlowdown"}
)


class StampAnomalyError(RuntimeError):
    """Raised when --strict-anomalies is set and hardware telemetry detects throttling or VRAM waste."""


def build_deterministic_env(
    zone: str,
    base_env: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Apply Zone A or Zone B determinism pins onto base_env without wiping OS system variables."""
    norm = zone.strip().upper()
    if norm == "A":
        pins = dict(ZONE_A_PINS)
    elif norm == "B":
        pins = dict(ZONE_B_PINS)
    else:
        raise ValueError(f"ERR_STAMP_ZONE: zone={zone!r} must be 'A' or 'B'")

    merged = dict(os.environ if base_env is None else base_env)
    merged.update(pins)
    merged["MLOS_DETERMINISM_ZONE"] = norm
    return merged, pins


def normalize_cpu_isa(raw_machine: str | None = None) -> str | None:
    raw = (raw_machine if raw_machine is not None else platform.machine()).strip()
    if not raw:
        return None
    low = raw.lower()
    if low in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if low in ("arm64", "aarch64"):
        return "aarch64"
    return low


def decode_clocks_event_reasons(raw_value: str | None) -> dict[str, Any]:
    """Decode nvidia-smi clocks_event_reasons.active hex bitmask or symbolic reason string."""
    if raw_value is None:
        return {
            "raw": None,
            "bitmask": None,
            "decoded_reasons": [],
            "thermal_throttled": False,
            "slowdown_active": False,
        }

    cleaned = raw_value.strip()
    if not cleaned or cleaned.lower() in ("none", "n/a", "[not supported]", "not active"):
        return {
            "raw": cleaned or None,
            "bitmask": 0,
            "decoded_reasons": [],
            "thermal_throttled": False,
            "slowdown_active": False,
        }

    if cleaned.lower().startswith("0x"):
        try:
            mask = int(cleaned.split()[0], 16)
        except ValueError:
            mask = None
        if mask is not None:
            reasons = [name for bit, name, _ in NVML_CLOCKS_REASONS_BITS if (mask & bit) != 0]
            thermal = any(r in THERMAL_REASON_NAMES for r in reasons)
            slowdown = any(r in SLOWDOWN_REASON_NAMES for r in reasons)
            return {
                "raw": cleaned,
                "bitmask": mask,
                "decoded_reasons": reasons,
                "thermal_throttled": thermal,
                "slowdown_active": slowdown,
            }

    # Symbolic fallback if nvidia-smi returned named flags.
    tokens = [t.strip() for t in cleaned.replace("|", ",").split(",") if t.strip()]
    thermal_sym = any("thermal" in t.lower() for t in tokens)
    slow_sym = thermal_sym or any(
        kw in t.lower() for t in tokens for kw in ("slowdown", "powercap", "power_cap", "powerbrake")
    )
    return {
        "raw": cleaned,
        "bitmask": None,
        "decoded_reasons": tokens,
        "thermal_throttled": thermal_sym,
        "slowdown_active": slow_sym,
    }


def _parse_float_or_none(val: str) -> float | None:
    cleaned = val.strip()
    if not cleaned or cleaned.lower() in ("n/a", "[not supported]", "none"):
        return None
    token = cleaned.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def _parse_int_or_none(val: str) -> int | None:
    cleaned = val.strip()
    if not cleaned or cleaned.lower() in ("n/a", "[not supported]", "none"):
        return None
    token = cleaned.split()[0]
    try:
        return int(float(token))
    except ValueError:
        return None


def sample_nvidia_telemetry(
    runner: CommandRunner = default_command_runner,
    *,
    host_os: str | None = None,
) -> dict[str, Any] | None:
    """Capture a single real nvidia-smi CSV snapshot + compute vs desktop VRAM breakdown."""
    rc, out, _ = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,vbios_version,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,memory.free,clocks_event_reasons.active",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0 or not out.strip():
        return None

    line = out.strip().splitlines()[0]
    cols = [c.strip() for c in line.split(",")]
    if len(cols) < 12:
        return None

    rc_apps, out_apps, _ = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    compute_vram = 0
    compute_apps: list[dict[str, Any]] = []
    if rc_apps == 0 and out_apps.strip():
        for app_line in out_apps.strip().splitlines():
            acols = [c.strip() for c in app_line.split(",")]
            if len(acols) < 3:
                continue
            pid_v = _parse_int_or_none(acols[0])
            mem_v = _parse_int_or_none(acols[2])
            if pid_v is None:
                continue
            mib = max(0, mem_v if mem_v is not None else 0)
            compute_vram += mib
            compute_apps.append(
                {
                    "pid": pid_v,
                    "process_name": acols[1] or None,
                    "used_gpu_memory_mib": mib,
                }
            )

    procs_queried, all_gpu_procs = probe_nvidia_gpu_processes(runner=runner)
    graphics_procs = extract_graphics_processes(all_gpu_procs)

    mem_used = _parse_int_or_none(cols[9])
    mem_free = _parse_int_or_none(cols[10])
    desktop_vram = max(0, mem_used - compute_vram) if mem_used is not None else None
    kmd_baseline_mib = compute_linux_kmd_baseline_mib(
        host_os=host_os,
        desktop_vram_mib=desktop_vram,
        gpu_processes_queried=procs_queried,
        graphics_processes=graphics_procs,
    )
    clocks_decoded = decode_clocks_event_reasons(cols[11])

    return {
        "gpu_name": cols[0] or None,
        "driver_version": cols[1] or None,
        "vbios_version": cols[2] or None,
        "pstate": cols[3] or None,
        "clocks_sm_mhz": _parse_int_or_none(cols[4]),
        "clocks_mem_mhz": _parse_int_or_none(cols[5]),
        "power_draw_w": _parse_float_or_none(cols[6]),
        "power_limit_w": _parse_float_or_none(cols[7]),
        "temperature_gpu_c": _parse_int_or_none(cols[8]),
        "memory_used_mib": mem_used,
        "memory_free_mib": mem_free,
        "compute_vram_mib": compute_vram,
        "non_compute_desktop_vram_mib": desktop_vram,
        "linux_kmd_baseline_vram_mib": kmd_baseline_mib,
        "gpu_processes_queried": procs_queried,
        "graphics_processes": graphics_procs,
        "compute_processes": compute_apps,
        "clocks_event_reasons": clocks_decoded,
    }


def _read_git_head(cwd: Path | None, runner: CommandRunner) -> str | None:
    cmd = ["git"]
    if cwd is not None:
        cmd.extend(["-C", str(cwd)])
    cmd.extend(["rev-parse", "HEAD"])
    rc, out, _ = runner(cmd)
    if rc != 0:
        return None
    sha = out.strip()
    if len(sha) == 40 and all(ch in "0123456789abcdef" for ch in sha.lower()):
        return sha.lower()
    return None


def evaluate_telemetry_anomalies(
    *,
    start_gpu: dict[str, Any] | None,
    end_gpu: dict[str, Any] | None,
    max_desktop_vram_mib: int = 0,
    host_os: str | None = None,
    max_linux_kmd_baseline_mib: int = LINUX_KMD_BASELINE_VRAM_MAX_MIB,
) -> list[str]:
    """Detect desktop/WDDM VRAM waste, active desktop GPU processes, throttling, and lingering VRAM.

    On Linux, NVIDIA KMD drivers can reserve <= 4 MiB of internal driver metadata with zero
    active graphics processes. That baseline is accepted only when `host_os == 'linux'`,
    `nvidia-smi -q -d PIDS` succeeded (`gpu_processes_queried is True`), zero graphics or
    hybrid (`G` / `C+G`) processes are attached, and `non_compute_desktop_vram_mib <= 4`.
    Any real graphics/desktop process or unverified VRAM overhead fails closed.
    """
    anomalies: list[str] = []
    norm_os = (host_os if host_os is not None else platform.system()).strip().lower()

    for phase_label, snap in (("start", start_gpu), ("end", end_gpu)):
        if snap is None:
            continue
        graphics_procs = snap.get("graphics_processes")
        if isinstance(graphics_procs, list) and len(graphics_procs) > 0:
            anomalies.append(
                f"DESKTOP_GPU_PROCESS_ACTIVE:{phase_label}:{len(graphics_procs)}proc"
            )

        desktop_mib = snap.get("non_compute_desktop_vram_mib")
        if type(desktop_mib) is int and desktop_mib > max_desktop_vram_mib:
            is_verified_linux_kmd_baseline = (
                norm_os == "linux"
                and snap.get("gpu_processes_queried") is True
                and isinstance(graphics_procs, list)
                and len(graphics_procs) == 0
                and type(max_linux_kmd_baseline_mib) is int
                and max_linux_kmd_baseline_mib >= 0
                and desktop_mib <= max_linux_kmd_baseline_mib
            )
            if not is_verified_linux_kmd_baseline:
                anomalies.append(
                    f"WDDM_OR_DESKTOP_VRAM_NONZERO:{phase_label}:{desktop_mib}MiB>{max_desktop_vram_mib}MiB"
                )
        reasons = snap.get("clocks_event_reasons") or {}
        if reasons.get("thermal_throttled") is True:
            anomalies.append(
                f"THERMAL_THROTTLE_ACTIVE:{phase_label}:{','.join(reasons.get('decoded_reasons', []))}"
            )
        elif reasons.get("slowdown_active") is True:
            anomalies.append(
                f"CLOCKS_SLOWDOWN_ACTIVE:{phase_label}:{','.join(reasons.get('decoded_reasons', []))}"
            )

    if end_gpu is not None:
        end_compute = end_gpu.get("compute_vram_mib")
        if type(end_compute) is int and end_compute > 0:
            anomalies.append(f"LINGERING_COMPUTE_VRAM_NONZERO:end:{end_compute}MiB")

    return anomalies


def stamp_run(
    *,
    command: list[str],
    profile_path: Path,
    zone: str = "B",
    output_path: Path | None = None,
    cwd: Path | None = None,
    require_gpu: bool = False,
    strict_anomalies: bool = False,
    runner: CommandRunner = default_command_runner,
    base_env: dict[str, str] | None = None,
    host_os: str | None = None,
) -> dict[str, Any]:
    """Execute command under deterministic Zone A/B pins and emit a SHA-256 provenance receipt."""
    if not command:
        raise ValueError("ERR_STAMP_CMD_EMPTY: command to stamp must not be empty")

    profile, profile_sha = load_profile(profile_path)
    violations = collect_profile_violations(profile)
    if violations:
        raise ProfileValidationError(violations)

    exec_env, applied_pins = build_deterministic_env(zone, base_env=base_env)
    start_gpu = sample_nvidia_telemetry(runner=runner, host_os=host_os)
    if require_gpu and start_gpu is None:
        raise StampAnomalyError("ERR_STAMP_GPU_MISSING: --require-gpu set, and nvidia-smi is unavailable")

    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=exec_env,
        check=False,
    )
    elapsed_ms = round((time.monotonic() - t0) * 1000.0, 3)
    ended_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    end_gpu = sample_nvidia_telemetry(runner=runner, host_os=host_os)
    desktop_cap = int(profile["ring0_locked"]["dgpu_desktop_vram_mib"])
    anomalies = evaluate_telemetry_anomalies(
        start_gpu=start_gpu,
        end_gpu=end_gpu,
        max_desktop_vram_mib=desktop_cap,
        host_os=host_os,
    )

    body: dict[str, Any] = {
        "schema": STAMP_RECEIPT_SCHEMA,
        "profile_id": str(profile["profile_id"]),
        "profile_sha256": profile_sha,
        "determinism_zone": zone.strip().upper(),
        "determinism_pins": applied_pins,
        "host": {
            "cpu_isa": normalize_cpu_isa(),
            "host_os": platform.system().lower() or None,
            "os_release": platform.release() or None,
            "python_version": platform.python_version(),
        },
        "git_head": _read_git_head(cwd, runner=runner),
        "execution": {
            "command_argc": len(command),
            "command_exe": Path(command[0]).name,
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "duration_ms": elapsed_ms,
            "exit_code": proc.returncode,
        },
        "gpu_telemetry_available": start_gpu is not None or end_gpu is not None,
        "start_gpu_telemetry": start_gpu,
        "end_gpu_telemetry": end_gpu,
        "anomalies": anomalies,
        "clean_hardware_run": len(anomalies) == 0 and proc.returncode == 0,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["receipt_sha256"] = sha256_bytes(canonical)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".mlos-receipt-",
            suffix=".json",
            dir=str(output_path.parent),
        )
        tmp_file = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write((json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_file, output_path)
        except OSError:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
            raise

    if strict_anomalies and anomalies:
        raise StampAnomalyError(f"ERR_STAMP_HARDWARE_ANOMALY: {'; '.join(anomalies)}")

    return body
