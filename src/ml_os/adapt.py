"""Ring 0 safety core validator and Ring 2 safe adaptation engine (mlos-adapt)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any

from ml_os import PROFILE_SCHEMA

REQUIRED_LSMS: tuple[str, ...] = (
    "landlock",
    "lockdown",
    "yama",
    "integrity",
    "apparmor",
    "bpf",
)

REQUIRED_READONLY_MOUNTS: tuple[str, ...] = (
    "/var/lib/mlos/models",
    "/etc/mlos/canaries",
    "/usr",
)

REQUIRED_BTRFS_SUBVOLUMES: tuple[str, ...] = (
    "@",
    "@snapshots",
    "@models",
    "@experiments",
    "@ledger",
)

FORBIDDEN_PRIVILEGE_PACKAGES: frozenset[str] = frozenset(
    {
        "sudo",
        "sudo-rs",
        "opendoas",
        "doas",
        "polkit",
        "pkexec",
    }
)

ALLOWED_DISPLAY_MODES: frozenset[str] = frozenset(
    {
        "headless-ssh",
        "dual-screen-igpu",
        "single-primary-igpu",
    }
)

ALLOWED_REMOTE_TUNNELS: frozenset[str] = frozenset(
    {
        "tailscale",
        "wireguard",
        "ssh-only",
    }
)

ALLOWED_SCREENPAD_ROLES: frozenset[str] = frozenset(
    {
        "ips-telemetry-strip",
        "secondary-workspace-igpu",
        "disabled",
    }
)

IMMUTABLE_TOP_SECTIONS: tuple[str, ...] = (
    "schema",
    "profile_id",
    "role",
    "base_distro",
    "distro_pin",
    "hardware",
    "ring0_locked",
    "slices",
    "thermal_profiles",
)


class ProfileValidationError(ValueError):
    """Raised when an mlos-profile/v0.1 document violates Ring 0 or schema invariants."""

    def __init__(self, violations: list[str]) -> None:
        if not violations:
            violations = ["ERR_PROFILE_INVALID: unspecified validation failure"]
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))


class AdaptationRejectedError(ProfileValidationError):
    """Raised when a proposed Ring 2 adaptation weakens Ring 0 or mutates locked sections."""


@dataclass(frozen=True)
class ValidationReport:
    """Structured result from Ring 0 and schema validation."""

    passed: bool
    profile_id: str
    profile_sha256: str
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class AdaptationCheckReport:
    """Structured result from checking a candidate profile against a locked base profile."""

    passed: bool
    base_sha256: str
    candidate_sha256: str
    changed_ring2_paths: tuple[str, ...]
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "base_sha256": self.base_sha256,
            "candidate_sha256": self.candidate_sha256,
            "changed_ring2_paths": list(self.changed_ring2_paths),
            "violations": list(self.violations),
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_profile_bytes(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProfileValidationError([f"ERR_PROFILE_UTF8: invalid UTF-8 bytes: {exc}"]) from exc
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileValidationError([f"ERR_PROFILE_TOML: invalid TOML syntax: {exc}"]) from exc
    if not isinstance(parsed, dict):
        raise ProfileValidationError(["ERR_PROFILE_ROOT: root TOML value must be a table"])
    return parsed


def load_profile(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return load_profile_bytes(raw), sha256_bytes(raw)


def _is_strict_bool(value: Any) -> bool:
    return type(value) is bool


def _is_strict_int(value: Any) -> bool:
    return type(value) is int


def collect_profile_violations(profile: dict[str, Any]) -> list[str]:
    """Return every Ring 0, Ring 1, and Ring 2 violation in a parsed profile."""
    violations: list[str] = []

    if profile.get("schema") != PROFILE_SCHEMA:
        violations.append(
            f"ERR_SCHEMA_VERSION: expected schema={PROFILE_SCHEMA!r}, got {profile.get('schema')!r}"
        )

    for top_str in ("profile_id", "role", "base_distro"):
        val = profile.get(top_str)
        if not isinstance(val, str) or not val.strip():
            violations.append(f"ERR_TOP_FIELD: {top_str} must be a non-empty string")

    for section in ("distro_pin", "hardware", "ring0_locked", "slices", "thermal_profiles", "ring2_user"):
        if not isinstance(profile.get(section), dict):
            violations.append(f"ERR_MISSING_SECTION: [{section}] table is required")

    ring0 = profile.get("ring0_locked")
    if isinstance(ring0, dict):
        _validate_ring0_locked(ring0, violations)

    hardware = profile.get("hardware")
    if isinstance(hardware, dict):
        _validate_hardware_section(hardware, violations)

    distro_pin = profile.get("distro_pin")
    if isinstance(distro_pin, dict):
        _validate_distro_pin(distro_pin, violations)

    slices = profile.get("slices")
    if isinstance(slices, dict):
        _validate_slices(slices, hardware if isinstance(hardware, dict) else {}, violations)

    thermal_profiles = profile.get("thermal_profiles")
    if isinstance(thermal_profiles, dict):
        _validate_thermal_profiles(
            thermal_profiles,
            hardware if isinstance(hardware, dict) else {},
            violations,
        )

    ring2 = profile.get("ring2_user")
    if isinstance(ring2, dict):
        _validate_ring2_user(
            ring2,
            hardware if isinstance(hardware, dict) else {},
            thermal_profiles if isinstance(thermal_profiles, dict) else {},
            violations,
        )

    for top_key in profile:
        if top_key not in (*IMMUTABLE_TOP_SECTIONS, "ring2_user"):
            violations.append(f"ERR_UNKNOWN_TOP_KEY: unexpected top-level key {top_key!r}")

    return violations


def _validate_ring0_locked(ring0: dict[str, Any], violations: list[str]) -> None:
    bool_invariants: tuple[tuple[str, bool, str], ...] = (
        ("allow_sudo", False, "ERR_RING0_SUDO_LOCKED: allow_sudo must be strictly false"),
        (
            "tool_executor_required",
            True,
            "ERR_RING0_SANDBOX_LOCKED: tool_executor_required must be strictly true",
        ),
        (
            "exclusive_gpu_lease",
            True,
            "ERR_RING0_GPU_LEASE_LOCKED: exclusive_gpu_lease must be strictly true",
        ),
        (
            "require_loopback_inference",
            True,
            "ERR_RING0_LOOPBACK_LOCKED: require_loopback_inference must be strictly true",
        ),
        (
            "require_uki_rollback",
            True,
            "ERR_RING0_UKI_ROLLBACK_LOCKED: require_uki_rollback must be strictly true",
        ),
        (
            "bare_metal_repartition_unlocked",
            False,
            "ERR_RING0_REPARTITION_LOCKED: bare_metal_repartition_unlocked must remain false",
        ),
    )
    for key, expected, err_msg in bool_invariants:
        val = ring0.get(key)
        if not _is_strict_bool(val) or val is not expected:
            violations.append(err_msg)

    desktop_vram = ring0.get("dgpu_desktop_vram_mib")
    if not _is_strict_int(desktop_vram) or desktop_vram != 0:
        violations.append(
            "ERR_RING0_DGPU_VRAM_LOCKED: dgpu_desktop_vram_mib must be exact integer 0"
        )

    lsm_list = ring0.get("lsm_list")
    if not isinstance(lsm_list, list) or not all(isinstance(x, str) for x in lsm_list):
        violations.append("ERR_RING0_LSM_LIST: lsm_list must be a list of strings")
    else:
        missing_lsms = [lsm for lsm in REQUIRED_LSMS if lsm not in lsm_list]
        if missing_lsms:
            violations.append(
                f"ERR_RING0_LSM_MISSING: missing mandatory LSM flags: {', '.join(missing_lsms)}"
            )

    kparams = ring0.get("required_kernel_params")
    if not isinstance(kparams, list) or not all(isinstance(x, str) for x in kparams):
        violations.append("ERR_RING0_KERNEL_PARAMS: required_kernel_params must be a list of strings")
    else:
        joined = " ".join(kparams)
        if "rootfstype=btrfs" not in kparams:
            violations.append("ERR_RING0_BTRFS_ROOT: required_kernel_params must include rootfstype=btrfs")
        if "subvol=@" not in joined:
            violations.append("ERR_RING0_ROOT_SUBVOL: required_kernel_params must mount subvol=@")
        for bad_param in ("apparmor=0", "selinux=0", "mitigations=off", "lsm="):
            if any(p == bad_param or (bad_param == "lsm=" and p == "lsm=") for p in kparams):
                violations.append(f"ERR_RING0_WEAK_KPARAM: forbidden kernel parameter {bad_param!r}")

    ro_mounts = ring0.get("readonly_mounts")
    if not isinstance(ro_mounts, list) or not all(isinstance(x, str) for x in ro_mounts):
        violations.append("ERR_RING0_RO_MOUNTS: readonly_mounts must be a list of paths")
    else:
        for req_mount in REQUIRED_READONLY_MOUNTS:
            if req_mount not in ro_mounts:
                violations.append(
                    f"ERR_RING0_RO_MOUNT_MISSING: {req_mount} must remain read-only in readonly_mounts"
                )

    subvols = ring0.get("btrfs_required_subvolumes")
    if not isinstance(subvols, list) or not all(isinstance(x, str) for x in subvols):
        violations.append("ERR_RING0_BTRFS_SUBVOLS: btrfs_required_subvolumes must be a list of names")
    else:
        for req_sub in REQUIRED_BTRFS_SUBVOLUMES:
            if req_sub not in subvols:
                violations.append(
                    f"ERR_RING0_SUBVOL_MISSING: Btrfs subvolume {req_sub!r} is required for rollback safety"
                )

    ancestors_mode = ring0.get("protected_ancestors_mode")
    if ancestors_mode != "0755":
        violations.append("ERR_RING0_ANCESTORS_MODE: protected_ancestors_mode must be '0755' (go-w)")


def _validate_distro_pin(distro_pin: dict[str, Any], violations: list[str]) -> None:
    if distro_pin.get("builder") != "mkosi":
        violations.append("ERR_DISTRO_BUILDER: distro_pin.builder must be 'mkosi'")
    snap = distro_pin.get("arch_snapshot_date")
    if not isinstance(snap, str) or len(snap) != 10 or snap[4] != "-" or snap[7] != "-":
        violations.append("ERR_DISTRO_SNAPSHOT: arch_snapshot_date must be YYYY-MM-DD")
    lean_tc = distro_pin.get("lean_toolchain")
    if not isinstance(lean_tc, str) or not lean_tc.startswith("leanprover/lean4:v4."):
        violations.append("ERR_DISTRO_LEAN_PIN: lean_toolchain must pin leanprover/lean4:v4.x")
    min_build = distro_pin.get("llama_cpp_min_build")
    if not _is_strict_int(min_build) or min_build < 11146:
        violations.append("ERR_DISTRO_LLAMA_BUILD: llama_cpp_min_build must be >= 11146")


def _validate_hardware_section(hardware: dict[str, Any], violations: list[str]) -> None:
    for pos_int_key in ("cpu_cores", "cpu_threads", "ram_mib", "ram_gib", "zram_gib", "nvme_capacity_gb"):
        val = hardware.get(pos_int_key)
        if not _is_strict_int(val) or val <= 0:
            violations.append(f"ERR_HARDWARE_INT: hardware.{pos_int_key} must be a positive integer")

    dgpu_present = hardware.get("dgpu_present")
    if not _is_strict_bool(dgpu_present):
        violations.append("ERR_HARDWARE_DGPU_FLAG: hardware.dgpu_present must be boolean")
    elif dgpu_present:
        vram = hardware.get("dgpu_vram_mib")
        if not _is_strict_int(vram) or vram <= 0:
            violations.append("ERR_HARDWARE_VRAM: dgpu_vram_mib must be > 0 when dgpu_present is true")
        igpu_pci = hardware.get("igpu_pci")
        dgpu_pci = hardware.get("dgpu_pci")
        if not isinstance(igpu_pci, str) or not isinstance(dgpu_pci, str) or igpu_pci == dgpu_pci:
            violations.append(
                "ERR_HARDWARE_PCI_SPLIT: igpu_pci and dgpu_pci must be distinct PCI addresses"
            )
        drm_by_path = hardware.get("igpu_drm_by_path")
        if not isinstance(drm_by_path, str) or str(igpu_pci) not in drm_by_path:
            violations.append(
                "ERR_HARDWARE_DRM_PATH: igpu_drm_by_path must reference the Intel/iGPU PCI address, never dGPU"
            )

    displays = hardware.get("displays")
    if not isinstance(displays, dict):
        violations.append("ERR_HARDWARE_DISPLAYS: [hardware.displays] table is required")
    else:
        dual = displays.get("dual_screen")
        if not _is_strict_bool(dual):
            violations.append("ERR_HARDWARE_DUAL_SCREEN: hardware.displays.dual_screen must be boolean")
        elif dual:
            p_conn = displays.get("primary_connector")
            s_conn = displays.get("secondary_connector")
            if not isinstance(p_conn, str) or not isinstance(s_conn, str) or p_conn == s_conn:
                violations.append(
                    "ERR_DISPLAY_CONNECTORS: dual-screen hardware requires distinct primary and secondary connectors"
                )


def _validate_slices(
    slices: dict[str, Any],
    hardware: dict[str, Any],
    violations: list[str],
) -> None:
    ram_gib = hardware.get("ram_gib") if _is_strict_int(hardware.get("ram_gib")) else 32
    for req_slice in ("formal", "infer", "compute", "agents"):
        cfg = slices.get(req_slice)
        if not isinstance(cfg, dict):
            violations.append(f"ERR_SLICE_MISSING: [slices.{req_slice}] is required")
            continue
        unit = cfg.get("unit")
        expected_unit = f"mlos-{req_slice}.slice"
        if unit != expected_unit:
            violations.append(f"ERR_SLICE_UNIT: slices.{req_slice}.unit must be {expected_unit!r}")
        mem_high = cfg.get("memory_high_gib")
        mem_max = cfg.get("memory_max_gib")
        if not _is_strict_int(mem_high) or not _is_strict_int(mem_max) or mem_high <= 0 or mem_max < mem_high:
            violations.append(
                f"ERR_SLICE_MEMORY: slices.{req_slice} requires 0 < memory_high_gib <= memory_max_gib"
            )
        elif mem_max >= ram_gib:
            violations.append(
                f"ERR_SLICE_UNCAPPED: slices.{req_slice}.memory_max_gib ({mem_max}) must stay below host ram_gib ({ram_gib})"
            )

    formal = slices.get("formal")
    if isinstance(formal, dict):
        if formal.get("allow_dgpu") is not False:
            violations.append("ERR_FORMAL_SLICE_DGPU: mlos-formal.slice must enforce allow_dgpu = false (0 MiB VRAM)")
        if formal.get("oom_policy") != "kill":
            violations.append("ERR_FORMAL_SLICE_OOM: mlos-formal.slice must set oom_policy = 'kill'")

    agents = slices.get("agents")
    if isinstance(agents, dict) and agents.get("allow_dgpu") is not False:
        violations.append("ERR_AGENTS_SLICE_DGPU: mlos-agents.slice must enforce allow_dgpu = false")


def _validate_thermal_profiles(
    thermal_profiles: dict[str, Any],
    hardware: dict[str, Any],
    violations: list[str],
) -> None:
    if "quiet-edit" not in thermal_profiles:
        violations.append("ERR_THERMAL_QUIET_MISSING: [thermal_profiles.quiet-edit] is required")
    dgpu_present = hardware.get("dgpu_present") is True
    max_power = hardware.get("dgpu_max_power_w", 0)
    aas_lid_required = bool((hardware.get("displays") or {}).get("aas_plus_intake_requires_lid_open"))

    if dgpu_present:
        for req_tp in ("quiet-edit", "sustained-train", "benchmark-locked"):
            if req_tp not in thermal_profiles:
                violations.append(f"ERR_THERMAL_PROFILE_MISSING: [thermal_profiles.{req_tp}] is required for dGPU hosts")

    for name, tp in thermal_profiles.items():
        if not isinstance(tp, dict):
            violations.append(f"ERR_THERMAL_TABLE: thermal_profiles.{name} must be a table")
            continue
        cap = tp.get("dgpu_power_cap_w")
        if not _is_strict_int(cap) or cap < 0:
            violations.append(f"ERR_THERMAL_POWER_CAP: thermal_profiles.{name}.dgpu_power_cap_w must be >= 0")
        elif dgpu_present and _is_strict_int(max_power) and (cap < 50 or cap > max_power):
            violations.append(
                f"ERR_THERMAL_POWER_RANGE: {name} dgpu_power_cap_w ({cap} W) outside safe [50, {max_power}] W range"
            )
        if name == "benchmark-locked" and dgpu_present:
            if tp.get("require_owner_clock_lock_approval") is not True:
                violations.append(
                    "ERR_THERMAL_CLOCK_GATE: benchmark-locked must require owner_clock_lock_approved"
                )
        if aas_lid_required and name in ("sustained-train", "benchmark-locked"):
            if tp.get("require_lid_open") is not True:
                violations.append(
                    f"ERR_AAS_PLUS_LID_GATE: {name} on AAS Plus dual-screen hardware must set require_lid_open = true"
                )
            if tp.get("lid_closed_fallback_profile") != "quiet-edit":
                violations.append(
                    f"ERR_AAS_PLUS_FALLBACK: {name} must fall back to 'quiet-edit' when lid is closed"
                )


def _validate_ring2_user(
    ring2: dict[str, Any],
    hardware: dict[str, Any],
    thermal_profiles: dict[str, Any],
    violations: list[str],
) -> None:
    allowed_ring2_keys = {
        "display_mode",
        "thermal_profile",
        "dgpu_power_cap_w",
        "owner_clock_lock_approved",
        "remote_tunnel",
        "active_packs",
        "extra_packages",
        "dual_screen",
    }
    for key in ring2:
        if key not in allowed_ring2_keys:
            violations.append(f"ERR_RING2_UNKNOWN_KEY: unknown or forbidden Ring 2 key {key!r}")

    display_mode = ring2.get("display_mode")
    if display_mode not in ALLOWED_DISPLAY_MODES:
        violations.append(
            f"ERR_RING0_DGPU_DISPLAY_FORBIDDEN: display_mode={display_mode!r} not in {sorted(ALLOWED_DISPLAY_MODES)}"
        )

    tunnel = ring2.get("remote_tunnel")
    if tunnel not in ALLOWED_REMOTE_TUNNELS:
        violations.append(
            f"ERR_RING2_TUNNEL_INVALID: remote_tunnel={tunnel!r} not in {sorted(ALLOWED_REMOTE_TUNNELS)}"
        )

    tp_name = ring2.get("thermal_profile")
    if not isinstance(tp_name, str) or tp_name not in thermal_profiles:
        violations.append(f"ERR_RING2_THERMAL_UNKNOWN: thermal_profile={tp_name!r} is not defined")
    else:
        tp = thermal_profiles[tp_name]
        if isinstance(tp, dict):
            owner_ok = ring2.get("owner_clock_lock_approved")
            if not _is_strict_bool(owner_ok):
                violations.append("ERR_RING2_CLOCK_BOOL: owner_clock_lock_approved must be boolean")
            elif tp.get("lock_sm_clocks") is True and tp.get("require_owner_clock_lock_approval") is True and not owner_ok:
                violations.append(
                    "ERR_RING2_CLOCK_LOCK_UNAPPROVED: benchmark-locked SM clock locking requires owner_clock_lock_approved = true"
                )

    cap = ring2.get("dgpu_power_cap_w")
    dgpu_present = hardware.get("dgpu_present") is True
    max_power = hardware.get("dgpu_max_power_w", 0)
    if not _is_strict_int(cap) or cap < 0:
        violations.append("ERR_RING2_POWER_CAP: ring2_user.dgpu_power_cap_w must be a non-negative integer")
    elif dgpu_present and _is_strict_int(max_power) and (cap < 50 or cap > max_power):
        violations.append(
            f"ERR_RING2_POWER_CEILING: ring2_user.dgpu_power_cap_w ({cap} W) outside [50, {max_power}] W ceiling"
        )
    elif not dgpu_present and cap != 0:
        violations.append("ERR_RING2_POWER_NODGPU: dgpu_power_cap_w must be 0 when dgpu_present is false")

    extra_pkgs = ring2.get("extra_packages", [])
    if not isinstance(extra_pkgs, list) or not all(isinstance(p, str) for p in extra_pkgs):
        violations.append("ERR_RING2_PACKAGES_TYPE: extra_packages must be a list of package names")
    else:
        for pkg in extra_pkgs:
            norm = pkg.strip().lower()
            if not norm or any(ch.isspace() or ch in "/;|$`" for ch in norm):
                violations.append(f"ERR_RING2_PACKAGE_UNSAFE: invalid package token {pkg!r}")
            if norm in FORBIDDEN_PRIVILEGE_PACKAGES:
                violations.append(
                    f"ERR_RING0_SUDO_FORBIDDEN: privilege escalation package {pkg!r} is banned by Ring 0"
                )

    packs = ring2.get("active_packs", [])
    if not isinstance(packs, list) or not all(isinstance(p, str) and p.strip() for p in packs):
        violations.append("ERR_RING2_PACKS_TYPE: active_packs must be a list of pack identifiers")

    dual_cfg = ring2.get("dual_screen")
    if not isinstance(dual_cfg, dict):
        violations.append("ERR_RING2_DUAL_SCREEN_MISSING: [ring2_user.dual_screen] table is required")
    else:
        if dual_cfg.get("fallback_to_headless_on_dgpu_leak") is not True:
            violations.append(
                "ERR_RING0_DGPU_FALLBACK_REQUIRED: dual_screen.fallback_to_headless_on_dgpu_leak must be true"
            )
        if dual_cfg.get("bind_touch_per_connector") is not True:
            violations.append(
                "ERR_DUAL_SCREEN_TOUCH_MAP: dual_screen.bind_touch_per_connector must be true to prevent cross-panel touch drift"
            )
        role = dual_cfg.get("screenpad_role")
        if role not in ALLOWED_SCREENPAD_ROLES:
            violations.append(
                f"ERR_DUAL_SCREEN_ROLE: screenpad_role={role!r} not in {sorted(ALLOWED_SCREENPAD_ROLES)}"
            )
        blank_sec = dual_cfg.get("oled_idle_blank_sec")
        primary_panel = (hardware.get("displays") or {}).get("primary_panel_type")
        if not _is_strict_int(blank_sec) or blank_sec < 0:
            violations.append("ERR_DUAL_SCREEN_OLED_BLANK: oled_idle_blank_sec must be a non-negative integer")
        elif primary_panel == "oled" and display_mode != "headless-ssh" and (blank_sec == 0 or blank_sec > 600):
            violations.append(
                "ERR_OLED_BURNIN_GUARD: active OLED display modes require 1 <= oled_idle_blank_sec <= 600"
            )


def validate_profile(profile_path: Path) -> ValidationReport:
    """Parse and validate a profile file on disk without modifying any file."""
    raw = profile_path.read_bytes()
    digest = sha256_bytes(raw)
    try:
        parsed = load_profile_bytes(raw)
    except ProfileValidationError as exc:
        return ValidationReport(
            passed=False,
            profile_id="invalid",
            profile_sha256=digest,
            violations=tuple(exc.violations),
        )
    violations = collect_profile_violations(parsed)
    profile_id = str(parsed.get("profile_id", "unknown"))
    return ValidationReport(
        passed=len(violations) == 0,
        profile_id=profile_id,
        profile_sha256=digest,
        violations=tuple(violations),
    )


def _diff_nested(prefix: str, left: Any, right: Any) -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        diffs: list[str] = []
        all_keys = sorted(set(left.keys()) | set(right.keys()))
        for key in all_keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                diffs.append(child)
            else:
                diffs.extend(_diff_nested(child, left[key], right[key]))
        return diffs
    if left != right:
        return [prefix]
    return []


def check_adaptation(base_path: Path, candidate_path: Path) -> AdaptationCheckReport:
    """Verify a candidate profile only adapts [ring2_user] and keeps all Ring 0 rules intact.

    Performs strictly zero disk writes.
    """
    base_raw = base_path.read_bytes()
    cand_raw = candidate_path.read_bytes()
    base_sha = sha256_bytes(base_raw)
    cand_sha = sha256_bytes(cand_raw)

    violations: list[str] = []
    try:
        base = load_profile_bytes(base_raw)
    except ProfileValidationError as exc:
        return AdaptationCheckReport(
            passed=False,
            base_sha256=base_sha,
            candidate_sha256=cand_sha,
            changed_ring2_paths=(),
            violations=tuple(f"BASE_{v}" for v in exc.violations),
        )

    base_violations = collect_profile_violations(base)
    if base_violations:
        return AdaptationCheckReport(
            passed=False,
            base_sha256=base_sha,
            candidate_sha256=cand_sha,
            changed_ring2_paths=(),
            violations=tuple(f"BASE_{v}" for v in base_violations),
        )

    try:
        candidate = load_profile_bytes(cand_raw)
    except ProfileValidationError as exc:
        return AdaptationCheckReport(
            passed=False,
            base_sha256=base_sha,
            candidate_sha256=cand_sha,
            changed_ring2_paths=(),
            violations=tuple(exc.violations),
        )

    violations.extend(collect_profile_violations(candidate))

    for locked_section in IMMUTABLE_TOP_SECTIONS:
        section_diffs = _diff_nested(
            locked_section,
            base.get(locked_section),
            candidate.get(locked_section),
        )
        for path in section_diffs:
            violations.append(
                f"ERR_RING0_IMMUTABLE_SECTION: adaptation attempted to mutate locked path {path!r}"
            )

    for extra_key in sorted(set(candidate.keys()) - set(base.keys())):
        if extra_key != "ring2_user":
            violations.append(
                f"ERR_RING0_IMMUTABLE_SECTION: adaptation attempted to inject top-level key {extra_key!r}"
            )

    ring2_changes = _diff_nested("ring2_user", base.get("ring2_user"), candidate.get("ring2_user"))
    dedup_violations = list(dict.fromkeys(violations))
    return AdaptationCheckReport(
        passed=len(dedup_violations) == 0,
        base_sha256=base_sha,
        candidate_sha256=cand_sha,
        changed_ring2_paths=tuple(ring2_changes),
        violations=tuple(dedup_violations),
    )


def apply_adaptation(
    *,
    active_profile_path: Path,
    candidate_profile_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    """Validate candidate profile fail-closed; on pass, atomically update profile and append ledger receipt.

    Guarantees zero disk writes anywhere if check_adaptation fails.
    """
    report = check_adaptation(active_profile_path, candidate_profile_path)
    if not report.passed:
        raise AdaptationRejectedError(list(report.violations))

    cand_bytes = candidate_profile_path.read_bytes()
    prev_entry_sha = "0" * 64
    if ledger_path.exists():
        existing_lines = [
            ln.strip()
            for ln in ledger_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if existing_lines:
            last_obj = json.loads(existing_lines[-1])
            prev_entry_sha = str(last_obj.get("entry_sha256", prev_entry_sha))

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ckpt_label = f"adapt-{report.candidate_sha256[:12]}"
    payload: dict[str, Any] = {
        "schema": "mlos-adaptation-receipt/v0.1",
        "timestamp_utc": timestamp,
        "prev_profile_sha256": report.base_sha256,
        "new_profile_sha256": report.candidate_sha256,
        "changed_ring2_paths": list(report.changed_ring2_paths),
        "prev_entry_sha256": prev_entry_sha,
        "required_checkpoint_cmd": f"/usr/local/bin/mlos-recovery.sh checkpoint {ckpt_label}",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["entry_sha256"] = sha256_bytes(canonical)

    active_profile_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".mlos-profile-",
        suffix=".toml",
        dir=str(active_profile_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(cand_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, active_profile_path)
    except OSError:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as lfh:
        lfh.write(json.dumps(payload, sort_keys=True) + "\n")

    return payload
