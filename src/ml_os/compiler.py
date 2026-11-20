"""Ring 1 deterministic profile compiler (mlos-profile/v0.1 -> mkosi, systemd, Wayland, Btrfs, thermal, WSL2)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ml_os.adapt import (
    ProfileValidationError,
    collect_profile_violations,
    load_profile,
    sha256_bytes,
)

COMPILE_MANIFEST_SCHEMA = "mlos-compile-manifest/v0.1"


def render_profile_artifacts(profile: dict[str, Any], profile_sha256: str) -> dict[str, str]:
    """Render deterministic configuration files in memory from a validated mlos-profile/v0.1."""
    violations = collect_profile_violations(profile)
    if violations:
        raise ProfileValidationError(violations)

    pid = str(profile["profile_id"])
    distro_pin = profile["distro_pin"]
    hw = profile["hardware"]
    displays = hw["displays"]
    ring0 = profile["ring0_locked"]
    slices = profile["slices"]
    thermals = profile["thermal_profiles"]
    ring2 = profile["ring2_user"]

    files: dict[str, str] = {}
    files["mkosi/mkosi.conf"] = _render_mkosi_conf(pid, distro_pin, hw, ring0, ring2)
    files["mkosi/mkosi.postinst"] = _render_mkosi_postinst()
    files["mkosi/harden-image-modes.sh"] = _render_harden_modes(str(ring0["protected_ancestors_mode"]))

    files["systemd/mlos-formal.slice"] = _render_formal_slice(slices["formal"])
    files["systemd/mlos-infer.slice"] = _render_infer_slice(slices["infer"])
    files["systemd/mlos-compute.slice"] = _render_compute_slice(slices["compute"])
    files["systemd/mlos-agents.slice"] = _render_agents_slice(slices["agents"])
    files["systemd/mlos-llama-server.service"] = _render_llama_service(hw, slices["infer"])
    files["systemd/mlos-modeld.service"] = _render_modeld_service(slices["infer"])
    files["systemd/mlos-recovery-checkpoint.service"] = _render_recovery_service()

    files["storage/fstab.mlos"] = _render_fstab(hw)
    files["recovery/mlos-recovery.sh"] = _render_recovery_script()

    files["display/mlos-display-env.conf"] = _render_display_env(hw, ring2)
    files["display/hyprland-dual-screen-igpu.conf"] = _render_dual_screen_wayland(hw, displays, ring2)
    files["display/90-mlos-igpu-touch.rules"] = _render_udev_display_rules(hw, displays)
    files["display/mlos-dgpu-zero-vram-guard.sh"] = _render_dgpu_zero_vram_guard(ring2)

    files["thermal/thermal-profiles.json"] = json.dumps(
        {
            "schema": "mlos-thermal-profiles/v0.1",
            "profile_id": pid,
            "active_profile": ring2["thermal_profile"],
            "active_dgpu_power_cap_w": ring2["dgpu_power_cap_w"],
            "owner_clock_lock_approved": ring2["owner_clock_lock_approved"],
            "aas_plus_intake_requires_lid_open": displays["aas_plus_intake_requires_lid_open"],
            "acpi_lid_glob": displays["acpi_lid_glob"],
            "screenpad_sysfs_led": displays["screenpad_sysfs_led"],
            "profiles": thermals,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    files["thermal/mlos-thermal-apply.sh"] = _render_thermal_script(hw, displays, thermals, ring2)

    files["sandbox/landlock-policy.json"] = _render_landlock_policy()
    files["sandbox/seccomp-tool-filter.json"] = _render_seccomp_policy()
    files["sandbox/mlos-sandbox-exec.sh"] = _render_sandbox_exec_script()

    files["wsl2/.wslconfig.mlos"] = _render_wslconfig(hw, slices)

    artifact_hashes = {
        rel_path: sha256_bytes(content.encode("utf-8"))
        for rel_path, content in sorted(files.items())
    }
    tree_concat = "".join(f"{ sha }  { rel }\n" for rel, sha in sorted(artifact_hashes.items()))
    manifest = {
        "schema": COMPILE_MANIFEST_SCHEMA,
        "profile_id": pid,
        "profile_sha256": profile_sha256,
        "lean_toolchain": distro_pin["lean_toolchain"],
        "cuda_arch": distro_pin["cuda_arch"],
        "artifact_count": len(artifact_hashes),
        "artifacts_tree_sha256": sha256_bytes(tree_concat.encode("utf-8")),
        "artifacts": artifact_hashes,
    }
    files["compile-manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return files


def compile_profile(profile_path: Path, output_dir: Path) -> dict[str, Any]:
    """Compile a declarative profile into deterministic OS blueprints on disk."""
    profile, profile_sha = load_profile(profile_path)
    rendered = render_profile_artifacts(profile, profile_sha)

    for rel_path, content in sorted(rendered.items()):
        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content.encode("utf-8"))
        if rel_path.endswith(".sh") and os.name != "nt":
            dest.chmod(0o755)

    return json.loads(rendered["compile-manifest.json"])


def verify_compiled_directory(profile_path: Path, target_dir: Path) -> tuple[bool, list[str]]:
    """Verify that target_dir matches the freshly compiled profile byte-for-byte (0-byte drift)."""
    profile, profile_sha = load_profile(profile_path)
    expected = render_profile_artifacts(profile, profile_sha)
    mismatches: list[str] = []

    for rel_path, exp_text in sorted(expected.items()):
        actual_file = target_dir / rel_path
        if not actual_file.is_file():
            mismatches.append(f"MISSING_ARTIFACT: {rel_path}")
            continue
        actual_bytes = actual_file.read_bytes()
        exp_bytes = exp_text.encode("utf-8")
        if actual_bytes != exp_bytes:
            mismatches.append(
                f"DRIFT_ARTIFACT: {rel_path} (expected {sha256_bytes(exp_bytes)[:12]}, got {sha256_bytes(actual_bytes)[:12]})"
            )

    if target_dir.is_dir():
        existing_rels = {
            p.relative_to(target_dir).as_posix()
            for p in target_dir.rglob("*")
            if p.is_file()
        }
        for extra in sorted(existing_rels - set(expected.keys())):
            mismatches.append(f"UNTRACKED_ARTIFACT: {extra}")
    else:
        mismatches.append(f"MISSING_DIRECTORY: {target_dir}")

    return (len(mismatches) == 0, mismatches)


def _render_mkosi_conf(
    profile_id: str,
    distro_pin: dict[str, Any],
    hw: dict[str, Any],
    ring0: dict[str, Any],
    ring2: dict[str, Any],
) -> str:
    snap_slash = str(distro_pin["arch_snapshot_date"]).replace("-", "/")
    arch_mkosi = "x86-64" if hw["architecture"] == "x86_64" else str(hw["architecture"])
    lsm_csv = ",".join(ring0["lsm_list"])
    kparams = " ".join(["quiet", "rw", *ring0["required_kernel_params"], f"lsm={lsm_csv}"])

    base_pkgs = [
        "base",
        "linux",
        "linux-firmware",
        "btrfs-progs",
        "systemd",
        "systemd-ukify",
        "zram-generator",
        "bubblewrap",
        "libseccomp",
        "python",
        "git",
        "openssh",
        "nftables",
        "jq",
        "ca-certificates",
        "curl",
        "cmake",
        "ninja",
        "gcc",
        "pkgconf",
    ]
    if ring2["remote_tunnel"] == "tailscale":
        base_pkgs.append("tailscale")
    elif ring2["remote_tunnel"] == "wireguard":
        base_pkgs.append("wireguard-tools")

    if hw["dgpu_present"] and hw["dgpu_vendor"] == "nvidia":
        base_pkgs.extend(["nvidia-open", "nvidia-utils", "cuda", "lm_sensors"])

    if ring2["display_mode"] in ("dual-screen-igpu", "single-primary-igpu"):
        base_pkgs.extend(["hyprland", "waybar", "foot", "wl-clipboard", "brightnessctl"])

    for pkg in ring2.get("extra_packages", []):
        if pkg not in base_pkgs:
            base_pkgs.append(pkg)

    pkg_block = "\n".join(f"    {p}" for p in base_pkgs)
    return (
        f"# Generated deterministically by mlos-compile for profile {profile_id}.\n"
        "# Ring 0 invariant: privilege-escalation binaries (sudo, doas, pkexec) are strictly omitted.\n\n"
        "[Distribution]\n"
        "Distribution=arch\n"
        "Release=rolling\n"
        f"Architecture={arch_mkosi}\n"
        f"Mirror=https://archive.archlinux.org/repos/{snap_slash}\n\n"
        "[Output]\n"
        "Format=disk\n"
        f"Output=mlos-{profile_id}\n"
        "OutputDirectory=mkosi.output\n"
        "ManifestFormat=json\n"
        "CompressOutput=zstd\n\n"
        "[Content]\n"
        "Bootable=yes\n"
        "UnifiedKernelImages=yes\n"
        "Bootloader=systemd-boot\n"
        f"KernelCommandLine={kparams}\n"
        "Packages=\n"
        f"{pkg_block}\n"
        "PostInstallationScripts=\n"
        "    mkosi.postinst\n"
        "    harden-image-modes.sh\n\n"
        "[Validation]\n"
        "Checksum=yes\n"
    )


def _render_mkosi_postinst() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

# Deterministic post-install setup for Moissanite-Labs/ml-os.
install -d -m 0755 /etc/mlos /etc/mlos/canaries /etc/mlos/keys
install -d -m 0755 /var/lib/mlos /var/lib/mlos/models /var/lib/mlos/snapshots /var/lib/mlos/experiments /var/lib/mlos/workspace
install -d -m 0750 /var/lib/mlos/ledger

if ! getent group mlos-rpc >/dev/null 2>&1; then
    groupadd --system mlos-rpc
fi

for svc_user in mlos-llama mlos-modeld; do
    if ! id -u "${svc_user}" >/dev/null 2>&1; then
        useradd --system --no-create-home --shell /usr/bin/nologin "${svc_user}"
    fi
done

chown root:root /etc/mlos /etc/mlos/canaries /var/lib/mlos/models
chmod 0755 /etc/mlos/canaries /var/lib/mlos/models

chown root:mlos-modeld /etc/mlos/keys
chmod 0750 /etc/mlos/keys

chown mlos-modeld:mlos-modeld /var/lib/mlos/ledger
chmod 0750 /var/lib/mlos/ledger

cat >/etc/profile.d/mlos-sandbox.sh <<'EOF'
export MLOS_TOOL_EXECUTOR_REQUIRED=1
export WARP_TOOL_EXECUTOR_REQUIRED=1
export MLOS_SANDBOX_POLICY=/etc/mlos/landlock-policy.json
EOF
chmod 0644 /etc/profile.d/mlos-sandbox.sh
"""


def _render_harden_modes(ancestors_mode: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Enforce root:root ownership and {ancestors_mode} / go-w modes on trusted Ring 0 paths.
umask 022
ROOT="${{MLOS_IMAGE_ROOT:-}}"

normalize_ancestor() {{
    local target="${{ROOT%/}}$1"
    if [[ ! -e "${{target}}" && ! -L "${{target}}" ]]; then
        return 0
    fi
    if [[ "$(id -u)" -eq 0 ]]; then
        chown root:root "${{target}}"
    fi
    chmod {ancestors_mode} "${{target}}"
    chmod go-w "${{target}}"
}}

normalize_tree() {{
    local target="${{ROOT%/}}$1"
    if [[ ! -e "${{target}}" && ! -L "${{target}}" ]]; then
        return 0
    fi
    if [[ "$(id -u)" -eq 0 ]]; then
        find "${{target}}" -xdev \\( -type d -o -type f \\) -exec chown root:root {{}} +
    fi
    find "${{target}}" -xdev \\( -type d -o -type f \\) -exec chmod go-w {{}} +
}}

for ancestor in \\
    /etc \\
    /etc/profile.d \\
    /var \\
    /var/lib \\
    /var/lib/mlos \\
    /usr \\
    /usr/lib \\
    /usr/lib/systemd \\
    /usr/lib/systemd/system \\
    /usr/local \\
    /usr/local/bin; do
    normalize_ancestor "${{ancestor}}"
done

for tree in \\
    /etc/mlos/canaries \\
    /var/lib/mlos/models \\
    /usr/lib/systemd/system; do
    normalize_tree "${{tree}}"
done
"""


def _render_formal_slice(formal: dict[str, Any]) -> str:
    return (
        "[Unit]\n"
        f"Description=ml-os Formal Verification Slice ({formal['lean_toolchain']}, 0 MiB dGPU VRAM)\n"
        "Before=slices.target\n\n"
        "[Slice]\n"
        "MemoryAccounting=yes\n"
        "CPUAccounting=yes\n"
        "TasksAccounting=yes\n"
        f"MemoryHigh={formal['memory_high_gib']}G\n"
        f"MemoryMax={formal['memory_max_gib']}G\n"
        f"TasksMax={formal['tasks_max']}\n"
        f"ManagedOOMMemoryPressure=kill\n"
        "DevicePolicy=closed\n"
        "DeviceAllow=/dev/null rw\n"
        "DeviceAllow=/dev/zero rw\n"
        "DeviceAllow=/dev/urandom r\n"
    )


def _render_infer_slice(infer: dict[str, Any]) -> str:
    return (
        "[Unit]\n"
        "Description=ml-os Verified Local Inference Slice (warp-modeld + llama-server)\n"
        "Before=slices.target\n\n"
        "[Slice]\n"
        "MemoryAccounting=yes\n"
        "CPUAccounting=yes\n"
        f"MemoryMin={infer['memory_min_gib']}G\n"
        f"MemoryHigh={infer['memory_high_gib']}G\n"
        f"MemoryMax={infer['memory_max_gib']}G\n"
    )


def _render_compute_slice(compute: dict[str, Any]) -> str:
    return (
        "[Unit]\n"
        "Description=ml-os Serialized CPU float64 Compiler and CUDA Experiment Slice\n"
        "Before=slices.target\n\n"
        "[Slice]\n"
        "MemoryAccounting=yes\n"
        "CPUAccounting=yes\n"
        f"MemoryHigh={compute['memory_high_gib']}G\n"
        f"MemoryMax={compute['memory_max_gib']}G\n"
    )


def _render_agents_slice(agents: dict[str, Any]) -> str:
    return (
        "[Unit]\n"
        "Description=ml-os Sandboxed Agent Tool Execution Slice (0 MiB dGPU VRAM)\n"
        "Before=slices.target\n\n"
        "[Slice]\n"
        "MemoryAccounting=yes\n"
        "CPUAccounting=yes\n"
        f"MemoryHigh={agents['memory_high_gib']}G\n"
        f"MemoryMax={agents['memory_max_gib']}G\n"
        f"CPUQuota={agents['cpu_quota_percent']}%\n"
        "DevicePolicy=closed\n"
        "DeviceAllow=/dev/null rw\n"
        "DeviceAllow=/dev/zero rw\n"
        "DeviceAllow=/dev/urandom r\n"
    )


def _render_llama_service(hw: dict[str, Any], infer: dict[str, Any]) -> str:
    ngl_flag = " \\\n    --n-gpu-layers 99" if hw["dgpu_present"] and hw["dgpu_vendor"] == "nvidia" else ""
    return (
        "[Unit]\n"
        f"Description=ml-os Local llama-server Backend ({hw['dgpu_arch']})\n"
        "After=network.target\n"
        "Before=mlos-modeld.service\n"
        "Conflicts=mlos-gpu-train.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"Slice={infer['unit']}\n"
        f"OOMScoreAdjust={infer['oom_score_adj']}\n"
        "User=mlos-llama\n"
        "Group=mlos-llama\n"
        "RuntimeDirectory=mlos-llama-slots\n"
        "RuntimeDirectoryMode=0700\n"
        "EnvironmentFile=/etc/mlos/llama-server.env\n"
        "ExecStart=/usr/local/bin/llama-server \\\n"
        "    --host 127.0.0.1 \\\n"
        "    --port 8080 \\\n"
        "    --model ${MLOS_BASE_MODEL_GGUF} \\\n"
        "    --lora-scaled ${MLOS_LORA_SCALED_SPECS} \\\n"
        "    --lora-init-without-apply \\\n"
        "    --slots \\\n"
        "    --slot-save-path /run/mlos-llama-slots \\\n"
        "    --no-cache-idle-slots \\\n"
        "    --slot-prompt-similarity 0.0 \\\n"
        "    --ctx-size ${MLOS_CTX_SIZE} \\\n"
        f"    --ctx-checkpoints ${{MLOS_CTX_CHECKPOINTS}}{ngl_flag}\n"
        "Restart=on-failure\n"
        "RestartSec=3\n\n"
        "NoNewPrivileges=yes\n"
        "ProtectSystem=strict\n"
        "ProtectHome=yes\n"
        "PrivateTmp=yes\n"
        "ProtectKernelTunables=yes\n"
        "ProtectKernelModules=yes\n"
        "ProtectControlGroups=yes\n"
        "ReadOnlyPaths=/var/lib/mlos/models /etc/mlos/llama-server.env\n"
        "InaccessiblePaths=/etc/mlos/keys /etc/mlos/canaries /var/lib/mlos/ledger /run/mlos-modeld\n"
        "RestrictAddressFamilies=AF_INET AF_INET6\n"
        "IPAddressDeny=any\n"
        "IPAddressAllow=localhost\n"
        "UMask=0077\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _render_modeld_service(infer: dict[str, Any]) -> str:
    return (
        "[Unit]\n"
        "Description=ml-os L3 Fail-Closed Inference Gatekeeper (warp-modeld)\n"
        "After=mlos-llama-server.service\n"
        "Requires=mlos-llama-server.service\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"Slice={infer['unit']}\n"
        f"OOMScoreAdjust={infer['oom_score_adj']}\n"
        "User=mlos-modeld\n"
        "Group=mlos-rpc\n"
        "RuntimeDirectory=mlos-modeld\n"
        "RuntimeDirectoryMode=0750\n"
        "ExecStart=/usr/bin/warp-modeld serve --config /etc/mlos/modeld.json\n"
        "Restart=on-failure\n"
        "RestartSec=2\n\n"
        "NoNewPrivileges=yes\n"
        "ProtectSystem=strict\n"
        "ProtectHome=yes\n"
        "PrivateTmp=yes\n"
        "ProtectKernelTunables=yes\n"
        "ProtectKernelModules=yes\n"
        "ProtectControlGroups=yes\n"
        "ReadOnlyPaths=/var/lib/mlos/models /etc/mlos/canaries /etc/mlos/keys /etc/mlos/modeld.json\n"
        "ReadWritePaths=/var/lib/mlos/ledger /run/mlos-modeld\n"
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6\n"
        "IPAddressDeny=any\n"
        "IPAddressAllow=localhost\n"
        "CapabilityBoundingSet=\n"
        "AmbientCapabilities=\n"
        "LockPersonality=yes\n"
        "MemoryDenyWriteExecute=yes\n"
        "UMask=0027\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _render_recovery_service() -> str:
    return (
        "[Unit]\n"
        "Description=ml-os Pre-Adaptation Btrfs Snapshot and UKI Checkpoint\n"
        "DefaultDependencies=no\n"
        "After=local-fs.target\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        "ExecStart=/usr/local/bin/mlos-recovery.sh checkpoint pre-adaptation\n"
        "StandardOutput=journal\n"
        "StandardError=journal\n\n"
        "[Install]\n"
        "WantedBy=sysinit.target\n"
    )


def _render_fstab(hw: dict[str, Any]) -> str:
    return (
        "# Btrfs subvolume layout and read-only model vault for Moissanite-Labs/ml-os.\n"
        "# Persistent subvolumes survive 1-reboot @ root rollback untouched.\n"
        "LABEL=MLOS_BTRFS  /                         btrfs  subvol=@,compress=zstd:3,noatime,ssd,discard=async  0 0\n"
        "LABEL=MLOS_BTRFS  /var                      btrfs  subvol=@var,compress=zstd:3,noatime,ssd,discard=async  0 0\n"
        "LABEL=MLOS_BTRFS  /var/lib/mlos/workspace   btrfs  subvol=@workspace,compress=zstd:3,noatime,ssd,discard=async  0 0\n"
        "LABEL=MLOS_BTRFS  /var/lib/mlos/ledger      btrfs  subvol=@ledger,compress=zstd:3,noatime,ssd,discard=async  0 0\n"
        "LABEL=MLOS_BTRFS  /var/lib/mlos/models      btrfs  subvol=@models,ro,nosuid,nodev,noexec,compress=zstd:3,noatime  0 0\n"
        "LABEL=MLOS_BTRFS  /var/lib/mlos/experiments btrfs  subvol=@experiments,compress=zstd:3,noatime,ssd,discard=async  0 0\n"
        "LABEL=MLOS_BTRFS  /var/lib/mlos/snapshots   btrfs  subvol=@snapshots,compress=zstd:3,noatime,ssd,discard=async  0 0\n"
        "tmpfs             /tmp                      tmpfs  size=4G,nosuid,nodev  0 0\n"
        f"# zram-generator target: {hw['zram_gib']} GiB zstd swap\n"
    )


def _render_recovery_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

# Single-reboot Btrfs @ snapshot + archived UKI rollback helper for Moissanite-Labs/ml-os.
SNAPSHOT_ROOT="${MLOS_SNAPSHOT_ROOT:-/var/lib/mlos/snapshots}"
EFI_LINUX_DIR="${MLOS_EFI_LINUX_DIR:-/efi/EFI/Linux}"
ACTIVE_UKI="${MLOS_ACTIVE_UKI:-${EFI_LINUX_DIR}/mlos-current.efi}"
ROOT_SUBVOL="${MLOS_ROOT_SUBVOL:-/}"

usage() {
    echo "Usage: $0 <checkpoint <label> | list | stage-rollback <checkpoint-id>>" >&2
    exit 2
}

cmd_checkpoint() {
    local label="${1:-}"
    if [[ -z "${label}" || ! "${label}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
        echo "ERR_RECOVERY_LABEL: label must match [a-zA-Z0-9._-]+" >&2
        exit 2
    fi
    local stamp
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    local ckpt_id="${stamp}-${label}"
    local snap_dir="${SNAPSHOT_ROOT}/${ckpt_id}"
    local archived_uki="${EFI_LINUX_DIR}/mlos-rollback-${ckpt_id}.efi"

    if ! command -v btrfs >/dev/null 2>&1; then
        echo "ERR_RECOVERY_TOOL: btrfs-progs is required" >&2
        exit 126
    fi
    if [[ ! -f "${ACTIVE_UKI}" ]]; then
        echo "ERR_RECOVERY_UKI: active UKI missing at ${ACTIVE_UKI}" >&2
        exit 126
    fi

    mkdir -p "${SNAPSHOT_ROOT}"
    btrfs subvolume snapshot -r "${ROOT_SUBVOL}" "${snap_dir}"
    cp --preserve=timestamps "${ACTIVE_UKI}" "${archived_uki}"
    sha256sum "${archived_uki}" > "${SNAPSHOT_ROOT}/${ckpt_id}.uki.sha256"
    echo "checkpoint_created=${ckpt_id}"
}

cmd_list() {
    if [[ ! -d "${SNAPSHOT_ROOT}" ]]; then
        echo "no checkpoints found"
        return 0
    fi
    find "${SNAPSHOT_ROOT}" -maxdepth 1 -name '*.uki.sha256' -print | sort
}

cmd_stage_rollback() {
    local ckpt_id="${1:-}"
    if [[ -z "${ckpt_id}" || ! "${ckpt_id}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
        echo "ERR_RECOVERY_ID: invalid checkpoint id" >&2
        exit 2
    fi
    local manifest="${SNAPSHOT_ROOT}/${ckpt_id}.uki.sha256"
    local snap_dir="${SNAPSHOT_ROOT}/${ckpt_id}"
    if [[ ! -f "${manifest}" || ! -d "${snap_dir}" ]]; then
        echo "ERR_RECOVERY_MISSING: checkpoint ${ckpt_id} not found" >&2
        exit 1
    fi

    sha256sum -c "${manifest}"
    if ! command -v bootctl >/dev/null 2>&1; then
        echo "ERR_RECOVERY_BOOTCTL: systemd bootctl is required" >&2
        exit 126
    fi
    bootctl set-oneshot "mlos-rollback-${ckpt_id}.efi"
    echo "rollback_staged=${ckpt_id}"
}

if [[ $# -lt 1 ]]; then
    usage
fi

case "$1" in
    checkpoint)
        shift
        cmd_checkpoint "${1:-}"
        ;;
    list)
        cmd_list
        ;;
    stage-rollback)
        shift
        cmd_stage_rollback "${1:-}"
        ;;
    *)
        usage
        ;;
esac
"""


def _render_display_env(hw: dict[str, Any], ring2: dict[str, Any]) -> str:
    drm_path = hw["igpu_drm_by_path"]
    return (
        "# Strict Intel/iGPU display binding enforcing 0 MiB desktop VRAM on discrete NVIDIA GPUs.\n"
        f"MLOS_DISPLAY_MODE={ring2['display_mode']}\n"
        f"AQ_DRM_DEVICES={drm_path}\n"
        f"WLR_DRM_DEVICES={drm_path}\n"
        "__NV_PRIME_RENDER_OFFLOAD=0\n"
        "GBM_BACKEND=iris\n"
        "LIBVA_DRIVER_NAME=iHD\n"
    )


def _render_dual_screen_wayland(
    hw: dict[str, Any],
    displays: dict[str, Any],
    ring2: dict[str, Any],
) -> str:
    drm_path = hw["igpu_drm_by_path"]
    dual_cfg = ring2["dual_screen"]
    p_conn = displays["primary_connector"]
    p_res = displays["primary_resolution"]
    p_hz = displays["primary_refresh_hz"]
    p_scale = displays["primary_scale"]

    lines = [
        "# Dual-screen and iGPU-only Wayland blueprint generated by mlos-compile.",
        "# Keeps discrete NVIDIA GPU at exact 0 MiB desktop VRAM while handling",
        "# UX582HS primary OLED + tilted matte IPS ScreenPad Plus geometry and touch binding.",
        f"env = AQ_DRM_DEVICES,{drm_path}",
        f"env = WLR_DRM_DEVICES,{drm_path}",
        "env = __NV_PRIME_RENDER_OFFLOAD,0",
        "",
        f"monitor = {p_conn},{p_res}@{p_hz},0x0,{p_scale}",
    ]

    if displays["dual_screen"]:
        s_conn = displays["secondary_connector"]
        s_res = displays["secondary_resolution"]
        s_hz = displays["secondary_refresh_hz"]
        s_scale = displays["secondary_scale"]
        p_height = int(str(p_res).split("x")[1])
        s_y = p_height // max(1, int(p_scale))
        if dual_cfg["screenpad_role"] == "disabled" or ring2["display_mode"] == "single-primary-igpu":
            lines.append(f"monitor = {s_conn},disable")
        else:
            lines.append(f"monitor = {s_conn},{s_res}@{s_hz},0x{s_y},{s_scale}")
        lines.extend(
            [
                "",
                "# Per-panel touch and stylus coordinate binding (prevents ScreenPad touches clicking main OLED).",
                "device {",
                "    name = elan9008:00-04f3:2c82",
                f"    output = {p_conn}",
                "}",
                "device {",
                "    name = elan9009:00-04f3:2c83",
                f"    output = {s_conn}",
                "}",
                "",
                f"# OLED burn-in protection: blank {p_conn} after {dual_cfg['oled_idle_blank_sec']}s idle;",
                f"# {s_conn} ({displays['secondary_panel_type']}) hosts persistent low-brightness VRAM/slice telemetry.",
                f"workspace = 1, monitor:{p_conn}, default:true",
                f"workspace = 9, monitor:{s_conn}, default:true",
            ]
        )

    return "\n".join(lines) + "\n"


def _render_udev_display_rules(hw: dict[str, Any], displays: dict[str, Any]) -> str:
    lines = [
        "# Udev tags for iGPU-primary compositor selection and ASUS ScreenPad Plus backlight control.",
        f'SUBSYSTEM=="drm", KERNELS=="{hw["igpu_pci"]}", TAG+="mlos-primary-igpu"',
    ]
    if hw["dgpu_present"] and hw["dgpu_pci"] != "none":
        lines.append(
            f'SUBSYSTEM=="drm", KERNELS=="{hw["dgpu_pci"]}", TAG-="mutter-device-preferred-primary", ENV{{MLOS_COMPUTE_ONLY_DGPU}}="1"'
        )
    if displays["screenpad_sysfs_led"] != "none":
        lines.append(
            'SUBSYSTEM=="leds", KERNEL=="asus::screenpad", MODE="0664", GROUP="video"'
        )
    return "\n".join(lines) + "\n"


def _render_dgpu_zero_vram_guard(ring2: dict[str, Any]) -> str:
    fallback = "1" if ring2["dual_screen"]["fallback_to_headless_on_dgpu_leak"] else "0"
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Verifies 0 MiB desktop/compositor VRAM on discrete NVIDIA GPU.
# Falls back to headless SSH console mode if any local compositor allocates dGPU scanout buffers.
FALLBACK_HEADLESS="{fallback}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    exit 0
fi

USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
COMPUTE_MIB="$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits | awk '{{s+=$1}} END {{print s+0}}')"
DESKTOP_MIB=$(( USED_MIB - COMPUTE_MIB ))

if (( DESKTOP_MIB > 0 )); then
    echo "ERR_RING0_DGPU_DESKTOP_VRAM: non-compute desktop VRAM is ${{DESKTOP_MIB}} MiB (expected 0 MiB)" >&2
    if [[ "${{FALLBACK_HEADLESS}}" == "1" ]] && command -v systemctl >/dev/null 2>&1; then
        echo "ACTION_FALLBACK_HEADLESS: isolating multi-user.target to preserve full 8 GiB CUDA pool" >&2
        systemctl isolate multi-user.target || true
    fi
    exit 1
fi

echo "dgpu_desktop_vram_mib=0"
"""


def _render_thermal_script(
    hw: dict[str, Any],
    displays: dict[str, Any],
    thermals: dict[str, Any],
    ring2: dict[str, Any],
) -> str:
    active_name = str(ring2["thermal_profile"])
    active_tp = thermals[active_name]
    quiet_tp = thermals["quiet-edit"]
    owner_clock_ok = "1" if ring2["owner_clock_lock_approved"] else "0"
    aas_lid_req = "1" if (displays["aas_plus_intake_requires_lid_open"] and active_tp["require_lid_open"]) else "0"
    lid_glob = str(displays["acpi_lid_glob"])
    sp_led = str(displays["screenpad_sysfs_led"])

    return f"""#!/usr/bin/env bash
set -euo pipefail

# Applies ASUS thermal profile, ScreenPad Plus brightness, NVIDIA power cap, and AAS Plus lid-intake guard.
REQUESTED_PROFILE="${{1:-{active_name}}}"
DGPU_PRESENT="{1 if hw['dgpu_present'] else 0}"
OWNER_CLOCK_OK="{owner_clock_ok}"
AAS_LID_REQ="{aas_lid_req}"
LID_GLOB="{lid_glob}"
SCREENPAD_LED="{sp_led}"

FAN_PROFILE="{active_tp['fan_profile']}"
POWER_CAP_W="{ring2['dgpu_power_cap_w']}"
LOCK_CLOCKS="{1 if active_tp['lock_sm_clocks'] else 0}"
SM_CLOCK_MHZ="{active_tp['sm_clock_mhz']}"
SCREENPAD_BRIGHTNESS="{active_tp['screenpad_default_brightness']}"

# UX582HS AAS Plus hinge guard: closing the lid blocks the upper keyboard intake above the vapor chamber.
if [[ "${{AAS_LID_REQ}}" == "1" && "${{LID_GLOB}}" != "none" ]]; then
    for lid_state_file in ${{LID_GLOB}}; do
        if [[ -f "${{lid_state_file}}" ]] && grep -qi "closed" "${{lid_state_file}}"; then
            echo "WARN_AAS_PLUS_LID_CLOSED_CLAMP: lid is closed; clamping ${{REQUESTED_PROFILE}} to quiet-edit ({quiet_tp['dgpu_power_cap_w']} W)" >&2
            REQUESTED_PROFILE="quiet-edit"
            FAN_PROFILE="{quiet_tp['fan_profile']}"
            POWER_CAP_W="{quiet_tp['dgpu_power_cap_w']}"
            LOCK_CLOCKS="0"
            SCREENPAD_BRIGHTNESS="0"
            break
        fi
    done
fi

if command -v asusctl >/dev/null 2>&1; then
    asusctl profile -P "${{FAN_PROFILE}}"
fi

if [[ "${{SCREENPAD_LED}}" != "none" && -w "${{SCREENPAD_LED}}" ]]; then
    printf '%s\\n' "${{SCREENPAD_BRIGHTNESS}}" > "${{SCREENPAD_LED}}"
fi

if [[ "${{DGPU_PRESENT}}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -pm 1
    nvidia-smi -pl "${{POWER_CAP_W}}"
    if [[ "${{LOCK_CLOCKS}}" == "1" ]]; then
        if [[ "${{OWNER_CLOCK_OK}}" != "1" ]]; then
            echo "ERR_CLOCK_LOCK_UNAPPROVED: owner_clock_lock_approved is required for SM clock lock" >&2
            exit 1
        fi
        nvidia-smi -lgc "${{SM_CLOCK_MHZ}},${{SM_CLOCK_MHZ}}"
    else
        nvidia-smi -rgc || true
    fi
fi

echo "thermal_applied=${{REQUESTED_PROFILE}} power_cap_w=${{POWER_CAP_W}}"
"""


def _render_landlock_policy() -> str:
    policy = {
        "abi_version": 4,
        "default_action": "deny",
        "env_gate": "MLOS_TOOL_EXECUTOR_REQUIRED=1",
        "fail_closed_if_unsupported": True,
        "network": {
            "unshare_net_default": True,
            "zone_a_allow_network": False,
            "zone_b_requires_egress_proxy": True,
        },
        "rules": {
            "explicit_deny": [
                "/efi",
                "/etc/mlos/keys",
                "/run/mlos-modeld",
                "/var/lib/mlos/ledger",
                "/var/lib/mlos/snapshots",
            ],
            "read_only": [
                "/etc/mlos/canaries",
                "/etc/ssl/certs",
                "/lib",
                "/lib64",
                "/usr",
                "/var/lib/mlos/models",
            ],
            "read_write": [
                "$MLOS_EPHEMERAL_WORKTREE",
                "/tmp",
            ],
        },
        "schema_version": "0.1.0",
    }
    return json.dumps(policy, indent=2, sort_keys=True) + "\n"


def _render_seccomp_policy() -> str:
    seccomp = {
        "allow_groups": [
            "basic_io",
            "filesystem_stat_and_read",
            "memory_alloc",
            "process_lifecycle",
            "signals_and_clocks",
            "worktree_file_write",
        ],
        "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
        "default_action": "SCMP_ACT_ERRNO",
        "deny_explicit": [
            "bpf",
            "chroot",
            "delete_module",
            "finit_module",
            "init_module",
            "kexec_file_load",
            "kexec_load",
            "keyctl",
            "mount",
            "pivot_root",
            "process_vm_readv",
            "process_vm_writev",
            "ptrace",
            "reboot",
            "setns",
            "swapoff",
            "swapon",
            "umount2",
            "unshare",
            "userfaultfd",
        ],
        "schema_version": "0.1.0",
    }
    return json.dumps(seccomp, indent=2, sort_keys=True) + "\n"


def _render_sandbox_exec_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

# Fail-closed bubblewrap tool executor for Moissanite-Labs/ml-os.
if [[ "${MLOS_TOOL_EXECUTOR_REQUIRED:-1}" != "1" || "${WARP_TOOL_EXECUTOR_REQUIRED:-1}" != "1" ]]; then
    echo "ERR_SANDBOX_POLICY: MLOS_TOOL_EXECUTOR_REQUIRED and WARP_TOOL_EXECUTOR_REQUIRED must remain 1" >&2
    exit 126
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ephemeral-worktree-dir> <command> [args...]" >&2
    exit 2
fi

WORKTREE="$1"
shift

if [[ ! -d "${WORKTREE}" ]]; then
    echo "ERR_SANDBOX_WORKTREE: worktree directory does not exist: ${WORKTREE}" >&2
    exit 126
fi

if ! command -v bwrap >/dev/null 2>&1; then
    echo "ERR_SANDBOX_UNAVAILABLE: bubblewrap (bwrap) is required; refusing unsandboxed execution" >&2
    exit 126
fi

BWRAP_ARGS=(
    --unshare-all
    --unshare-net
    --die-with-parent
    --new-session
    --clearenv
    --setenv PATH "/usr/local/bin:/usr/bin:/bin"
    --setenv LANG "C.UTF-8"
    --setenv PYTHONHASHSEED "0"
    --setenv MLOS_SANDBOXED "1"
    --ro-bind /usr /usr
    --ro-bind /bin /bin
    --ro-bind /lib /lib
    --proc /proc
    --dev /dev
    --tmpfs /tmp
    --bind "${WORKTREE}" /workspace
    --chdir /workspace
)

if [[ -d /lib64 ]]; then
    BWRAP_ARGS+=(--ro-bind /lib64 /lib64)
fi
if [[ -d /etc/mlos/canaries ]]; then
    BWRAP_ARGS+=(--ro-bind /etc/mlos/canaries /etc/mlos/canaries)
fi
if [[ -d /var/lib/mlos/models ]]; then
    BWRAP_ARGS+=(--ro-bind /var/lib/mlos/models /var/lib/mlos/models)
fi

exec bwrap "${BWRAP_ARGS[@]}" -- "$@"
"""


def _render_wslconfig(hw: dict[str, Any], slices: dict[str, Any]) -> str:
    wsl_mem_gib = min(max(8, int(hw["ram_gib"]) - 8), int(slices["compute"]["memory_high_gib"]))
    return (
        "# Stage 0 non-destructive WSL2 resource cap for Windows 11 pre-dual-boot CUDA workbench.\n"
        "[wsl2]\n"
        f"memory={wsl_mem_gib}GB\n"
        f"processors={hw['cpu_threads']}\n"
        f"swap={hw['zram_gib']}GB\n"
        "localhostForwarding=true\n"
        "\n"
        "[experimental]\n"
        "autoMemoryReclaim=gradual\n"
        "sparseVhd=true\n"
    )
