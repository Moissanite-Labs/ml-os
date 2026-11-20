"""Ring 1 read-only cross-OS hardware probe (Linux, Windows 11, and macOS dev parity).

Strictly read-only: performs zero disk modifications and never fabricates fallback
hardware names, PCI IDs, VRAM numbers, or git commits (uses None when absent).
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Callable

PROBE_SCHEMA = "mlos-hardware-probe/v0.1"

CommandRunner = Callable[[list[str]], tuple[int, str, str]]


def default_command_runner(cmd: list[str]) -> tuple[int, str, str]:
    """Execute a read-only probe command with stripped interactive state and short timeout."""
    exe = cmd[0]
    resolved = shutil.which(exe)
    if resolved is None:
        if os.name == "nt" and exe == "nvidia-smi":
            win_smi = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
            if win_smi.is_file():
                resolved = str(win_smi)
        if resolved is None:
            return (127, "", f"{exe} not found")

    try:
        proc = subprocess.run(
            [resolved, *cmd[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
        return (proc.returncode, proc.stdout, proc.stderr)
    except (OSError, subprocess.SubprocessError) as exc:
        return (126, "", str(exc))


def compute_cap_to_sm_arch(compute_cap: str | None) -> str | None:
    """Convert NVIDIA compute capability like '8.6' to 'sm_86', or None if unknown."""
    if not compute_cap:
        return None
    cleaned = compute_cap.strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", cleaned)
    if not match:
        return None
    return f"sm_{match.group(1)}{match.group(2)}"


def _parse_int_or_none(raw: str | None) -> int | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("[not supported]", "n/a", "none", "null"):
        return None
    match = re.search(r"-?\d+", cleaned.replace(",", ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


@dataclass(frozen=True)
class NvidiaProbeTelemetry:
    """Read-only snapshot of NVIDIA dGPU hardware and WDDM/desktop vs compute VRAM usage."""

    gpu_name: str | None
    pci_bus_id: str | None
    driver_version: str | None
    compute_cap: str | None
    dgpu_arch: str | None
    vram_total_mib: int | None
    vram_used_mib: int | None
    vram_free_mib: int | None
    compute_vram_mib: int
    non_compute_desktop_vram_mib: int | None
    compute_processes: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "pci_bus_id": self.pci_bus_id,
            "driver_version": self.driver_version,
            "compute_cap": self.compute_cap,
            "dgpu_arch": self.dgpu_arch,
            "vram_total_mib": self.vram_total_mib,
            "vram_used_mib": self.vram_used_mib,
            "vram_free_mib": self.vram_free_mib,
            "compute_vram_mib": self.compute_vram_mib,
            "non_compute_desktop_vram_mib": self.non_compute_desktop_vram_mib,
            "compute_processes": list(self.compute_processes),
        }


def probe_nvidia_smi(runner: CommandRunner = default_command_runner) -> NvidiaProbeTelemetry | None:
    """Query nvidia-smi read-only for GPU model, arch, total/used VRAM, and compute vs desktop split."""
    rc, out, _ = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,pci.bus_id,driver_version,compute_cap,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0 or not out.strip():
        return None

    first_line = out.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 7:
        return None

    gpu_name = parts[0] or None
    raw_pci = parts[1] or None
    pci_norm: str | None = None
    if raw_pci:
        # NVIDIA sometimes prefixes 8-hex domain 00000000:01:00.0; normalize to 0000:01:00.0.
        pci_lower = raw_pci.lower()
        if len(pci_lower) >= 16 and pci_lower.startswith("00000000:"):
            pci_norm = "0000:" + pci_lower[9:]
        else:
            pci_norm = pci_lower

    driver = parts[2] or None
    comp_cap = parts[3] if parts[3] and parts[3].lower() not in ("n/a", "[not supported]") else None
    total_mib = _parse_int_or_none(parts[4])
    used_mib = _parse_int_or_none(parts[5])
    free_mib = _parse_int_or_none(parts[6])

    rc_apps, out_apps, _ = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    compute_procs: list[dict[str, Any]] = []
    compute_sum = 0
    if rc_apps == 0 and out_apps.strip():
        for line in out_apps.strip().splitlines():
            row = [c.strip() for c in line.split(",")]
            if len(row) < 3:
                continue
            pid_val = _parse_int_or_none(row[0])
            mem_val = _parse_int_or_none(row[2])
            if pid_val is None:
                continue
            mib = max(0, mem_val if mem_val is not None else 0)
            compute_sum += mib
            compute_procs.append(
                {
                    "pid": pid_val,
                    "process_name": row[1] or None,
                    "used_gpu_memory_mib": mib,
                }
            )

    desktop_overhead: int | None = None
    if used_mib is not None:
        desktop_overhead = max(0, used_mib - compute_sum)

    return NvidiaProbeTelemetry(
        gpu_name=gpu_name,
        pci_bus_id=pci_norm,
        driver_version=driver,
        compute_cap=comp_cap,
        dgpu_arch=compute_cap_to_sm_arch(comp_cap),
        vram_total_mib=total_mib,
        vram_used_mib=used_mib,
        vram_free_mib=free_mib,
        compute_vram_mib=compute_sum,
        non_compute_desktop_vram_mib=desktop_overhead,
        compute_processes=tuple(compute_procs),
    )


def _read_optional_text(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        val = path.read_text(encoding="utf-8", errors="replace").strip()
        return val if val else None
    except OSError:
        return None


def probe_linux(
    *,
    runner: CommandRunner = default_command_runner,
    root: Path = Path("/"),
    screenpad_led_subpath: str = "sys/class/leds/asus::screenpad/brightness",
) -> dict[str, Any]:
    """Read-only Linux hardware probe (/sys, /proc, lspci, DRM, ScreenPad, lid, nvidia-smi)."""
    dmi_dir = root / "sys" / "class" / "dmi" / "id"
    manufacturer = _read_optional_text(dmi_dir / "sys_vendor")
    model = _read_optional_text(dmi_dir / "product_name")

    cpu_model: str | None = None
    cpu_threads: int | None = None
    cpu_cores: int | None = None
    cpuinfo_text = _read_optional_text(root / "proc" / "cpuinfo")
    if cpuinfo_text:
        logical_count = 0
        core_pairs: set[tuple[str, str]] = set()
        phys_id = "0"
        for line in cpuinfo_text.splitlines():
            if ":" not in line:
                continue
            k, v = [x.strip() for x in line.split(":", 1)]
            if k == "processor":
                logical_count += 1
            elif k in ("model name", "Processor", "Hardware") and cpu_model is None and v:
                cpu_model = v
            elif k == "physical id":
                phys_id = v
            elif k == "core id":
                core_pairs.add((phys_id, v))
        if logical_count > 0:
            cpu_threads = logical_count
        if core_pairs:
            cpu_cores = len(core_pairs)

    ram_mib: int | None = None
    ram_gib: int | None = None
    meminfo_text = _read_optional_text(root / "proc" / "meminfo")
    if meminfo_text:
        for line in meminfo_text.splitlines():
            if line.startswith("MemTotal:"):
                kib = _parse_int_or_none(line.split(":", 1)[1])
                if kib is not None and kib > 0:
                    ram_mib = kib // 1024
                    ram_gib = round(kib / (1024 * 1024))
                break

    igpu_pci: str | None = None
    igpu_model: str | None = None
    dgpu_pci: str | None = None
    dgpu_model: str | None = None

    rc_pci, out_pci, _ = runner(["lspci", "-Dnns"])
    if rc_pci == 0 and out_pci.strip():
        for line in out_pci.splitlines():
            # Match VGA compatible controller [0300], 3D controller [0302], Display controller [0380].
            if not any(cls in line for cls in ("[0300]", "[0302]", "[0380]")):
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            slot = parts[0].strip().lower()
            desc = parts[1].strip()
            if "[8086:" in desc or "intel" in desc.lower():
                if igpu_pci is None:
                    igpu_pci = slot
                    igpu_model = desc
            elif "[10de:" in desc or "nvidia" in desc.lower():
                if dgpu_pci is None:
                    dgpu_pci = slot
                    dgpu_model = desc

    drm_connectors: list[dict[str, Any]] = []
    drm_dir = root / "sys" / "class" / "drm"
    if drm_dir.is_dir():
        for entry in sorted(drm_dir.iterdir(), key=lambda p: p.name):
            if "-" not in entry.name or not entry.name.startswith("card"):
                continue
            status = _read_optional_text(entry / "status")
            if status != "connected":
                continue
            modes_raw = _read_optional_text(entry / "modes")
            top_mode = modes_raw.splitlines()[0].strip() if modes_raw else None
            conn_name = entry.name.split("-", 1)[1]
            drm_connectors.append(
                {
                    "card_connector": entry.name,
                    "connector": conn_name,
                    "status": status,
                    "preferred_mode": top_mode,
                }
            )

    screenpad_led_path = root / screenpad_led_subpath
    try:
        screenpad_present = screenpad_led_path.is_file()
    except OSError:
        screenpad_present = False
    screenpad_brightness = _parse_int_or_none(_read_optional_text(screenpad_led_path)) if screenpad_present else None

    lid_open: bool | None = None
    lid_glob = str(root / "proc" / "acpi" / "button" / "lid" / "*" / "state")
    for lid_file in sorted(glob.glob(lid_glob)):
        state_txt = _read_optional_text(Path(lid_file))
        if state_txt:
            if "open" in state_txt.lower():
                lid_open = True
                break
            if "closed" in state_txt.lower():
                lid_open = False
                break

    nvme_models: list[str] = []
    nvme_dir = root / "sys" / "class" / "nvme"
    if nvme_dir.is_dir():
        for ctrl in sorted(nvme_dir.iterdir(), key=lambda p: p.name):
            m_txt = _read_optional_text(ctrl / "model")
            if m_txt:
                nvme_models.append(m_txt)

    secure_boot_enabled: bool | None = None
    efivars_dir = root / "sys" / "firmware" / "efi" / "efivars"
    if efivars_dir.is_dir():
        for sb_path in sorted(efivars_dir.glob("SecureBoot-*")):
            try:
                raw_sb = sb_path.read_bytes()
                if len(raw_sb) >= 5:
                    secure_boot_enabled = raw_sb[4] == 1
                    break
            except OSError:
                continue

    nvidia = probe_nvidia_smi(runner=runner)
    if nvidia is not None:
        if nvidia.gpu_name:
            dgpu_model = nvidia.gpu_name
        if nvidia.pci_bus_id and dgpu_pci is None:
            dgpu_pci = nvidia.pci_bus_id

    return {
        "schema": PROBE_SCHEMA,
        "host_os": "linux",
        "os_release": platform.release() if root == Path("/") else None,
        "architecture": platform.machine() or None,
        "manufacturer": manufacturer,
        "model": model,
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "ram_mib": ram_mib,
        "ram_gib": ram_gib,
        "igpu_pci": igpu_pci,
        "igpu_model": igpu_model,
        "dgpu_pci": dgpu_pci,
        "dgpu_model": dgpu_model,
        "dgpu_arch": nvidia.dgpu_arch if nvidia else None,
        "dgpu_vram_mib": nvidia.vram_total_mib if nvidia else None,
        "nvme_model": nvme_models[0] if nvme_models else None,
        "secure_boot_enabled": secure_boot_enabled,
        "bitlocker_protection": None,
        "dual_screen": {
            "connected_outputs": drm_connectors,
            "dual_panel_detected": len(drm_connectors) >= 2 or screenpad_present,
            "asus_screenpad_sysfs_present": screenpad_present,
            "asus_screenpad_brightness": screenpad_brightness,
            "lid_open": lid_open,
        },
        "nvidia_telemetry": nvidia.to_dict() if nvidia else None,
    }


WINDOWS_POWERSHELL_PROBE_SCRIPT = (
    "$ErrorActionPreference = 'SilentlyContinue'; "
    "$cs = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory; "
    "$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,BuildNumber,OSArchitecture; "
    "$cpu = @(Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors); "
    "$gpus = @(Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,AdapterRAM,DriverVersion); "
    "$disks = @(Get-CimInstance Win32_DiskDrive | Select-Object Model,Size,MediaType,InterfaceType); "
    "$sb = $null; try { $sb = [bool](Confirm-SecureBootUEFI) } catch { $sb = $null }; "
    "$bl = @(); try { $bl = @(Get-BitLockerVolume | Select-Object MountPoint,ProtectionStatus) } catch { $bl = @() }; "
    "[pscustomobject]@{ "
    "ComputerSystem=$cs; OperatingSystem=$os; Processor=$cpu; "
    "VideoControllers=$gpus; DiskDrives=$disks; SecureBoot=$sb; BitLocker=$bl "
    "} | ConvertTo-Json -Depth 4 -Compress"
)


def probe_windows(runner: CommandRunner = default_command_runner) -> dict[str, Any]:
    """Read-only Windows 11 hardware, BitLocker, Secure Boot, and WDDM VRAM probe."""
    ps_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_POWERSHELL_PROBE_SCRIPT]
    rc, out, _ = runner(ps_cmd)
    if rc != 0 or not out.strip():
        rc, out, _ = runner(["pwsh", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_POWERSHELL_PROBE_SCRIPT])

    cim: dict[str, Any] = {}
    if rc == 0 and out.strip():
        try:
            parsed = json.loads(out.strip())
            if isinstance(parsed, dict):
                cim = parsed
        except json.JSONDecodeError:
            cim = {}

    cs = cim.get("ComputerSystem") if isinstance(cim.get("ComputerSystem"), dict) else {}
    os_info = cim.get("OperatingSystem") if isinstance(cim.get("OperatingSystem"), dict) else {}
    procs = cim.get("Processor") if isinstance(cim.get("Processor"), list) else []
    proc0 = procs[0] if procs and isinstance(procs[0], dict) else {}
    gpus = cim.get("VideoControllers") if isinstance(cim.get("VideoControllers"), list) else []
    disks = cim.get("DiskDrives") if isinstance(cim.get("DiskDrives"), list) else []

    total_bytes = _parse_int_or_none(str(cs.get("TotalPhysicalMemory"))) if cs.get("TotalPhysicalMemory") is not None else None
    ram_mib = total_bytes // (1024 * 1024) if total_bytes and total_bytes > 0 else None
    ram_gib = round(total_bytes / (1024 * 1024 * 1024)) if total_bytes and total_bytes > 0 else None

    igpu_model: str | None = None
    dgpu_model: str | None = None
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        gname = str(gpu.get("Name") or "").strip()
        if not gname:
            continue
        low = gname.lower()
        if "nvidia" in low and dgpu_model is None:
            dgpu_model = gname
        elif ("intel" in low or "uhd" in low or "iris" in low or "radeon graphics" in low) and igpu_model is None:
            igpu_model = gname

    nvme_model: str | None = None
    nvme_capacity_gb: int | None = None
    for disk in disks:
        if not isinstance(disk, dict):
            continue
        dmodel = str(disk.get("Model") or "").strip()
        if not dmodel:
            continue
        if nvme_model is None or "nvme" in dmodel.lower() or "mzvl" in dmodel.lower():
            nvme_model = dmodel
            dsize = _parse_int_or_none(str(disk.get("Size"))) if disk.get("Size") is not None else None
            if dsize and dsize > 0:
                nvme_capacity_gb = round(dsize / 1_000_000_000)

    nvidia = probe_nvidia_smi(runner=runner)
    if nvidia is not None and nvidia.gpu_name:
        dgpu_model = nvidia.gpu_name

    sb_raw = cim.get("SecureBoot")
    secure_boot = sb_raw if type(sb_raw) is bool else None

    bl_raw = cim.get("BitLocker")
    bitlocker_summary: list[dict[str, Any]] | None = None
    if isinstance(bl_raw, list) and bl_raw:
        bitlocker_summary = [
            {
                "mount_point": item.get("MountPoint"),
                "protection_status": item.get("ProtectionStatus"),
            }
            for item in bl_raw
            if isinstance(item, dict)
        ]

    caption = str(os_info.get("Caption") or "").strip() or None
    build_num = str(os_info.get("BuildNumber") or "").strip() or None
    os_release = f"{caption} (Build {build_num})" if caption and build_num else (caption or build_num)

    manufacturer = str(cs.get("Manufacturer") or "").strip() or None
    model = str(cs.get("Model") or "").strip() or None
    dual_detected = bool(model and "ux582" in model.lower())

    return {
        "schema": PROBE_SCHEMA,
        "host_os": "windows",
        "os_release": os_release,
        "architecture": "x86_64" if "64" in str(os_info.get("OSArchitecture", platform.machine())) else (platform.machine() or None),
        "manufacturer": manufacturer,
        "model": model,
        "cpu_model": str(proc0.get("Name") or "").strip() or None,
        "cpu_cores": _parse_int_or_none(str(proc0.get("NumberOfCores"))) if proc0.get("NumberOfCores") is not None else None,
        "cpu_threads": _parse_int_or_none(str(proc0.get("NumberOfLogicalProcessors"))) if proc0.get("NumberOfLogicalProcessors") is not None else None,
        "ram_mib": ram_mib,
        "ram_gib": ram_gib,
        "igpu_pci": None,
        "igpu_model": igpu_model,
        "dgpu_pci": nvidia.pci_bus_id if nvidia else None,
        "dgpu_model": dgpu_model,
        "dgpu_arch": nvidia.dgpu_arch if nvidia else None,
        "dgpu_vram_mib": nvidia.vram_total_mib if nvidia else None,
        "nvme_model": nvme_model,
        "nvme_capacity_gb": nvme_capacity_gb,
        "secure_boot_enabled": secure_boot,
        "bitlocker_protection": bitlocker_summary,
        "dual_screen": {
            "connected_outputs": [],
            "dual_panel_detected": dual_detected,
            "asus_screenpad_sysfs_present": None,
            "asus_screenpad_brightness": None,
            "lid_open": None,
        },
        "nvidia_telemetry": nvidia.to_dict() if nvidia else None,
    }


def probe_macos(runner: CommandRunner = default_command_runner) -> dict[str, Any]:
    """Read-only macOS development-host probe using sysctl (never fabricates NVIDIA fields)."""

    def _sysctl(key: str) -> str | None:
        rc, out, _ = runner(["sysctl", "-n", key])
        if rc != 0:
            return None
        txt = out.strip()
        return txt if txt else None

    mem_bytes = _parse_int_or_none(_sysctl("hw.memsize"))
    ram_mib = mem_bytes // (1024 * 1024) if mem_bytes and mem_bytes > 0 else None
    ram_gib = round(mem_bytes / (1024 * 1024 * 1024)) if mem_bytes and mem_bytes > 0 else None
    cpu_cores = _parse_int_or_none(_sysctl("hw.physicalcpu"))
    cpu_threads = _parse_int_or_none(_sysctl("hw.logicalcpu"))
    cpu_brand = _sysctl("machdep.cpu.brand_string")
    hw_model = _sysctl("hw.model")
    macos_ver = _sysctl("kern.osproductversion")
    arch = _sysctl("hw.machine") or platform.machine() or None

    return {
        "schema": PROBE_SCHEMA,
        "host_os": "macos",
        "os_release": macos_ver,
        "architecture": arch,
        "manufacturer": "Apple Inc." if hw_model else None,
        "model": hw_model,
        "cpu_model": cpu_brand,
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "ram_mib": ram_mib,
        "ram_gib": ram_gib,
        "igpu_pci": None,
        "igpu_model": cpu_brand,
        "dgpu_pci": None,
        "dgpu_model": None,
        "dgpu_arch": None,
        "dgpu_vram_mib": None,
        "nvme_model": None,
        "secure_boot_enabled": None,
        "bitlocker_protection": None,
        "dual_screen": {
            "connected_outputs": [],
            "dual_panel_detected": False,
            "asus_screenpad_sysfs_present": None,
            "asus_screenpad_brightness": None,
            "lid_open": None,
        },
        "nvidia_telemetry": None,
    }


def probe_host(
    *,
    system_override: str | None = None,
    runner: CommandRunner = default_command_runner,
    linux_root: Path = Path("/"),
) -> dict[str, Any]:
    """Dispatch read-only hardware probing for Linux, Windows 11, or macOS."""
    sys_name = (system_override or platform.system()).strip().lower()
    if sys_name == "linux":
        return probe_linux(runner=runner, root=linux_root)
    if sys_name in ("windows", "win32", "nt"):
        return probe_windows(runner=runner)
    if sys_name in ("darwin", "macos"):
        return probe_macos(runner=runner)
    return {
        "schema": PROBE_SCHEMA,
        "host_os": sys_name or None,
        "os_release": None,
        "architecture": platform.machine() or None,
        "manufacturer": None,
        "model": None,
        "cpu_model": None,
        "cpu_cores": None,
        "cpu_threads": None,
        "ram_mib": None,
        "ram_gib": None,
        "igpu_pci": None,
        "igpu_model": None,
        "dgpu_pci": None,
        "dgpu_model": None,
        "dgpu_arch": None,
        "dgpu_vram_mib": None,
        "nvme_model": None,
        "secure_boot_enabled": None,
        "bitlocker_protection": None,
        "dual_screen": None,
        "nvidia_telemetry": None,
    }


def parse_windows_text_inventory(report_text: str) -> dict[str, Any]:
    """Parse a Windows pre-install text inventory report (such as Moissanite-Labs/MDE issue 4)."""
    result: dict[str, Any] = {
        "os_release": None,
        "manufacturer": None,
        "model": None,
        "cpu_model": None,
        "cpu_cores": None,
        "cpu_threads": None,
        "ram_gib": None,
        "ram_mib": None,
        "dgpu_model": None,
        "dgpu_vram_mib": None,
        "igpu_model": None,
        "nvme_model": None,
        "nvme_capacity_gb": None,
        "architecture": None,
    }
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        item = line.lstrip("-").strip()
        if item.startswith("OS:"):
            result["os_release"] = item.split(":", 1)[1].strip() or None
        elif item.startswith("Manufacturer:"):
            result["manufacturer"] = item.split(":", 1)[1].strip() or None
        elif item.startswith("Model:"):
            result["model"] = item.split(":", 1)[1].strip() or None
        elif item.startswith("Processor:"):
            result["cpu_model"] = item.split(":", 1)[1].strip() or None
        elif item.startswith("Cores/Threads:"):
            ct = re.findall(r"(\d+)", item)
            if len(ct) >= 2:
                result["cpu_cores"] = int(ct[0])
                result["cpu_threads"] = int(ct[1])
        elif item.startswith("Total RAM:"):
            gb_match = re.search(r"(\d+)\s*GB", item, re.IGNORECASE)
            mb_match = re.search(r"\(([\d,]+)\s*MB\)", item, re.IGNORECASE)
            if gb_match:
                result["ram_gib"] = int(gb_match.group(1))
            if mb_match:
                result["ram_mib"] = int(mb_match.group(1).replace(",", ""))
        elif "NVIDIA" in item:
            vram_m = re.search(r"\((\d+)\s*GB\s*VRAM\)", item, re.IGNORECASE)
            name_part = re.sub(r"\s*\(\d+\s*GB\s*VRAM\)\s*$", "", item, flags=re.IGNORECASE).strip()
            result["dgpu_model"] = name_part or None
            if vram_m:
                result["dgpu_vram_mib"] = int(vram_m.group(1)) * 1024
        elif "Intel" in item and ("Graphics" in item or "UHD" in item):
            name_part = re.sub(r"\s*\(integrated\)\s*$", "", item, flags=re.IGNORECASE).strip()
            result["igpu_model"] = name_part or None
        elif item.startswith("NVMe SSD:"):
            val = item.split(":", 1)[1].strip()
            tb_m = re.search(r"\((\d+)\s*TB\)", val, re.IGNORECASE)
            gb_m = re.search(r"\((\d+)\s*GB\)", val, re.IGNORECASE)
            model_only = re.sub(r"\s*\([^)]*\)\s*$", "", val).strip()
            result["nvme_model"] = model_only or None
            if tb_m:
                result["nvme_capacity_gb"] = int(tb_m.group(1)) * 1000
            elif gb_m:
                result["nvme_capacity_gb"] = int(gb_m.group(1))
        elif item.startswith("System Type:"):
            st = item.split(":", 1)[1].strip().lower()
            if "x64" in st or "x86_64" in st:
                result["architecture"] = "x86_64"
            elif "arm64" in st or "aarch64" in st:
                result["architecture"] = "aarch64"
    return result
