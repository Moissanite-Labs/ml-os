"""Cross-OS read-only hardware probe and deterministic compiler tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from ml_os.adapt import load_profile
from ml_os.compiler import compile_profile, verify_compiled_directory
from ml_os.probe import (
    parse_nvidia_smi_pids,
    parse_windows_text_inventory,
    probe_host,
    probe_linux,
    probe_macos,
    probe_windows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UX582_PROFILE = REPO_ROOT / "profiles" / "ux582hs-rtx3080-8gb.toml"
UX582_BLUEPRINTS = REPO_ROOT / "blueprints" / "ux582hs-rtx3080-8gb"

MDE_ISSUE4_REPORT = """
System
- OS: Microsoft Windows 11 Pro (Build 22621)
- Manufacturer: ASUSTeK COMPUTER INC.
- Model: ZenBook Pro Duo UX582HS

CPU
- Processor: 11th Gen Intel Core i9-11900H @ 2.50 GHz
- Cores/Threads: 8 cores / 16 threads
- Max Clock Speed: 2.50 GHz

Memory
- Total RAM: 32 GB (32,459 MB)
- Available: ~18.8 GB

Graphics
- NVIDIA GeForce RTX 3080 Laptop GPU (8 GB VRAM)
- Intel UHD Graphics (integrated)

Storage
- NVMe SSD: Samsung MZVL21T0HCLR (1 TB)

Architecture
- System Type: x64-based PC
"""


class TestProbeAndCompiler(unittest.TestCase):
    def test_mde_issue4_inventory_matches_ux582hs_profile(self) -> None:
        parsed_inv = parse_windows_text_inventory(MDE_ISSUE4_REPORT)
        profile, _ = load_profile(UX582_PROFILE)
        hw = profile["hardware"]

        self.assertEqual(parsed_inv["manufacturer"], hw["manufacturer"])
        self.assertEqual(parsed_inv["model"], hw["model"])
        self.assertEqual(parsed_inv["cpu_model"], hw["cpu_model"])
        self.assertEqual(parsed_inv["cpu_cores"], hw["cpu_cores"])
        self.assertEqual(parsed_inv["cpu_threads"], hw["cpu_threads"])
        self.assertEqual(parsed_inv["ram_gib"], hw["ram_gib"])
        self.assertEqual(parsed_inv["ram_mib"], hw["ram_mib"])
        self.assertEqual(parsed_inv["dgpu_model"], hw["dgpu_model"])
        self.assertEqual(parsed_inv["dgpu_vram_mib"], hw["dgpu_vram_mib"])
        self.assertEqual(parsed_inv["igpu_model"], hw["igpu_model"])
        self.assertEqual(parsed_inv["nvme_model"], hw["nvme_model"])
        self.assertEqual(parsed_inv["nvme_capacity_gb"], hw["nvme_capacity_gb"])
        self.assertEqual(parsed_inv["architecture"], hw["architecture"])

    def test_linux_probe_parses_ux582hs_sysfs_dual_screen_and_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "sys/class/dmi/id").mkdir(parents=True)
            (root / "sys/class/dmi/id/sys_vendor").write_text("ASUSTeK COMPUTER INC.\n", encoding="utf-8")
            (root / "sys/class/dmi/id/product_name").write_text("ZenBook Pro Duo UX582HS\n", encoding="utf-8")

            (root / "proc/acpi/button/lid/LID0").mkdir(parents=True)
            (root / "proc/acpi/button/lid/LID0/state").write_text("state:      open\n", encoding="utf-8")

            cpu_blocks = []
            for idx in range(16):
                cpu_blocks.append(
                    f"processor\t: {idx}\nmodel name\t: 11th Gen Intel Core i9-11900H @ 2.50GHz\n"
                    f"physical id\t: 0\ncore id\t\t: {idx % 8}\n"
                )
            (root / "proc/cpuinfo").write_text("\n".join(cpu_blocks), encoding="utf-8")
            (root / "proc/meminfo").write_text("MemTotal:       33238016 kB\n", encoding="utf-8")

            for card_conn, res in (("card0-eDP-1", "3840x2160"), ("card0-DP-1", "3840x1100")):
                cdir = root / "sys/class/drm" / card_conn
                cdir.mkdir(parents=True)
                (cdir / "status").write_text("connected\n", encoding="utf-8")
                (cdir / "modes").write_text(f"{res}\n", encoding="utf-8")

            led_subpath = (
                "sys/class/leds/asus__screenpad/brightness"
                if os.name == "nt"
                else "sys/class/leds/asus::screenpad/brightness"
            )
            led_file = root / led_subpath
            led_file.parent.mkdir(parents=True)
            led_file.write_text("64\n", encoding="utf-8")

            nvme_dir = root / "sys/class/nvme/nvme0"
            nvme_dir.mkdir(parents=True)
            (nvme_dir / "model").write_text("Samsung MZVL21T0HCLR\n", encoding="utf-8")

            lspci_calls: list[list[str]] = []

            def fake_runner(cmd: list[str]) -> tuple[int, str, str]:
                if cmd[0] == "lspci":
                    lspci_calls.append(list(cmd))
                    if cmd != ["lspci", "-Dnn"]:
                        return (1, "", "lspci: option requires an argument -- 's'")
                    return (
                        0,
                        "0000:00:02.0 VGA compatible controller [0300]: Intel Corporation TigerLake-H GT1 [UHD Graphics] [8086:9a60]\n"
                        "0000:01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA104M [GeForce RTX 3080 Mobile] [10de:249c]\n",
                        "",
                    )
                if cmd[0] == "nvidia-smi" and "--query-gpu" in cmd[1]:
                    return (
                        0,
                        "NVIDIA GeForce RTX 3080 Laptop GPU, 00000000:01:00.0, 565.77, 8.6, 8192, 0, 8192\n",
                        "",
                    )
                if cmd[0] == "nvidia-smi" and "--query-compute-apps" in cmd[1]:
                    return (0, "", "")
                if cmd == ["nvidia-smi", "-q", "-d", "PIDS"]:
                    return (
                        0,
                        "GPU 00000000:01:00.0\n    Processes                                          : None\n",
                        "",
                    )
                return (127, "", "not mocked")

            info = probe_linux(runner=fake_runner, root=root, screenpad_led_subpath=led_subpath)
            self.assertEqual(lspci_calls, [["lspci", "-Dnn"]])
            self.assertEqual(info["manufacturer"], "ASUSTeK COMPUTER INC.")
            self.assertEqual(info["model"], "ZenBook Pro Duo UX582HS")
            self.assertEqual(info["cpu_cores"], 8)
            self.assertEqual(info["cpu_threads"], 16)
            self.assertEqual(info["igpu_pci"], "0000:00:02.0")
            self.assertIn("Intel Corporation TigerLake-H GT1", info["igpu_model"])
            self.assertEqual(info["dgpu_pci"], "0000:01:00.0")
            self.assertEqual(info["dgpu_arch"], "sm_86")
            self.assertEqual(info["dgpu_vram_mib"], 8192)
            self.assertTrue(info["dual_screen"]["dual_panel_detected"])
            self.assertTrue(info["dual_screen"]["asus_screenpad_sysfs_present"])
            self.assertEqual(info["dual_screen"]["asus_screenpad_brightness"], 64)
            self.assertTrue(info["dual_screen"]["lid_open"])
            self.assertEqual(info["nvidia_telemetry"]["non_compute_desktop_vram_mib"], 0)
            self.assertTrue(info["nvidia_telemetry"]["gpu_processes_queried"])
            self.assertEqual(info["nvidia_telemetry"]["graphics_processes"], [])
            self.assertEqual(info["nvidia_telemetry"]["linux_kmd_baseline_vram_mib"], 0)

    def test_nvidia_pids_parser_and_omarchy_lua_reference(self) -> None:
        ok_none, procs_none = parse_nvidia_smi_pids(
            "GPU 00000000:01:00.0\n    Processes                                          : None\n"
        )
        self.assertTrue(ok_none)
        self.assertEqual(procs_none, ())

        ok_unsup, procs_unsup = parse_nvidia_smi_pids(
            "GPU 00000000:01:00.0\n    Processes                                          : Not Supported\n"
        )
        self.assertFalse(ok_unsup)
        self.assertEqual(procs_unsup, ())

        ok_missing, procs_missing = parse_nvidia_smi_pids("Attached GPUs : 1\n")
        self.assertFalse(ok_missing)
        self.assertEqual(procs_missing, ())

        ok_mixed, procs_mixed = parse_nvidia_smi_pids(
            "GPU 00000000:01:00.0\n"
            "    Processes\n"
            "        GPU instance ID                   : N/A\n"
            "        Compute instance ID               : N/A\n"
            "        Process ID                        : 4242\n"
            "            Type                          : G\n"
            "            Name                          : /usr/lib/Xorg\n"
            "            Used GPU Memory               : 18 MiB\n"
            "        Process ID                        : 5151\n"
            "            Type                          : C\n"
            "            Name                          : /usr/bin/python3\n"
            "            Used GPU Memory               : 512 MiB\n"
        )
        self.assertTrue(ok_mixed)
        self.assertEqual(len(procs_mixed), 2)
        self.assertEqual(procs_mixed[0]["type"], "G")
        self.assertEqual(procs_mixed[0]["used_gpu_memory_mib"], 18)
        self.assertEqual(procs_mixed[1]["type"], "C")

        ok_bad, _ = parse_nvidia_smi_pids(
            "GPU 00000000:01:00.0\n"
            "    Processes\n"
            "        Process ID                        : 777\n"
            "            Name                          : /usr/bin/missing-type\n"
        )
        self.assertFalse(ok_bad)

        omarchy_lua = REPO_ROOT / "docs" / "omarchy-4.5-ux582hs-monitors.lua"
        self.assertTrue(omarchy_lua.is_file())
        lua_txt = omarchy_lua.read_text(encoding="utf-8")
        self.assertIn('output = "eDP-1"', lua_txt)
        self.assertIn('output = "DP-3"', lua_txt)
        self.assertIn('mode = "3840x1100@60"', lua_txt)
        self.assertIn('position = "0x1080"', lua_txt)
        self.assertIn('name = "elan9009:00-04f3:2c23"', lua_txt)
        self.assertIn('name = "elan9008:00-04f3:2c56"', lua_txt)

    def test_windows_and_macos_probes_never_fabricate_missing_fields(self) -> None:
        win_json = json.dumps(
            {
                "ComputerSystem": {
                    "Manufacturer": "ASUSTeK COMPUTER INC.",
                    "Model": "ZenBook Pro Duo UX582HS",
                    "TotalPhysicalMemory": 34035728384,
                },
                "OperatingSystem": {
                    "Caption": "Microsoft Windows 11 Pro",
                    "BuildNumber": "22621",
                    "OSArchitecture": "64-bit",
                },
                "Processor": [
                    {
                        "Name": "11th Gen Intel Core i9-11900H @ 2.50 GHz",
                        "NumberOfCores": 8,
                        "NumberOfLogicalProcessors": 16,
                    }
                ],
                "VideoControllers": [
                    {"Name": "Intel UHD Graphics"},
                    {"Name": "NVIDIA GeForce RTX 3080 Laptop GPU"},
                ],
                "DiskDrives": [
                    {"Model": "Samsung MZVL21T0HCLR", "Size": 1000204886016},
                ],
                "SecureBoot": True,
                "BitLocker": [{"MountPoint": "C:", "ProtectionStatus": 1}],
            }
        )

        def win_runner(cmd: list[str]) -> tuple[int, str, str]:
            if cmd[0] in ("powershell", "pwsh"):
                return (0, win_json, "")
            if cmd[0] == "nvidia-smi" and "--query-gpu" in cmd[1]:
                return (
                    0,
                    "NVIDIA GeForce RTX 3080 Laptop GPU, 00000000:01:00.0, 566.14, 8.6, 8192, 480, 7712\n",
                    "",
                )
            if cmd[0] == "nvidia-smi" and "--query-compute-apps" in cmd[1]:
                return (0, "", "")
            return (127, "", "")

        winfo = probe_windows(runner=win_runner)
        self.assertEqual(winfo["host_os"], "windows")
        self.assertEqual(winfo["dgpu_arch"], "sm_86")
        self.assertEqual(winfo["nvidia_telemetry"]["non_compute_desktop_vram_mib"], 480)
        self.assertIsNone(winfo["igpu_pci"])

        def empty_mac_runner(cmd: list[str]) -> tuple[int, str, str]:
            return (127, "", "missing")

        minfo = probe_macos(runner=empty_mac_runner)
        self.assertIsNone(minfo["dgpu_model"])
        self.assertIsNone(minfo["dgpu_vram_mib"])
        self.assertIsNone(minfo["nvidia_telemetry"])
        self.assertIsNone(minfo["bitlocker_protection"])

        live = probe_host()
        self.assertEqual(live["schema"], "mlos-hardware-probe/v0.1")

    def test_compiler_zero_drift_and_dual_screen_blueprint_invariants(self) -> None:
        ok, mismatches = verify_compiled_directory(UX582_PROFILE, UX582_BLUEPRINTS)
        self.assertTrue(ok, msg=f"checked-in UX582HS blueprints drifted: {mismatches}")

        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            out1 = Path(td1) / "b1"
            out2 = Path(td2) / "b2"
            m1 = compile_profile(UX582_PROFILE, out1)
            m2 = compile_profile(UX582_PROFILE, out2)
            self.assertEqual(m1["artifacts_tree_sha256"], m2["artifacts_tree_sha256"])

            # Verify 1-byte tamper is caught fail-closed.
            tampered = out1 / "systemd" / "mlos-formal.slice"
            tampered.write_text(tampered.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
            ok_tamper, drift = verify_compiled_directory(UX582_PROFILE, out1)
            self.assertFalse(ok_tamper)
            self.assertTrue(any("DRIFT_ARTIFACT: systemd/mlos-formal.slice" in d for d in drift))

        # Verify critical architectural contents in UX582HS blueprints.
        mkosi_txt = (UX582_BLUEPRINTS / "mkosi" / "mkosi.conf").read_text(encoding="utf-8")
        self.assertIn("UnifiedKernelImages=yes", mkosi_txt)
        self.assertIn("lsm=landlock,lockdown,yama,integrity,apparmor,bpf", mkosi_txt)
        for line in mkosi_txt.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                self.assertNotEqual(stripped, "sudo")

        formal_txt = (UX582_BLUEPRINTS / "systemd" / "mlos-formal.slice").read_text(encoding="utf-8")
        self.assertIn("MemoryMax=22G", formal_txt)
        self.assertIn("DevicePolicy=closed", formal_txt)
        self.assertIn("leanprover/lean4:v4.35.0", formal_txt)

        llama_txt = (UX582_BLUEPRINTS / "systemd" / "mlos-llama-server.service").read_text(encoding="utf-8")
        self.assertIn("--host 127.0.0.1", llama_txt)
        self.assertIn("--lora-init-without-apply", llama_txt)
        self.assertIn("--n-gpu-layers 99", llama_txt)
        self.assertIn("IPAddressDeny=any", llama_txt)

        wayland_txt = (UX582_BLUEPRINTS / "display" / "hyprland-dual-screen-igpu.conf").read_text(encoding="utf-8")
        self.assertIn("AQ_DRM_DEVICES,/dev/dri/by-path/pci-0000:00:02.0-card", wayland_txt)
        self.assertIn("monitor = eDP-1,3840x2160@60,0x0,2", wayland_txt)
        self.assertIn("monitor = DP-1,3840x1100@60,0x1080,2", wayland_txt)
        self.assertIn("output = DP-1", wayland_txt)

        bash_bin = shutil.which("bash")
        if bash_bin is not None and os.name != "nt":
            for sh_file in UX582_BLUEPRINTS.rglob("*.sh"):
                res = subprocess.run([bash_bin, "-n", str(sh_file)], capture_output=True, text=True, check=False)
                self.assertEqual(res.returncode, 0, msg=f"bash -n failed on {sh_file}: {res.stderr}")

            sandbox_sh = UX582_BLUEPRINTS / "sandbox" / "mlos-sandbox-exec.sh"
            with tempfile.TemporaryDirectory() as work_td:
                env = {**os.environ, "MLOS_TOOL_EXECUTOR_REQUIRED": "0"}
                denied = subprocess.run(
                    [bash_bin, str(sandbox_sh), work_td, "echo", "blocked"],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=False,
                )
                self.assertEqual(denied.returncode, 126)
                self.assertIn("ERR_SANDBOX_POLICY", denied.stderr)


if __name__ == "__main__":
    unittest.main()
