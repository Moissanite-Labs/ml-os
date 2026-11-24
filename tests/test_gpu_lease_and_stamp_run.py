"""Cross-process GPU mutual-exclusion lease and provenance run-stamper tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from ml_os.gpu_lease import (
    GpuLeaseBusyError,
    GpuVramLeakError,
    GpuVramNotCleanError,
    acquire_gpu_lease,
    read_lease_status,
)
from ml_os.stamp_run import (
    StampAnomalyError,
    decode_clocks_event_reasons,
    evaluate_telemetry_anomalies,
    stamp_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
UX582_PROFILE = REPO_ROOT / "profiles" / "ux582hs-rtx3080-8gb.toml"


class TestGpuLeaseAndStampRun(unittest.TestCase):
    def test_gpu_lease_blocks_concurrent_process_fail_closed(self) -> None:
        holder_code = (
            "import sys, time; from pathlib import Path; "
            f"sys.path.insert(0, {str(SRC_DIR)!r}); "
            "from ml_os.gpu_lease import acquire_gpu_lease; "
            "lock = Path(sys.argv[1]); ready = Path(sys.argv[2]); stop = Path(sys.argv[3]); "
            "ctx = acquire_gpu_lease(lock_path=lock, mode='infer', vram_query=lambda: (0, [])); "
            "ctx.__enter__(); "
            "ready.write_text('held', encoding='utf-8'); "
            "\nwhile not stop.exists():\n    time.sleep(0.02)\n"
            "ctx.__exit__(None, None, None)"
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            lock_file = td_path / "gpu.lease.lock"
            ready_file = td_path / "ready.flag"
            stop_file = td_path / "stop.flag"

            proc = subprocess.Popen(
                [sys.executable, "-c", holder_code, str(lock_file), str(ready_file), str(stop_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 8.0
                while not ready_file.exists() and time.monotonic() < deadline:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertTrue(ready_file.exists(), msg="background holder failed to acquire GPU lease")

                status = read_lease_status(lock_file)
                self.assertTrue(status["held"])

                with self.assertRaises(GpuLeaseBusyError):
                    with acquire_gpu_lease(
                        lock_path=lock_file,
                        mode="train",
                        vram_query=lambda: (0, []),
                    ):
                        pass
            finally:
                stop_file.write_text("stop", encoding="utf-8")
                proc.communicate(timeout=8)

            self.assertEqual(proc.returncode, 0)
            after_status = read_lease_status(lock_file)
            self.assertFalse(after_status["held"])

    def test_gpu_lease_enforces_clean_entry_and_zero_lingering_vram(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock_file = Path(td) / "gpu.lease.lock"

            with self.assertRaises(GpuVramNotCleanError):
                with acquire_gpu_lease(
                    lock_path=lock_file,
                    mode="train",
                    vram_query=lambda: (512, [{"pid": 999, "process_name": "stale-llama", "used_gpu_memory_mib": 512}]),
                ):
                    pass

            calls = {"count": 0}

            def leaking_query() -> tuple[int, list[dict[str, object]]]:
                calls["count"] += 1
                if calls["count"] == 1:
                    return (0, [])
                return (256, [{"pid": 1001, "process_name": "zombie-cuda", "used_gpu_memory_mib": 256}])

            with self.assertRaises(GpuVramLeakError):
                with acquire_gpu_lease(
                    lock_path=lock_file,
                    mode="train",
                    vram_query=leaking_query,
                ):
                    pass

    def test_nvml_clocks_bitmask_decoder(self) -> None:
        idle = decode_clocks_event_reasons("0x0000000000000001")
        self.assertEqual(idle["decoded_reasons"], ["GpuIdle"])
        self.assertFalse(idle["thermal_throttled"])
        self.assertFalse(idle["slowdown_active"])

        app_lock = decode_clocks_event_reasons("0x0000000000000002")
        self.assertEqual(app_lock["decoded_reasons"], ["ApplicationsClocksSetting"])
        self.assertFalse(app_lock["slowdown_active"])

        therm = decode_clocks_event_reasons("0x0000000000000060")
        self.assertIn("SwThermalSlowdown", therm["decoded_reasons"])
        self.assertIn("HwThermalSlowdown", therm["decoded_reasons"])
        self.assertTrue(therm["thermal_throttled"])
        self.assertTrue(therm["slowdown_active"])

    def test_stamp_run_enforces_zone_pins_and_flags_wddm_or_thermal_anomalies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            env_dump = td_path / "child_env.json"
            receipt_file = td_path / "receipt.json"

            child_script = (
                "import json, os, pathlib, sys; "
                "keys = ['PYTHONHASHSEED', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', "
                "'CUBLAS_WORKSPACE_CONFIG', 'HF_HUB_OFFLINE', 'MLOS_DETERMINISM_ZONE']; "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps({k: os.environ.get(k) for k in keys}), encoding='utf-8')"
            )

            def clean_gpu_runner(cmd: list[str]) -> tuple[int, str, str]:
                if cmd[0] == "nvidia-smi" and "--query-gpu" in cmd[1]:
                    return (
                        0,
                        "NVIDIA GeForce RTX 3080 Laptop GPU, 565.77, 94.04.48, P0, 1545, 7000, 88.4, 95.0, 71, 0, 8192, 0x0000000000000001\n",
                        "",
                    )
                if cmd[0] == "nvidia-smi" and "--query-compute-apps" in cmd[1]:
                    return (0, "", "")
                if cmd[0] == "git":
                    return (0, "a" * 40 + "\n", "")
                return (127, "", "")

            receipt = stamp_run(
                command=[sys.executable, "-c", child_script, str(env_dump)],
                profile_path=UX582_PROFILE,
                zone="B",
                output_path=receipt_file,
                strict_anomalies=True,
                runner=clean_gpu_runner,
            )
            self.assertTrue(receipt["clean_hardware_run"])
            self.assertEqual(receipt["anomalies"], [])
            self.assertEqual(receipt["git_head"], "a" * 40)
            self.assertTrue(receipt_file.is_file())

            captured_env = json.loads(env_dump.read_text(encoding="utf-8"))
            self.assertEqual(captured_env["PYTHONHASHSEED"], "0")
            self.assertEqual(captured_env["OMP_NUM_THREADS"], "1")
            self.assertEqual(captured_env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual(captured_env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(captured_env["MLOS_DETERMINISM_ZONE"], "B")

            # Now simulate WDDM desktop VRAM > 0 MiB (540 MiB) + thermal throttling (0x20).
            def throttled_wddm_runner(cmd: list[str]) -> tuple[int, str, str]:
                if cmd[0] == "nvidia-smi" and "--query-gpu" in cmd[1]:
                    return (
                        0,
                        "NVIDIA GeForce RTX 3080 Laptop GPU, 566.14, 94.04.48, P0, 990, 7000, 94.9, 95.0, 87, 540, 7652, 0x0000000000000020\n",
                        "",
                    )
                if cmd[0] == "nvidia-smi" and "--query-compute-apps" in cmd[1]:
                    return (0, "", "")
                return (127, "", "")

            with self.assertRaises(StampAnomalyError):
                stamp_run(
                    command=[sys.executable, "-c", "pass"],
                    profile_path=UX582_PROFILE,
                    zone="B",
                    strict_anomalies=True,
                    runner=throttled_wddm_runner,
                )

            flagged = stamp_run(
                command=[sys.executable, "-c", "pass"],
                profile_path=UX582_PROFILE,
                zone="A",
                strict_anomalies=False,
                runner=throttled_wddm_runner,
            )
            self.assertFalse(flagged["clean_hardware_run"])
            self.assertTrue(any(a.startswith("WDDM_OR_DESKTOP_VRAM_NONZERO") for a in flagged["anomalies"]))
            self.assertTrue(any(a.startswith("THERMAL_THROTTLE_ACTIVE") for a in flagged["anomalies"]))

            # CPU-only host without nvidia-smi records explicit None (zero fake telemetry).
            no_gpu = stamp_run(
                command=[sys.executable, "-c", "pass"],
                profile_path=UX582_PROFILE,
                zone="A",
                runner=lambda cmd: (127, "", "missing"),
            )
            self.assertFalse(no_gpu["gpu_telemetry_available"])
            self.assertIsNone(no_gpu["start_gpu_telemetry"])
            self.assertIsNone(no_gpu["end_gpu_telemetry"])

    def test_linux_kmd_baseline_vram_and_fail_closed_desktop_process_guards(self) -> None:
        pids_none = "GPU 00000000:01:00.0\n    Processes                                          : None\n"
        pids_xorg = (
            "GPU 00000000:01:00.0\n"
            "    Processes\n"
            "        Process ID                        : 1200\n"
            "            Type                          : G\n"
            "            Name                          : /usr/lib/Xorg\n"
            "            Used GPU Memory               : 2 MiB\n"
        )
        pids_hybrid = (
            "GPU 00000000:01:00.0\n"
            "    Processes\n"
            "        Process ID                        : 1300\n"
            "            Type                          : C+G\n"
            "            Name                          : /usr/bin/compositor\n"
            "            Used GPU Memory               : 0 MiB\n"
        )

        def make_runner(used_mib: int, pids_rc: int, pids_out: str) -> object:
            def _runner(cmd: list[str]) -> tuple[int, str, str]:
                if cmd[0] == "nvidia-smi" and "--query-gpu" in cmd[1]:
                    free_mib = max(0, 8192 - used_mib)
                    return (
                        0,
                        f"NVIDIA GeForce RTX 3080 Laptop GPU, 615.78.08, 94.04.49.00.51, P0, 1245, 6001, 20.1, N/A, 55, {used_mib}, {free_mib}, 0x0000000000000001\n",
                        "",
                    )
                if cmd[0] == "nvidia-smi" and "--query-compute-apps" in cmd[1]:
                    return (0, "", "")
                if cmd == ["nvidia-smi", "-q", "-d", "PIDS"]:
                    return (pids_rc, pids_out, "")
                return (127, "", "")

            return _runner

        # 1. Verified Linux KMD zero-process baseline (2 MiB and exact 4 MiB boundary) -> clean_hardware_run=True.
        for baseline_mib in (2, 4):
            rec = stamp_run(
                command=[sys.executable, "-c", "pass"],
                profile_path=UX582_PROFILE,
                zone="B",
                strict_anomalies=True,
                runner=make_runner(baseline_mib, 0, pids_none),  # type: ignore[arg-type]
                host_os="linux",
            )
            self.assertTrue(rec["clean_hardware_run"])
            self.assertEqual(rec["anomalies"], [])
            self.assertTrue(rec["start_gpu_telemetry"]["gpu_processes_queried"])
            self.assertEqual(rec["start_gpu_telemetry"]["graphics_processes"], [])
            self.assertEqual(rec["start_gpu_telemetry"]["linux_kmd_baseline_vram_mib"], baseline_mib)

        # 2. Must-fail: Linux zero-process > 4 MiB (5 MiB) -> WDDM_OR_DESKTOP_VRAM_NONZERO.
        with self.assertRaises(StampAnomalyError):
            stamp_run(
                command=[sys.executable, "-c", "pass"],
                profile_path=UX582_PROFILE,
                zone="B",
                strict_anomalies=True,
                runner=make_runner(5, 0, pids_none),  # type: ignore[arg-type]
                host_os="linux",
            )

        # 3. Must-fail: Linux 2 MiB when PIDS query fails or is Not Supported -> fail-closed!
        for rc_bad, out_bad in ((1, ""), (0, "GPU 00000000:01:00.0\n    Processes : Not Supported\n")):
            rec_unverified = stamp_run(
                command=[sys.executable, "-c", "pass"],
                profile_path=UX582_PROFILE,
                zone="B",
                strict_anomalies=False,
                runner=make_runner(2, rc_bad, out_bad),  # type: ignore[arg-type]
                host_os="linux",
            )
            self.assertFalse(rec_unverified["clean_hardware_run"])
            self.assertFalse(rec_unverified["start_gpu_telemetry"]["gpu_processes_queried"])
            self.assertEqual(rec_unverified["start_gpu_telemetry"]["linux_kmd_baseline_vram_mib"], 0)
            self.assertTrue(any(a.startswith("WDDM_OR_DESKTOP_VRAM_NONZERO") for a in rec_unverified["anomalies"]))

        # 4. Must-fail: Real X11/Wayland Type G process even at 2 MiB -> BOTH alerts fire!
        rec_xorg = stamp_run(
            command=[sys.executable, "-c", "pass"],
            profile_path=UX582_PROFILE,
            zone="B",
            strict_anomalies=False,
            runner=make_runner(2, 0, pids_xorg),  # type: ignore[arg-type]
            host_os="linux",
        )
        self.assertFalse(rec_xorg["clean_hardware_run"])
        self.assertTrue(any(a.startswith("DESKTOP_GPU_PROCESS_ACTIVE") for a in rec_xorg["anomalies"]))
        self.assertTrue(any(a.startswith("WDDM_OR_DESKTOP_VRAM_NONZERO") for a in rec_xorg["anomalies"]))

        # 5. Must-fail: Hybrid C+G desktop process even at 0 MiB VRAM -> DESKTOP_GPU_PROCESS_ACTIVE fires!
        rec_hybrid = stamp_run(
            command=[sys.executable, "-c", "pass"],
            profile_path=UX582_PROFILE,
            zone="B",
            strict_anomalies=False,
            runner=make_runner(0, 0, pids_hybrid),  # type: ignore[arg-type]
            host_os="linux",
        )
        self.assertFalse(rec_hybrid["clean_hardware_run"])
        self.assertTrue(any(a.startswith("DESKTOP_GPU_PROCESS_ACTIVE") for a in rec_hybrid["anomalies"]))

        # 6. Must-fail: Windows WDDM host NEVER receives the Linux KMD <= 4 MiB allowance!
        rec_win = stamp_run(
            command=[sys.executable, "-c", "pass"],
            profile_path=UX582_PROFILE,
            zone="B",
            strict_anomalies=False,
            runner=make_runner(2, 0, pids_none),  # type: ignore[arg-type]
            host_os="windows",
        )
        self.assertFalse(rec_win["clean_hardware_run"])
        self.assertEqual(rec_win["start_gpu_telemetry"]["linux_kmd_baseline_vram_mib"], 0)
        self.assertTrue(any(a.startswith("WDDM_OR_DESKTOP_VRAM_NONZERO") for a in rec_win["anomalies"]))

        # 7. Direct evaluate_telemetry_anomalies check with bool non_compute_desktop_vram_mib (no bool-as-int bypass).
        anoms_clean = evaluate_telemetry_anomalies(
            start_gpu={
                "non_compute_desktop_vram_mib": 2,
                "gpu_processes_queried": True,
                "graphics_processes": [],
                "clocks_event_reasons": {},
            },
            end_gpu=None,
            max_desktop_vram_mib=0,
            host_os="linux",
        )
        self.assertEqual(anoms_clean, [])


if __name__ == "__main__":
    unittest.main()
