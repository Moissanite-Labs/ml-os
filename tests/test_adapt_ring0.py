"""Adversarial Ring 0 invariant and Ring 2 adaptation tests for mlos-adapt."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ml_os.adapt import (
    AdaptationRejectedError,
    apply_adaptation,
    check_adaptation,
    validate_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UX582_PROFILE = REPO_ROOT / "profiles" / "ux582hs-rtx3080-8gb.toml"
VMTEST_PROFILE = REPO_ROOT / "profiles" / "vmtest-qemu-x86_64.toml"


class TestRing0AndAdaptation(unittest.TestCase):
    def test_reference_profiles_pass_all_invariants(self) -> None:
        ux_report = validate_profile(UX582_PROFILE)
        self.assertTrue(ux_report.passed, msg=f"UX582HS violations: {ux_report.violations}")
        self.assertEqual(ux_report.profile_id, "ux582hs-rtx3080-8gb")

        vm_report = validate_profile(VMTEST_PROFILE)
        self.assertTrue(vm_report.passed, msg=f"vmtest violations: {vm_report.violations}")
        self.assertEqual(vm_report.profile_id, "vmtest-qemu-x86_64")

    def test_adversarial_ring2_diffs_rejected_with_zero_disk_writes(self) -> None:
        base_text = UX582_PROFILE.read_text(encoding="utf-8")

        adversarial_mutations: tuple[tuple[str, str, str], ...] = (
            ("allow_sudo_true", "allow_sudo = false", "allow_sudo = true"),
            ("allow_sudo_int_zero", "allow_sudo = false", "allow_sudo = 0"),
            ("disable_sandbox", "tool_executor_required = true", "tool_executor_required = false"),
            ("nonzero_desktop_vram", "dgpu_desktop_vram_mib = 0", "dgpu_desktop_vram_mib = 256"),
            ("disable_gpu_lease", "exclusive_gpu_lease = true", "exclusive_gpu_lease = false"),
            (
                "unlock_bare_metal_repartition",
                "bare_metal_repartition_unlocked = false",
                "bare_metal_repartition_unlocked = true",
            ),
            ("drop_landlock_lsm", '    "landlock",\n', ""),
            ("drop_readonly_models", '    "/var/lib/mlos/models",\n', ""),
            ("drop_snapshots_subvol", '    "@snapshots",\n', ""),
            ("disable_uki_rollback", "require_uki_rollback = true", "require_uki_rollback = false"),
            (
                "trojan_sudo_in_ring2_packages",
                'extra_packages = ["tmux", "ripgrep", "fd"]',
                'extra_packages = ["tmux", "sudo"]',
            ),
            (
                "trojan_doas_in_ring2_packages",
                'extra_packages = ["tmux", "ripgrep", "fd"]',
                'extra_packages = ["doas"]',
            ),
            (
                "dgpu_display_mode",
                'display_mode = "headless-ssh"',
                'display_mode = "nvidia-dgpu"',
            ),
            (
                "disable_dgpu_leak_fallback",
                "fallback_to_headless_on_dgpu_leak = true",
                "fallback_to_headless_on_dgpu_leak = false",
            ),
            (
                "disable_per_panel_touch_binding",
                "bind_touch_per_connector = true",
                "bind_touch_per_connector = false",
            ),
            (
                "unapproved_sm_clock_lock",
                'thermal_profile = "sustained-train"',
                'thermal_profile = "benchmark-locked"',
            ),
            (
                "exceed_hardware_power_ceiling",
                "\ndgpu_power_cap_w = 95\nowner_clock_lock_approved = false",
                "\ndgpu_power_cap_w = 150\nowner_clock_lock_approved = false",
            ),
            (
                "mutate_formal_slice_dgpu",
                "unit = \"mlos-formal.slice\"\nmemory_high_gib = 16\nmemory_max_gib = 22\ntasks_max = 512\noom_policy = \"kill\"\nallow_dgpu = false",
                "unit = \"mlos-formal.slice\"\nmemory_high_gib = 16\nmemory_max_gib = 22\ntasks_max = 512\noom_policy = \"kill\"\nallow_dgpu = true",
            ),
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            active_copy = td_path / "active.toml"
            active_copy.write_text(base_text, encoding="utf-8")
            original_bytes = active_copy.read_bytes()
            ledger_file = td_path / "ledger.jsonl"

            for label, target, replacement in adversarial_mutations:
                self.assertIn(target, base_text, msg=f"fixture target missing for {label}")
                mutated = base_text.replace(target, replacement, 1)
                cand_path = td_path / f"cand-{label}.toml"
                cand_path.write_text(mutated, encoding="utf-8")

                report = check_adaptation(active_copy, cand_path)
                self.assertFalse(report.passed, msg=f"adversarial case {label} was not rejected")
                self.assertGreater(len(report.violations), 0, msg=f"no violations recorded for {label}")

                with self.assertRaises(AdaptationRejectedError, msg=f"apply_adaptation did not raise for {label}"):
                    apply_adaptation(
                        active_profile_path=active_copy,
                        candidate_profile_path=cand_path,
                        ledger_path=ledger_file,
                    )

                self.assertEqual(
                    active_copy.read_bytes(),
                    original_bytes,
                    msg=f"disk write occurred on active profile during rejected case {label}",
                )
                self.assertFalse(
                    ledger_file.exists(),
                    msg=f"ledger file created during rejected case {label}",
                )

    def test_valid_ring2_adaptations_apply_and_hash_chain_ledger(self) -> None:
        base_text = UX582_PROFILE.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            active = td_path / "active.toml"
            ledger = td_path / "adaptation-ledger.jsonl"
            active.write_text(base_text, encoding="utf-8")

            # Adaptation 1: switch headless-ssh -> dual-screen-igpu + quiet-edit (70 W)
            cand1 = td_path / "cand1.toml"
            text1 = (
                base_text.replace('display_mode = "headless-ssh"', 'display_mode = "dual-screen-igpu"', 1)
                .replace('thermal_profile = "sustained-train"', 'thermal_profile = "quiet-edit"', 1)
                .replace(
                    "\ndgpu_power_cap_w = 95\nowner_clock_lock_approved = false",
                    "\ndgpu_power_cap_w = 70\nowner_clock_lock_approved = false",
                    1,
                )
            )
            cand1.write_text(text1, encoding="utf-8")

            r1 = apply_adaptation(
                active_profile_path=active,
                candidate_profile_path=cand1,
                ledger_path=ledger,
            )
            self.assertEqual(r1["prev_entry_sha256"], "0" * 64)
            self.assertIn("ring2_user.display_mode", r1["changed_ring2_paths"])
            self.assertEqual(active.read_bytes(), cand1.read_bytes())

            # Adaptation 2: owner approves SM clock lock and selects benchmark-locked (95 W)
            cand2 = td_path / "cand2.toml"
            text2 = (
                text1.replace('thermal_profile = "quiet-edit"', 'thermal_profile = "benchmark-locked"', 1)
                .replace(
                    "\ndgpu_power_cap_w = 70\nowner_clock_lock_approved = false",
                    "\ndgpu_power_cap_w = 95\nowner_clock_lock_approved = true",
                    1,
                )
            )
            cand2.write_text(text2, encoding="utf-8")

            r2 = apply_adaptation(
                active_profile_path=active,
                candidate_profile_path=cand2,
                ledger_path=ledger,
            )
            self.assertEqual(r2["prev_entry_sha256"], r1["entry_sha256"])
            lines = [ln for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
