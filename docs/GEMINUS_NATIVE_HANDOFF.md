# Native Development Handoff for `geminus` (UX582HS)

This document specifies how agents running directly on `geminus` (ASUS ZenBook Pro Duo UX582HS, Windows 11 Pro Build 22621 + WSL2 / Linux dual-boot target) take over native hardware bring-up and ongoing `ml-os` development from inside the target machine.

## Seed Baseline Ready Today

`ml-os` v0.1 ships a zero-dependency Python 3.11+ control plane (`bin/mlos`), locked Ring 0 validator (`src/ml_os/adapt.py`), read-only cross-OS probe (`src/ml_os/probe.py`), deterministic blueprint compiler (`src/ml_os/compiler.py`), exclusive GPU lease (`src/ml_os/gpu_lease.py`), provenance run stamper (`src/ml_os/stamp_run.py`), compiled UX582HS blueprints (`blueprints/ux582hs-rtx3080-8gb/`), and Stage 0 Windows 11 PowerShell bootstrap helpers (`scripts/`).

Every native phase below preserves two hard rules:
1. Never resize partitions, format drives, or touch Windows boot sectors without explicit human owner checklist sign-off (`bare_metal_repartition_unlocked = false`).
2. Scrub private hostnames, local account paths, and VPN or LAN IP addresses from every committed receipt before pushing.

## Phase 1: Stage 0 Windows 11 Inventory and WSL2 CUDA `sm_86` Bring-Up (Zero Repartitioning)

Run directly on `geminus` under Windows 11 + WSL2 so CUDA `sm_86` `llama-server`, Lean `v4.35.0`, and `mlos` provenance stamping work immediately:

1. Capture the read-only Phase 0 hardware, BitLocker, Secure Boot, NVMe partition, and WDDM VRAM baseline:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\stage0-win11-wsl2-bootstrap.ps1 `
       -CheckOnly -ReceiptPath .\stage0-preflight.json
   python .\bin\mlos probe --json
   ```
2. Measure Windows WDDM desktop VRAM overhead (`wddm_desktop_vram_mib`) on the `8,192 MiB` RTX 3080 Laptop GPU. Close background Windows graphics consumers so `Qwen3.5-4B` (`Q8_0` or `Q4_K_M`) plus 3 F16 LoRAs and `16k` KV cache stay strictly inside the `7.2 GiB` (`7,372 MiB`) `T-C5` VRAM ceiling.
3. Apply the compiled WSL2 cap (`24 GiB` RAM, `16` threads, `8 GiB` swap) and bootstrap pinned `leanprover/lean4:v4.35.0` plus offline `sm_86` `llama-server` inside WSL2:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\stage0-win11-wsl2-bootstrap.ps1 `
       -WriteWslConfig -BootstrapWslToolchain
   ```
4. Inside WSL2, run `python3 -O -B -W error -m pytest -v` and stamp the live `T-C5` RTX 3080 inference benchmark using `mlos gpu-lease` and `mlos stamp-run --zone B`.

## Phase 2: Unprivileged `mkosi` Build and QEMU/KVM Gate

Before booting any custom Linux image on physical hardware, validate the compiled blueprints inside QEMU/KVM on `geminus` (WSL2 nested KVM or Linux host):

1. Compile `profiles/vmtest-qemu-x86_64.toml` (`python3 ./bin/mlos compile --profile profiles/vmtest-qemu-x86_64.toml --out-dir build/vmtest`) and build the disk image unprivileged with `mkosi`.
2. Boot under QEMU/KVM and verify four automated gates:
   - Zero `sudo`/`doas`/`pkexec` binaries in the image, `root:root` plus `go-w` permissions across `/etc/mlos/canaries`, `/var/lib/mlos/models`, and `/usr/lib/systemd/system`, and active loopback-only `mlos-llama-server.service` + `mlos-modeld.service`.
   - Mandatory `mlos-sandbox-exec.sh` (`bubblewrap --unshare-net`) blocking network egress, key reads, and read-only `@models` writes across 30 adversarial tool-execution trials.
   - Two-reboot Btrfs `@` snapshot + archived UKI rollback (`mlos-recovery.sh checkpoint` -> root fault injection -> `stage-rollback` -> clean recovery) preserving `@models`, `@experiments`, `@ledger`, and `@workspace`.
   - `mlos adapt --check` rejecting Ring 0 weakening diffs and applying valid Ring 2 edits cleanly.

## Phase 3: Non-Destructive Live USB, Dual-Screen `ScreenPad Plus` UX, and Thermal Calibration

Boot a read-only live USB on `geminus` (zero writes to the internal Samsung `MZVL21T0HCLR` 1 TB NVMe) to verify physical Linux drivers and tune UX582HS hardware parameters:

1. Record exact Linux PCI addresses (`lspci -Dnn`), DRM connector names (`/sys/class/drm/card*-*/status`), ELAN touch input device names (`libinput list-devices`), `/sys/class/leds/asus::screenpad/brightness` range, and `/proc/acpi/button/lid/*/state` paths. On live Omarchy 4.5.0 UX582HS hardware, ScreenPad Plus enumerates on `DP-3` (`3840x1100@60`, touch `elan9009:00-04f3:2c23`, primary OLED touch `elan9008:00-04f3:2c56`), see `docs/omarchy-4.5-ux582hs-monitors.lua`.
2. Verify iGPU-only Wayland output keeps zero graphics or desktop processes attached to the RTX 3080 (`nvidia-smi -q -d PIDS` reports `Processes : None`). Note that Linux NVIDIA KMD `615.78.08` holds `2 MiB` of internal driver metadata with zero active processes; `mlos stamp-run` accepts verified zero-process Linux KMD metadata up to `4 MiB` while failing closed on any attached desktop process or `> 4 MiB` overhead.
3. Run a 15-minute sustained GPU load with the lid open (`AAS Plus` intake active) under `sustained-train` (`95 W`) wrapped by `mlos stamp-run --strict-anomalies`. If `clocks_event_reasons` reports `SwThermalSlowdown` (`0x20`) or `HwThermalSlowdown` (`0x40`), lower `dgpu_power_cap_w` in `profiles/ux582hs-rtx3080-8gb.toml` to the highest throttle-free wattage and commit the measured CSV receipt.

## Phase 4: Owner-Signed Dual-Boot Btrfs Bring-Up and Full Workbench Certification

Only after Phase 2 and Phase 3 pass and the human owner signs off on the partition layout:

1. Install `ml-os` onto the dedicated Btrfs partition alongside the untouched Windows 11 partition (`@`, `@var`, `@workspace`, `@ledger`, `@models` read-only, `@experiments`, `@snapshots`, plus `8 GiB` `zram`).
2. Run the full L5 (`mde`), L4 (`exo`), L3 (`warp-modeld` CUDA live smoke), L2 (`weft` deterministic CPU `float64` + live `llama-server` LoRA), and `mlos-formal.slice` Lean `v4.35.0` verification suites under `mlos stamp-run`, then publish the signed readiness receipt.
