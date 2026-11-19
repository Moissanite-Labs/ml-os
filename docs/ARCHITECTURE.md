# ml-os v0.1 Architecture

`ml-os` compiles a machine-specific experimentation operating system around a fixed Ring 0 safety core. It implements the adaptive workbench architecture tracked in [siliconcordat/warp issue 4](https://github.com/siliconcordat/warp/issues/4) against the measured ASUS ZenBook Pro Duo UX582HS baseline recorded on the [Moissanite-Labs/MDE issue tracker](https://github.com/Moissanite-Labs/MDE/issues).

## 1. Concentric Ring Contract

| Ring | Scope | Mutability | Enforcement Mechanism |
| --- | --- | --- | --- |
| **Ring 0** | Kernel LSM stack, zero `sudo`, mandatory `bubblewrap`/Landlock tool execution, `0 MiB` discrete GPU desktop VRAM, atomic GPU lease, read-only `@models`, Btrfs `@`/`@snapshots` + UKI rollback, bare-metal repartition lock | Immutable across all user adaptations | `mlos adapt` rejects any `[ring0_locked]` edit, privilege package, or dGPU display mode with exit code `1` and zero disk writes |
| **Ring 1** | CPU/RAM/GPU/NVMe topology, pinned Arch Linux `mkosi` snapshot, systemd cgroup v2 slices, display connector bindings, ASUS thermal curves, Stage 0 WSL2 bridge | Recompiled deterministically from measured hardware profile | `mlos probe` captures read-only host facts (`None` when absent). `mlos compile` verifies `0`-byte drift via SHA-256 tree manifests |
| **Ring 2** | Display mode (`headless-ssh`, `dual-screen-igpu`, `single-primary-igpu`), active thermal profile, power cap within hardware ceiling, remote tunnel, active domain packs | Safe declarative diffs under `[ring2_user]` | `mlos adapt --apply` validates invariants first, updates atomically via `fsync` + `replace`, and appends a hash-chained receipt |

## 2. UX582HS Memory, VRAM, and Slice Topology

The reference profile (`profiles/ux582hs-rtx3080-8gb.toml`) targets the 8-core / 16-thread Intel Core i9-11900H with `32 GiB` (`32,459 MiB`) host RAM, `8 GiB` compressed `zram` swap, and an `8 GiB` (`8,192 MiB`) NVIDIA RTX 3080 Laptop GPU (`sm_86`):

- **`mlos-formal.slice`** (`MemoryHigh=16G`, `MemoryMax=22G`, `TasksMax=512`, `DevicePolicy=closed`): Isolates `leanprover/lean4:v4.35.0` builds and proof comparators so runaway elaboration is terminated cleanly before threatening SSH or kernel daemons, with zero access to `/dev/nvidia*` (`0 MiB` VRAM).
- **`mlos-infer.slice`** (`MemoryMin=2G`, `MemoryHigh=12G`, `MemoryMax=16G`, `OOMScoreAdjust=-500`): Hosts loopback-only `llama-server` (`127.0.0.1:8080`, `--lora-init-without-apply`, `--slots`, `--no-cache-idle-slots`) behind `warp-modeld`.
- **`mlos-compute.slice`** (`MemoryHigh=24G`, `MemoryMax=28G`): Runs serialized CPU `float64` `BetaEdit` null-space compilations (`max_parallel_large_svd_workers = 1` for `d_in >= 8192`) and CUDA harvesting jobs under `mlos gpu-lease`.
- **`mlos-agents.slice`** (`MemoryHigh=8G`, `MemoryMax=12G`, `CPUQuota=600%`, `DevicePolicy=closed`): Bounds untrusted tool sandboxes running inside `mlos-sandbox-exec.sh` (`bubblewrap --unshare-net`).

## 3. Dual-Screen (`ScreenPad Plus`) Hardware and UX Mechanics

The ASUS ZenBook Pro Duo UX582HS pairs a primary 15.6-inch `3840x2160` OLED touchscreen (`eDP-1`) with a secondary tilted 14-inch `3840x1100` matte IPS `ScreenPad Plus` touchscreen (`DP-1`). Ignoring dual-panel quirks breaks either GPU memory isolation or local ergonomics. `ml-os` compiles five concrete safeguards into `blueprints/ux582hs-rtx3080-8gb/display/` and `thermal/`:

1. **Zero-VRAM Intel iGPU Pinning**: `AQ_DRM_DEVICES` and `WLR_DRM_DEVICES` bind strictly to `/dev/dri/by-path/pci-0000:00:02.0-card` (`__NV_PRIME_RENDER_OFFLOAD=0`). Both `3840`-wide panels scan out from the Intel Tiger Lake iGPU so the RTX 3080 keeps `0 MiB` desktop VRAM. If `mlos-dgpu-zero-vram-guard.sh` observes `> 0 MiB` non-compute VRAM, it isolates `multi-user.target` (`headless-ssh` fallback) rather than surrendering CUDA pool capacity.
2. **Seamless Vertical Stacking**: Both displays share a `3840`-pixel width (`1920` logical pixels at `scale = 2`). Placing `eDP-1` at `0x0` and `DP-1` directly beneath at `0x1080` preserves continuous vertical cursor travel with zero dead corners.
3. **Per-Connector Touch and Stylus Binding**: Separate Wayland `device` blocks bind the upper and lower ELAN I2C touch digitizers to `eDP-1` and `DP-1` respectively. Without per-connector binding, a tap on `ScreenPad Plus` scales across the full `3260`-pixel virtual height and clicks the upper OLED panel instead.
4. **Asymmetric OLED Burn-In vs Matte IPS Telemetry Split**: Multi-hour formal verification or training sweeps risk static burn-in on the primary `eDP-1` OLED. `ml-os` enforces `oled_idle_blank_sec = 180` on `eDP-1` while routing persistent low-brightness telemetry (`nvidia-smi` power/VRAM, `mlos gpu-lease` state, `mlos-formal.slice` pressure) to the matte IPS `ScreenPad Plus` (`screenpad_role = "ips-telemetry-strip"`), with independent `/sys/class/leds/asus::screenpad/brightness` sysfs control.
5. **AAS Plus Hinge Intake and Lid-State Power Clamp**: Opening the UX582HS lid tilts `ScreenPad Plus` upward by 9.5 degrees to open the primary cooling intake above the CPU/GPU vapor chamber. Running `95 W` `sustained-train` or `benchmark-locked` with the lid closed chokes intake airflow and triggers thermal slowdown. `mlos-thermal-apply.sh` reads `/proc/acpi/button/lid/*/state` and automatically clamps closed-lid operation down to `quiet-edit` (`70 W`, `ScreenPad` LED `0`).

## 4. Exclusive GPU Lease and Provenance Run Stamper

- **`mlos gpu-lease`**: Uses non-blocking OS file locks (`fcntl.flock` on Linux/macOS, `msvcrt.locking` on Windows) paired with systemd `Conflicts=mlos-gpu-train.target`. Pre-entry checks reject dirty compute VRAM (`> 0 MiB`), and a `finally` exit gate raises `GpuVramLeakError` if any child process leaves `> 0 MiB` compute VRAM behind.
- **`mlos stamp-run`**: Applies Zone A (`PYTHONHASHSEED=0`, single-thread BLAS) and Zone B (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`, `TORCH_USE_DETERMINISTIC_ALGORITHMS=1`) pins without clearing OS system variables.
  * Receipts record `profile_sha256`, normalized `cpu_isa`, `git_head`, and start/end `nvidia-smi` CSV snapshots.
  * The NVML `clocks_event_reasons.active` bitmask decoder separates benign `GpuIdle` (`0x1`) and `ApplicationsClocksSetting` (`0x2`) states from real thermal (`0x20`, `0x40`) and power/hardware (`0x4`, `0x8`, `0x80`) slowdowns, and flags any WDDM or desktop VRAM above `0 MiB`.

## 5. What Is Not Verified Yet

This v0.1 seed verifies Ring 0 validation, cross-OS probe logic, deterministic blueprint compilation (`22` artifacts, `0`-byte drift), cross-platform GPU lease locking, and NVML telemetry decoding across Linux, Windows, and macOS CI runners.

Live execution on physical UX582HS hardware (`geminus`), QEMU/KVM `mkosi` boot gates, 15-minute `95 W` thermal sweeps, and bare-metal Btrfs/UKI dual-boot installation remain unexecuted pending native bring-up documented in [GEMINUS_NATIVE_HANDOFF.md](GEMINUS_NATIVE_HANDOFF.md).
