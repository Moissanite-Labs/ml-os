# ml-os

`ml-os` is an adaptive operating system workbench for machine learning and formal verification researchers. It fits hardware-specific Linux configurations, GPU leases, memory slices, and provenance receipts around a locked safety core. Built at Moissanite Labs for dual-GPU laptops and dedicated compute hosts, starting with the ASUS ZenBook Pro Duo UX582HS (`8 GiB` RTX 3080 Laptop GPU, `32 GiB` RAM, Intel i9-11900H).

## How the Three Rings Work

`ml-os` separates security rules that never bend from hardware settings and researcher preferences:

1. **Ring 0 (Locked Safety Core)**: Blocks `sudo` and privilege-escalation binaries, requires `bubblewrap` and Landlock tool sandboxes, locks discrete GPU desktop VRAM at `0 MiB`, and enforces atomic GPU mutual exclusion. Mounts `@models` read-only and pairs Btrfs `@` snapshots with Unified Kernel Image (UKI) single-reboot recovery while keeping bare-metal repartitioning locked off.
2. **Ring 1 (Cross-OS Hardware Probe and Profile Compiler)**: `mlos probe` inspects Linux, Windows 11, and macOS hosts without writing to disk or inventing missing fields. `mlos compile` turns a validated `mlos-profile/v0.1` TOML file into pinned Arch Linux `mkosi` manifests, systemd cgroup v2 slices, Intel iGPU dual-screen configs, ASUS thermal guards, and WSL2 bootstrap files with zero byte drift.
3. **Ring 2 (Safe User Adaptation)**: Researchers switch display modes (`headless-ssh`, `dual-screen-igpu`, `single-primary-igpu`), thermal profiles (`quiet-edit`, `sustained-train`, `benchmark-locked`), remote tunnels, and domain packs by proposing declarative TOML edits. `mlos adapt` rejects any edit that touches Ring 0 or weakens hardware caps before a single byte hits disk.

## Repository Layout

- `src/ml_os/adapt.py`: `mlos-profile/v0.1` parser, Ring 0 invariant validator, Ring 2 diff gate, and hash-chained adaptation ledger.
- `src/ml_os/probe.py`: Read-only Linux, Windows 11, and macOS hardware probe plus pre-install inventory parser.
- `src/ml_os/compiler.py`: Deterministic blueprint compiler and zero-drift directory verifier.
- `src/ml_os/gpu_lease.py`: Cross-platform (`fcntl` / `msvcrt`) atomic GPU lock blocking concurrent training and serving jobs fail-closed, checking `0 MiB` lingering compute VRAM on exit.
- `src/ml_os/stamp_run.py`: Zone A (CPU single-thread deterministic) and Zone B (CUDA deterministic + offline) run wrapper stamping receipts with profile SHA-256, CPU ISA, and start/end `nvidia-smi` telemetry, including NVML clock-throttling bitmask decoding and WDDM desktop VRAM alerts.
- `profiles/`: Declarative hardware profiles (`ux582hs-rtx3080-8gb.toml` and `vmtest-qemu-x86_64.toml`).
- `blueprints/ux582hs-rtx3080-8gb/`: Deterministically compiled `mkosi`, systemd, Wayland dual-screen, Btrfs, thermal, sandbox, and WSL2 artifacts.
- `scripts/`: Stage 0 non-destructive Windows 11 and WSL2 CUDA `sm_86` + Lean `v4.35.0` bootstrap (`stage0-win11-wsl2-bootstrap.ps1`) and Windows OpenSSH key ACL helper (`stage0-authorize-ssh-key.ps1`).
- `docs/`: Technical specification ([ARCHITECTURE.md](docs/ARCHITECTURE.md)) and native hardware takeover plan ([GEMINUS_NATIVE_HANDOFF.md](docs/GEMINUS_NATIVE_HANDOFF.md)).

## Quickstart

`ml-os` uses only the Python 3.11+ standard library at runtime. Run commands directly from a fresh checkout:

```bash
# Validate Ring 0 and Ring 2 invariants on the UX582HS profile
python3 ./bin/mlos adapt --check --profile profiles/ux582hs-rtx3080-8gb.toml

# Verify 0-byte drift between the UX582HS profile and checked-in blueprints
python3 ./bin/mlos compile --profile profiles/ux582hs-rtx3080-8gb.toml \
    --verify-dir blueprints/ux582hs-rtx3080-8gb

# Run a read-only hardware probe on Linux, Windows 11, or macOS
python3 ./bin/mlos probe --json

# Run a command under an exclusive GPU lease and Zone B provenance stamper
python3 ./bin/mlos gpu-lease --mode train -- \
    python3 ./bin/mlos stamp-run --profile profiles/ux582hs-rtx3080-8gb.toml --zone B \
    --output run-receipt.json -- python3 -c "print('experiment ok')"
```

On Windows 11 hosts before dual-boot Linux installation, inspect hardware, BitLocker, and WDDM VRAM overhead without touching partitions:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stage0-win11-wsl2-bootstrap.ps1 -CheckOnly
```

## Running the Test Suite

```bash
python3 -O -B -W error -m pytest -v
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
