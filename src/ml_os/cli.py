"""Command-line interface for Moissanite-Labs/ml-os (adapt, probe, compile, gpu-lease, stamp-run)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from ml_os.adapt import (
    AdaptationRejectedError,
    ProfileValidationError,
    apply_adaptation,
    check_adaptation,
    validate_profile,
)
from ml_os.compiler import compile_profile, verify_compiled_directory
from ml_os.gpu_lease import (
    ALLOWED_LEASE_MODES,
    GpuLeaseError,
    read_lease_status,
    run_under_gpu_lease,
)
from ml_os.probe import probe_host
from ml_os.stamp_run import StampAnomalyError, stamp_run


def _default_lock_path() -> Path:
    env_lock = os.environ.get("MLOS_GPU_LEASE_LOCK")
    if env_lock:
        return Path(env_lock)
    if os.name == "nt":
        base = Path(os.environ.get("TEMP", "."))
        return base / "mlos-gpu.lease.lock"
    run_dir = Path("/run/mlos")
    if run_dir.is_dir() and os.access(run_dir, os.W_OK):
        return run_dir / "gpu.lease.lock"
    return Path("/tmp/mlos-gpu.lease.lock")


def _strip_separator(cmd: list[str]) -> list[str]:
    if cmd and cmd[0] == "--":
        return cmd[1:]
    return cmd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlos",
        description="Moissanite-Labs ml-os 3-ring safety validator, hardware probe, compiler, GPU lease, and stamper",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # adapt
    p_adapt = sub.add_parser("adapt", help="Validate Ring 0 profile invariants or check/apply a Ring 2 diff")
    p_adapt.add_argument("--profile", type=Path, required=True, help="Base or standalone mlos-profile/v0.1 TOML path")
    p_adapt.add_argument("--candidate", type=Path, default=None, help="Optional candidate Ring 2 profile TOML path")
    p_adapt.add_argument("--check", action="store_true", help="Validate without modifying disk")
    p_adapt.add_argument("--apply", action="store_true", help="Atomically apply validated Ring 2 candidate and append ledger")
    p_adapt.add_argument("--ledger", type=Path, default=None, help="Ledger JSONL path required with --apply")
    p_adapt.add_argument("--json", action="store_true", help="Emit JSON report")

    # probe
    p_probe = sub.add_parser("probe", help="Run read-only cross-OS hardware probe (zero disk writes)")
    p_probe.add_argument("--json", action="store_true", help="Emit structured JSON report")
    p_probe.add_argument("--os-override", type=str, default=None, help="Optional OS family override for testing")

    # compile
    p_compile = sub.add_parser("compile", help="Compile mlos-profile/v0.1 into deterministic OS blueprints")
    p_compile.add_argument("--profile", type=Path, required=True, help="Input mlos-profile/v0.1 TOML path")
    p_compile.add_argument("--out-dir", type=Path, default=None, help="Output directory to write compiled blueprints")
    p_compile.add_argument(
        "--verify-dir",
        type=Path,
        default=None,
        help="Verify existing blueprint directory matches compiled profile with 0-byte drift",
    )

    # gpu-lease
    p_lease = sub.add_parser("gpu-lease", help="Acquire exclusive atomic GPU lease or inspect lease status")
    p_lease.add_argument(
        "--mode",
        choices=sorted(ALLOWED_LEASE_MODES),
        default="train",
        help="Workload class acquiring the GPU lease",
    )
    p_lease.add_argument("--lock-file", type=Path, default=None, help="Override GPU lease lockfile path")
    p_lease.add_argument("--status", action="store_true", help="Inspect current GPU lease state without executing a job")
    p_lease.add_argument("command", nargs=argparse.REMAINDER, help="Command to run while holding the GPU lease (after --)")

    # stamp-run
    p_stamp = sub.add_parser("stamp-run", help="Run command under Zone A/B determinism pins and emit provenance receipt")
    p_stamp.add_argument("--profile", type=Path, required=True, help="Active mlos-profile/v0.1 TOML path")
    p_stamp.add_argument("--zone", choices=["A", "B"], default="B", help="Determinism zone ('A' CPU or 'B' CUDA/offline)")
    p_stamp.add_argument("--output", type=Path, default=None, help="Path to write JSON provenance receipt")
    p_stamp.add_argument("--require-gpu", action="store_true", help="Fail closed if nvidia-smi telemetry is unavailable")
    p_stamp.add_argument(
        "--strict-anomalies",
        action="store_true",
        help="Fail closed if desktop VRAM > 0 MiB, thermal throttling, or lingering compute VRAM is detected",
    )
    p_stamp.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute under provenance stamper (after --)")

    return parser


def _cmd_adapt(args: argparse.Namespace) -> int:
    if args.apply:
        if args.candidate is None or args.ledger is None:
            print("mlos adapt --apply requires both --candidate and --ledger", file=sys.stderr)
            return 2
        try:
            receipt = apply_adaptation(
                active_profile_path=args.profile,
                candidate_profile_path=args.candidate,
                ledger_path=args.ledger,
            )
        except AdaptationRejectedError as exc:
            if args.json:
                print(json.dumps({"passed": False, "violations": exc.violations}, indent=2, sort_keys=True))
            else:
                for v in exc.violations:
                    print(v, file=sys.stderr)
            return 1
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    if args.candidate is not None:
        report = check_adaptation(args.profile, args.candidate)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        elif report.passed:
            print(
                f"ADAPT_CHECK_OK base={report.base_sha256[:12]} candidate={report.candidate_sha256[:12]} "
                f"ring2_changes={len(report.changed_ring2_paths)}"
            )
        else:
            for v in report.violations:
                print(v, file=sys.stderr)
        return 0 if report.passed else 1

    val = validate_profile(args.profile)
    if args.json:
        print(json.dumps(val.to_dict(), indent=2, sort_keys=True))
    elif val.passed:
        print(f"PROFILE_VALID id={val.profile_id} sha256={val.profile_sha256}")
    else:
        for v in val.violations:
            print(v, file=sys.stderr)
    return 0 if val.passed else 1


def _cmd_probe(args: argparse.Namespace) -> int:
    data = probe_host(system_override=args.os_override)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(
            f"host_os={data.get('host_os')} arch={data.get('architecture')} "
            f"model={data.get('model')} ram_gib={data.get('ram_gib')} dgpu={data.get('dgpu_model')}"
        )
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    if args.out_dir is None and args.verify_dir is None:
        print("mlos compile requires --out-dir and/or --verify-dir", file=sys.stderr)
        return 2

    try:
        if args.out_dir is not None:
            manifest = compile_profile(args.profile, args.out_dir)
            print(
                f"COMPILE_OK profile={manifest['profile_id']} artifacts={manifest['artifact_count']} "
                f"tree_sha256={manifest['artifacts_tree_sha256']}"
            )
        if args.verify_dir is not None:
            ok, mismatches = verify_compiled_directory(args.profile, args.verify_dir)
            if not ok:
                for m in mismatches:
                    print(m, file=sys.stderr)
                return 1
            print(f"VERIFY_ZERO_DRIFT_OK dir={args.verify_dir}")
    except ProfileValidationError as exc:
        for v in exc.violations:
            print(v, file=sys.stderr)
        return 1
    return 0


def _cmd_gpu_lease(args: argparse.Namespace) -> int:
    lock_file = args.lock_file if args.lock_file is not None else _default_lock_path()
    if args.status:
        status = read_lease_status(lock_file)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    cmd = _strip_separator(list(args.command))
    if not cmd:
        print("mlos gpu-lease requires --status or a command after --", file=sys.stderr)
        return 2

    try:
        receipt = run_under_gpu_lease(command=cmd, lock_path=lock_file, mode=args.mode)
    except GpuLeaseError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return int(receipt.command_exit_code or 0)


def _cmd_stamp_run(args: argparse.Namespace) -> int:
    cmd = _strip_separator(list(args.command))
    if not cmd:
        print("mlos stamp-run requires a command after --", file=sys.stderr)
        return 2
    try:
        receipt = stamp_run(
            command=cmd,
            profile_path=args.profile,
            zone=args.zone,
            output_path=args.output,
            require_gpu=args.require_gpu,
            strict_anomalies=args.strict_anomalies,
        )
    except (ProfileValidationError, StampAnomalyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output is None:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return int(receipt["execution"]["exit_code"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.subcommand == "adapt":
        return _cmd_adapt(args)
    if args.subcommand == "probe":
        return _cmd_probe(args)
    if args.subcommand == "compile":
        return _cmd_compile(args)
    if args.subcommand == "gpu-lease":
        return _cmd_gpu_lease(args)
    if args.subcommand == "stamp-run":
        return _cmd_stamp_run(args)
    return 2


def main_adapt() -> None:
    raise SystemExit(main(["adapt", *sys.argv[1:]]))


def main_probe() -> None:
    raise SystemExit(main(["probe", *sys.argv[1:]]))


def main_compile() -> None:
    raise SystemExit(main(["compile", *sys.argv[1:]]))


def main_gpu_lease() -> None:
    raise SystemExit(main(["gpu-lease", *sys.argv[1:]]))


def main_stamp_run() -> None:
    raise SystemExit(main(["stamp-run", *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
