"""Cross-platform atomic GPU mutual-exclusion lease and 0 MiB release VRAM verifier."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterator

if os.name == "nt":
    import msvcrt
    fcntl = None  # type: ignore[assignment]
else:
    import fcntl
    msvcrt = None  # type: ignore[assignment]

ALLOWED_LEASE_MODES: frozenset[str] = frozenset({"infer", "train", "eval", "benchmark"})

ComputeVramQuery = Callable[[], tuple[int, list[dict[str, Any]]]]


class GpuLeaseError(RuntimeError):
    """Base error for GPU lease acquisition or release failures."""


class GpuLeaseBusyError(GpuLeaseError):
    """Raised fail-closed when another workload already holds the exclusive GPU lease."""


class GpuVramNotCleanError(GpuLeaseError):
    """Raised when pre-existing compute processes hold > 0 MiB VRAM before lease acquisition."""


class GpuVramLeakError(GpuLeaseError):
    """Raised on lease release when lingering compute VRAM exceeds 0 MiB."""


def default_compute_vram_query() -> tuple[int, list[dict[str, Any]]]:
    """Query active CUDA compute processes via nvidia-smi; returns (0, []) if no NVIDIA GPU."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=6,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return (0, [])

    if proc.returncode != 0 or not proc.stdout.strip():
        return (0, [])

    apps: list[dict[str, Any]] = []
    total_mib = 0
    for line in proc.stdout.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 3:
            continue
        try:
            pid_val = int(cols[0])
        except ValueError:
            continue
        try:
            mib_val = max(0, int(cols[2]))
        except ValueError:
            mib_val = 0
        total_mib += mib_val
        apps.append(
            {
                "pid": pid_val,
                "process_name": cols[1] or "unknown",
                "used_gpu_memory_mib": mib_val,
            }
        )
    return (total_mib, apps)


def _try_lock_nonblocking(fd: int) -> None:
    if os.name == "nt":
        if msvcrt is None:
            raise GpuLeaseError("ERR_GPU_LOCK_PLATFORM: msvcrt unavailable on Windows")
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise GpuLeaseBusyError("ERR_GPU_LEASE_BUSY: GPU lease is held by another process") from exc
    else:
        if fcntl is None:
            raise GpuLeaseError("ERR_GPU_LOCK_PLATFORM: fcntl unavailable on POSIX")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise GpuLeaseBusyError("ERR_GPU_LEASE_BUSY: GPU lease is held by another process") from exc


def _unlock_fd(fd: int) -> bool:
    if os.name == "nt":
        if msvcrt is not None:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                return True
            except OSError:
                return False
        return False
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        except OSError:
            return False
    return False


@dataclass
class GpuLeaseReceipt:
    """Audit record of an exclusive GPU lease execution."""

    mode: str
    holder_pid: int
    lock_path: str
    acquired_utc: str
    released_utc: str | None
    pre_compute_vram_mib: int
    post_compute_vram_mib: int | None
    lingering_compute_processes: list[dict[str, Any]]
    command_exit_code: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mlos-gpu-lease-receipt/v0.1",
            "mode": self.mode,
            "holder_pid": self.holder_pid,
            "lock_path": self.lock_path,
            "acquired_utc": self.acquired_utc,
            "released_utc": self.released_utc,
            "pre_compute_vram_mib": self.pre_compute_vram_mib,
            "post_compute_vram_mib": self.post_compute_vram_mib,
            "lingering_compute_processes": self.lingering_compute_processes,
            "command_exit_code": self.command_exit_code,
        }


def read_lease_status(lock_path: Path) -> dict[str, Any]:
    """Non-destructively probe whether the GPU lease is currently held or free."""
    meta_path = lock_path.with_suffix(lock_path.suffix + ".meta.json")
    if not lock_path.exists():
        return {"held": False, "lock_path": str(lock_path), "holder": None}

    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\n")
        try:
            _try_lock_nonblocking(fd)
        except GpuLeaseBusyError:
            holder: dict[str, Any] | None = None
            if meta_path.is_file():
                try:
                    holder = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    holder = None
            return {"held": True, "lock_path": str(lock_path), "holder": holder}
        _unlock_fd(fd)
        return {"held": False, "lock_path": str(lock_path), "holder": None}
    finally:
        os.close(fd)


@contextmanager
def acquire_gpu_lease(
    *,
    lock_path: Path,
    mode: str,
    vram_query: ComputeVramQuery = default_compute_vram_query,
    require_clean_entry: bool = True,
    assert_zero_lingering_vram: bool = True,
) -> Iterator[GpuLeaseReceipt]:
    """Acquire atomic exclusive GPU lock fail-closed and assert 0 MiB lingering VRAM on exit."""
    if mode not in ALLOWED_LEASE_MODES:
        raise ValueError(f"ERR_GPU_LEASE_MODE: mode={mode!r} must be one of {sorted(ALLOWED_LEASE_MODES)}")

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = lock_path.with_suffix(lock_path.suffix + ".meta.json")

    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"L")
        _try_lock_nonblocking(fd)
        locked = True

        pre_mib, pre_procs = vram_query()
        if require_clean_entry and pre_mib > 0:
            raise GpuVramNotCleanError(
                f"ERR_GPU_VRAM_DIRTY_ENTRY: {pre_mib} MiB compute VRAM already allocated before {mode} lease: {pre_procs}"
            )

        acquired_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta = {
            "mode": mode,
            "holder_pid": os.getpid(),
            "acquired_utc": acquired_utc,
        }
        meta_path.write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")

        receipt = GpuLeaseReceipt(
            mode=mode,
            holder_pid=os.getpid(),
            lock_path=str(lock_path),
            acquired_utc=acquired_utc,
            released_utc=None,
            pre_compute_vram_mib=pre_mib,
            post_compute_vram_mib=None,
            lingering_compute_processes=[],
            command_exit_code=None,
        )

        body_exc: BaseException | None = None
        try:
            yield receipt
        except BaseException as exc:
            body_exc = exc
            raise
        finally:
            post_mib, post_procs = vram_query()
            receipt.post_compute_vram_mib = post_mib
            receipt.lingering_compute_processes = post_procs
            receipt.released_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            meta_path.unlink(missing_ok=True)
            if assert_zero_lingering_vram and post_mib > 0 and body_exc is None:
                raise GpuVramLeakError(
                    f"ERR_GPU_VRAM_LEAK: {post_mib} MiB lingering compute VRAM after {mode} lease release: {post_procs}"
                )
    finally:
        if locked:
            _unlock_fd(fd)
        os.close(fd)


def run_under_gpu_lease(
    *,
    command: list[str],
    lock_path: Path,
    mode: str,
    env: dict[str, str] | None = None,
    vram_query: ComputeVramQuery = default_compute_vram_query,
) -> GpuLeaseReceipt:
    """Execute a subprocess while holding the exclusive GPU lease and verify 0 MiB release."""
    if not command:
        raise ValueError("ERR_GPU_LEASE_CMD: command list must not be empty")

    merged_env = dict(os.environ if env is None else env)
    merged_env["MLOS_ACTIVE_GPU_LEASE_MODE"] = mode

    with acquire_gpu_lease(
        lock_path=lock_path,
        mode=mode,
        vram_query=vram_query,
        require_clean_entry=True,
        assert_zero_lingering_vram=True,
    ) as receipt:
        proc = subprocess.run(command, env=merged_env, check=False)
        receipt.command_exit_code = proc.returncode
    return receipt
