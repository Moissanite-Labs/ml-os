#!/usr/bin/env bash
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
