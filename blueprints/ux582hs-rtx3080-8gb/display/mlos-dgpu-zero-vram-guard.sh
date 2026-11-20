#!/usr/bin/env bash
set -euo pipefail

# Verifies 0 MiB desktop/compositor VRAM on discrete NVIDIA GPU.
# Falls back to headless SSH console mode if any local compositor allocates dGPU scanout buffers.
FALLBACK_HEADLESS="1"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    exit 0
fi

USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
COMPUTE_MIB="$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')"
DESKTOP_MIB=$(( USED_MIB - COMPUTE_MIB ))

if (( DESKTOP_MIB > 0 )); then
    echo "ERR_RING0_DGPU_DESKTOP_VRAM: non-compute desktop VRAM is ${DESKTOP_MIB} MiB (expected 0 MiB)" >&2
    if [[ "${FALLBACK_HEADLESS}" == "1" ]] && command -v systemctl >/dev/null 2>&1; then
        echo "ACTION_FALLBACK_HEADLESS: isolating multi-user.target to preserve full 8 GiB CUDA pool" >&2
        systemctl isolate multi-user.target || true
    fi
    exit 1
fi

echo "dgpu_desktop_vram_mib=0"
