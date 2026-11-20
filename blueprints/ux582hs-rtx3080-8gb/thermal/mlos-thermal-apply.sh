#!/usr/bin/env bash
set -euo pipefail

# Applies ASUS thermal profile, ScreenPad Plus brightness, NVIDIA power cap, and AAS Plus lid-intake guard.
REQUESTED_PROFILE="${1:-sustained-train}"
DGPU_PRESENT="1"
OWNER_CLOCK_OK="0"
AAS_LID_REQ="1"
LID_GLOB="/proc/acpi/button/lid/*/state"
SCREENPAD_LED="/sys/class/leds/asus::screenpad/brightness"

FAN_PROFILE="Performance"
POWER_CAP_W="95"
LOCK_CLOCKS="0"
SM_CLOCK_MHZ="0"
SCREENPAD_BRIGHTNESS="64"

# UX582HS AAS Plus hinge guard: closing the lid blocks the upper keyboard intake above the vapor chamber.
if [[ "${AAS_LID_REQ}" == "1" && "${LID_GLOB}" != "none" ]]; then
    for lid_state_file in ${LID_GLOB}; do
        if [[ -f "${lid_state_file}" ]] && grep -qi "closed" "${lid_state_file}"; then
            echo "WARN_AAS_PLUS_LID_CLOSED_CLAMP: lid is closed; clamping ${REQUESTED_PROFILE} to quiet-edit (70 W)" >&2
            REQUESTED_PROFILE="quiet-edit"
            FAN_PROFILE="Quiet"
            POWER_CAP_W="70"
            LOCK_CLOCKS="0"
            SCREENPAD_BRIGHTNESS="0"
            break
        fi
    done
fi

if command -v asusctl >/dev/null 2>&1; then
    asusctl profile -P "${FAN_PROFILE}"
fi

if [[ "${SCREENPAD_LED}" != "none" && -w "${SCREENPAD_LED}" ]]; then
    printf '%s\n' "${SCREENPAD_BRIGHTNESS}" > "${SCREENPAD_LED}"
fi

if [[ "${DGPU_PRESENT}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -pm 1
    nvidia-smi -pl "${POWER_CAP_W}"
    if [[ "${LOCK_CLOCKS}" == "1" ]]; then
        if [[ "${OWNER_CLOCK_OK}" != "1" ]]; then
            echo "ERR_CLOCK_LOCK_UNAPPROVED: owner_clock_lock_approved is required for SM clock lock" >&2
            exit 1
        fi
        nvidia-smi -lgc "${SM_CLOCK_MHZ},${SM_CLOCK_MHZ}"
    else
        nvidia-smi -rgc || true
    fi
fi

echo "thermal_applied=${REQUESTED_PROFILE} power_cap_w=${POWER_CAP_W}"
