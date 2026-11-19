#!/usr/bin/env bash
set -euo pipefail

# Enforce root:root ownership and 0755 / go-w modes on trusted Ring 0 paths.
umask 022
ROOT="${MLOS_IMAGE_ROOT:-}"

normalize_ancestor() {
    local target="${ROOT%/}$1"
    if [[ ! -e "${target}" && ! -L "${target}" ]]; then
        return 0
    fi
    if [[ "$(id -u)" -eq 0 ]]; then
        chown root:root "${target}"
    fi
    chmod 0755 "${target}"
    chmod go-w "${target}"
}

normalize_tree() {
    local target="${ROOT%/}$1"
    if [[ ! -e "${target}" && ! -L "${target}" ]]; then
        return 0
    fi
    if [[ "$(id -u)" -eq 0 ]]; then
        find "${target}" -xdev \( -type d -o -type f \) -exec chown root:root {} +
    fi
    find "${target}" -xdev \( -type d -o -type f \) -exec chmod go-w {} +
}

for ancestor in \
    /etc \
    /etc/profile.d \
    /var \
    /var/lib \
    /var/lib/mlos \
    /usr \
    /usr/lib \
    /usr/lib/systemd \
    /usr/lib/systemd/system \
    /usr/local \
    /usr/local/bin; do
    normalize_ancestor "${ancestor}"
done

for tree in \
    /etc/mlos/canaries \
    /var/lib/mlos/models \
    /usr/lib/systemd/system; do
    normalize_tree "${tree}"
done
