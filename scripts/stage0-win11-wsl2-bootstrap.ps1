<#
.SYNOPSIS
Stage 0 non-destructive Windows 11 + WSL2 CUDA sm_86 and Lean v4.35.0 workbench bootstrap for geminus (UX582HS).

.DESCRIPTION
Runs read-only Phase 0 hardware, Secure Boot, BitLocker, NVMe partition, and NVIDIA WDDM VRAM checks by default.
Never modifies disk partitions, bootloaders, or Windows system partitions.
When -WriteWslConfig or -BootstrapWslToolchain is explicitly passed, configures user-space .wslconfig (with backup)
and builds/verifies CUDA sm_86 llama-server + pinned Lean v4.35.0 inside WSL2.
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly = $true,
    [switch]$WriteWslConfig = $false,
    [switch]$BootstrapWslToolchain = $false,
    [string]$WslDistro = "",
    [string]$ReceiptPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CompiledWslConfig = Join-Path $RepoRoot "blueprints/ux582hs-rtx3080-8gb/wsl2/.wslconfig.mlos"

function Get-NvidiaStage0Telemetry {
    $smiExe = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($null -eq $smiExe) {
        $fallback = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
        if (Test-Path -LiteralPath $fallback) {
            $smiExe = $fallback
        } else {
            return $null
        }
    } else {
        $smiExe = $smiExe.Source
    }

    $gpuRaw = & $smiExe --query-gpu=name,driver_version,compute_cap,memory.total,memory.used,memory.free,power.limit,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gpuRaw)) {
        return $null
    }
    $firstLine = ($gpuRaw -split "`r?`n")[0]
    $cols = $firstLine -split "," | ForEach-Object { $_.Trim() }
    if ($cols.Count -lt 9) {
        return $null
    }

    $computeMib = 0
    $appsRaw = & $smiExe --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($appsRaw)) {
        foreach ($line in ($appsRaw -split "`r?`n")) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $acols = $line -split "," | ForEach-Object { $_.Trim() }
            if ($acols.Count -ge 3) {
                $parsed = 0
                if ([int]::TryParse($acols[2], [ref]$parsed)) {
                    $computeMib += [Math]::Max(0, $parsed)
                }
            }
        }
    }

    $totalMib = [int]$cols[3]
    $usedMib = [int]$cols[4]
    $freeMib = [int]$cols[5]
    $wddmDesktopMib = [Math]::Max(0, $usedMib - $computeMib)

    return [ordered]@{
        gpu_name = $cols[0]
        driver_version = $cols[1]
        compute_cap = $cols[2]
        vram_total_mib = $totalMib
        vram_used_mib = $usedMib
        vram_free_mib = $freeMib
        compute_vram_mib = $computeMib
        wddm_desktop_vram_mib = $wddmDesktopMib
        t_c5_budget_mib = 7372
        wddm_desktop_warning = ($wddmDesktopMib -gt 512)
        power_limit_w = $cols[6]
        temperature_gpu_c = [int]$cols[7]
        clocks_event_reasons_active = $cols[8]
    }
}

$cs = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, BuildNumber, OSArchitecture
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors
$disks = @(Get-CimInstance Win32_DiskDrive | Select-Object Model, Size, InterfaceType)

$secureBoot = $null
try {
    $secureBoot = [bool](Confirm-SecureBootUEFI)
} catch {
    $secureBoot = $null
}

$bitLockerVolumes = @()
try {
    $bitLockerVolumes = @(Get-BitLockerVolume -ErrorAction Stop | ForEach-Object {
        [ordered]@{
            mount_point = $_.MountPoint
            protection_status = [string]$_.ProtectionStatus
            volume_status = [string]$_.VolumeStatus
        }
    })
} catch {
    $bitLockerVolumes = @()
}

$partitions = @()
try {
    $partitions = @(Get-Partition -ErrorAction Stop | ForEach-Object {
        [ordered]@{
            disk_number = $_.DiskNumber
            partition_number = $_.PartitionNumber
            drive_letter = [string]$_.DriveLetter
            size_gib = [Math]::Round($_.Size / 1GB, 2)
            type = [string]$_.Type
        }
    })
} catch {
    $partitions = @()
}

$nvidia = Get-NvidiaStage0Telemetry

$wslAvailable = ($null -ne (Get-Command "wsl.exe" -ErrorAction SilentlyContinue))

$receipt = [ordered]@{
    schema = "mlos-stage0-win11-preflight/v0.1"
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    zero_partition_writes = $true
    system = [ordered]@{
        manufacturer = $cs.Manufacturer
        model = $cs.Model
        os_caption = $os.Caption
        os_build = $os.BuildNumber
        architecture = $os.OSArchitecture
        cpu_name = $cpu.Name
        cpu_cores = [int]$cpu.NumberOfCores
        cpu_threads = [int]$cpu.NumberOfLogicalProcessors
        ram_mib = [int]([Math]::Floor($cs.TotalPhysicalMemory / 1MB))
        secure_boot_enabled = $secureBoot
    }
    storage = [ordered]@{
        disks = $disks
        partitions = $partitions
        bitlocker = $bitLockerVolumes
    }
    wsl2_available = $wslAvailable
    nvidia_stage0 = $nvidia
    actions_executed = @()
}

if ($WriteWslConfig) {
    if (-not (Test-Path -LiteralPath $CompiledWslConfig)) {
        throw "ERR_STAGE0_WSLCONFIG_MISSING: Expected compiled blueprint at $CompiledWslConfig"
    }
    $userWslConfig = Join-Path $env:USERPROFILE ".wslconfig"
    if (Test-Path -LiteralPath $userWslConfig) {
        $backupPath = Join-Path $env:USERPROFILE ".wslconfig.mlos.bak"
        Copy-Item -LiteralPath $userWslConfig -Destination $backupPath -Force
    }
    Copy-Item -LiteralPath $CompiledWslConfig -Destination $userWslConfig -Force
    $receipt.actions_executed += "wrote_user_wslconfig"
}

if ($BootstrapWslToolchain) {
    if (-not $wslAvailable) {
        throw "ERR_STAGE0_WSL_MISSING: wsl.exe is not installed or enabled on this host"
    }
    $wslArgs = @()
    if (-not [string]::IsNullOrWhiteSpace($WslDistro)) {
        $wslArgs += @("-d", $WslDistro)
    }

    $bootstrapBash = @'
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1

if [[ ! -e /dev/dxg ]]; then
    echo "ERR_STAGE0_DXG_MISSING: /dev/dxg WSL2 GPU paravirtualization node not present" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv

if [[ ! -x "$HOME/.elan/bin/elan" ]]; then
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.35.0
else
    "$HOME/.elan/bin/elan" toolchain install leanprover/lean4:v4.35.0
    "$HOME/.elan/bin/elan" default leanprover/lean4:v4.35.0
fi
"$HOME/.elan/bin/lean" --version

if [[ -d "$HOME/src/llama.cpp" ]]; then
    cmake -S "$HOME/src/llama.cpp" -B "$HOME/src/llama.cpp/build-sm86" \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_CURL=OFF
    cmake --build "$HOME/src/llama.cpp/build-sm86" --target llama-server -j"$(nproc)"
fi
'@
    & wsl.exe @wslArgs -- bash -lc $bootstrapBash
    if ($LASTEXITCODE -ne 0) {
        throw "ERR_STAGE0_WSL_BOOTSTRAP_FAILED: WSL2 toolchain bootstrap returned exit code $LASTEXITCODE"
    }
    $receipt.actions_executed += "bootstrapped_wsl2_cuda_sm86_and_lean_4_35_0"
}

$jsonOut = $receipt | ConvertTo-Json -Depth 6
if (-not [string]::IsNullOrWhiteSpace($ReceiptPath)) {
    $parentDir = Split-Path -Parent $ReceiptPath
    if (-not [string]::IsNullOrWhiteSpace($parentDir) -and -not (Test-Path -LiteralPath $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }
    [System.IO.File]::WriteAllText($ReceiptPath, ($jsonOut + "`n"), (New-Object System.Text.UTF8Encoding($false)))
} else {
    Write-Output $jsonOut
}
