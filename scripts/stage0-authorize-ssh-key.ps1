<#
.SYNOPSIS
Idempotent, fail-closed Windows OpenSSH ed25519/ecdsa public key installer with strict ACL enforcement.

.DESCRIPTION
Validates SSH public key format (rejecting options injection, multi-line payloads, or private keys),
installs idempotently into C:\ProgramData\ssh\administrators_authorized_keys (when -AdminGroupKey is set)
or $env:USERPROFILE\.ssh\authorized_keys, and enforces exact Windows OpenSSH 9.x mandatory ACLs
(NT AUTHORITY\SYSTEM and BUILTIN\Administrators full control, inheritance disabled).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PublicKey,
    [switch]$AdminGroupKey = $true,
    [switch]$DryRun = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$trimmedKey = $PublicKey.Trim()
if ($trimmedKey.Contains("`n") -or $trimmedKey.Contains("`r")) {
    throw "ERR_SSH_KEY_MULTILINE: PublicKey must be a single line"
}
if ($trimmedKey -match "PRIVATE KEY") {
    throw "ERR_SSH_PRIVATE_KEY_REJECTED: Refusing private key material"
}
if ($trimmedKey -notmatch "^(ssh-ed25519|ecdsa-sha2-nistp256)\s+[A-Za-z0-9+/=]+(\s+[\w.@+-]+)?$") {
    throw "ERR_SSH_KEY_FORMAT: Only standard ssh-ed25519 or ecdsa-sha2-nistp256 keys without command options are accepted"
}

if ($AdminGroupKey) {
    $sshDir = Join-Path $env:ProgramData "ssh"
    $targetFile = Join-Path $sshDir "administrators_authorized_keys"
} else {
    $sshDir = Join-Path $env:USERPROFILE ".ssh"
    $targetFile = Join-Path $sshDir "authorized_keys"
}

if ($DryRun) {
    [ordered]@{
        dry_run = $true
        target_file = $targetFile
        admin_group_key = [bool]$AdminGroupKey
        key_type = ($trimmedKey -split "\s+")[0]
    } | ConvertTo-Json -Depth 3
    exit 0
}

if (-not (Test-Path -LiteralPath $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
}

$existingLines = @()
if (Test-Path -LiteralPath $targetFile) {
    Copy-Item -LiteralPath $targetFile -Destination ($targetFile + ".mlos.bak") -Force
    $existingLines = @(Get-Content -LiteralPath $targetFile | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
}

$keyBase64 = ($trimmedKey -split "\s+")[1]
$alreadyPresent = $false
foreach ($line in $existingLines) {
    $parts = $line -split "\s+"
    if ($parts.Count -ge 2 -and $parts[1] -eq $keyBase64) {
        $alreadyPresent = $true
        break
    }
}

if (-not $alreadyPresent) {
    $existingLines += $trimmedKey
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($targetFile, $existingLines, $utf8NoBom)

if ($AdminGroupKey) {
    & icacls.exe $targetFile /inheritance:r /grant "*S-1-5-18:(F)" /grant "*S-1-5-32-544:(F)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "ERR_SSH_ACL_FAILED: icacls failed on $targetFile"
    }
}

[ordered]@{
    dry_run = $false
    target_file = $targetFile
    already_present = $alreadyPresent
    total_authorized_keys = $existingLines.Count
    acl_hardened_sids = if ($AdminGroupKey) { @("S-1-5-18", "S-1-5-32-544") } else { @() }
} | ConvertTo-Json -Depth 3
