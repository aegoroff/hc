#Requires -Version 5.1
<#
.SYNOPSIS
  Provisions Windows C deps for the Zig build: static OpenSSL libcrypto + headers.

.DESCRIPTION
  Mirrors scripts/build_external_libs.sh (the Linux provisioner).

  Workspace install (what build.zig links):
    external_lib\openssl\...

  Persistent agent cache (CI / HC_EXTERNAL_LIB_CACHE only; local stays in
  workspace external_lib\openssl with no C:\external_lib write-back):
    ${HC_EXTERNAL_LIB_CACHE:-C:\external_lib}\openssl-${ver}-${arch}-windows-msvc-asm\
  Rebuild only when the cached tree is missing or its version stamp differs.
  After a successful CI build the tree is written back into the cache.

  OpenSSL: Configure VC-WIN64A (/FS for parallel PDB; native Windows keeps
  platform asm — no no-asm), jom (or nmake) build, then nmake install_sw
  (serial: jom install races on recursive depend). Perl is required only on
  the download path.

.PARAMETER Arch
  Target arch (default x86_64).

.PARAMETER OpenSslVer
  OpenSSL source version to fetch (default 4.0.1, matching Linux).
#>
[CmdletBinding()]
param(
    [string]$Arch = "x86_64",
    [string]$OpenSslVer = "4.0.1"
)

# See windows_build.ps1: native commands write progress to stderr, which
# PowerShell 5.1 would turn into a terminating error under Stop. Rely on the
# explicit $LASTEXITCODE checks below instead.
$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LibInstallSrc = Join-Path $Root "external_lib\src"
$OpenSslPrefix = Join-Path $Root "external_lib\openssl"
$OpenSslLib = Join-Path $OpenSslPrefix "lib\libcrypto.lib"
$OpenSslVersionStamp = Join-Path $OpenSslPrefix ".hc-openssl-version"

$CacheRoot = $null
$UseAgentCache = $false
if ($env:HC_EXTERNAL_LIB_CACHE) {
    $UseAgentCache = $true
    $CacheRoot = $env:HC_EXTERNAL_LIB_CACHE
} elseif ($env:GITHUB_ACTIONS -eq "true" -or $env:CI -eq "true") {
    $UseAgentCache = $true
    $CacheRoot = "C:\external_lib"
}

# Windows builds are always native MSVC + asm for this provisioner.
$CacheKey = "openssl-$OpenSslVer-$Arch-windows-msvc-asm"
$CacheDir = if ($UseAgentCache) { Join-Path $CacheRoot $CacheKey } else { $null }
$CachedLib = if ($UseAgentCache) { Join-Path $CacheDir "lib\libcrypto.lib" } else { $null }
$CachedVersionStamp = if ($UseAgentCache) { Join-Path $CacheDir ".hc-openssl-version" } else { $null }
$CacheSrcDir = if ($UseAgentCache) { Join-Path $CacheRoot "src" } else { $null }
$CacheTarball = if ($UseAgentCache) { Join-Path $CacheSrcDir "openssl-$OpenSslVer.tar.gz" } else { $null }

function Test-OpenSslReady {
    param([string]$Prefix, [string]$Lib, [string]$Stamp, [string]$Ver)
    if (-not (Test-Path -LiteralPath $Lib)) { return $false }
    if (-not (Test-Path -LiteralPath $Stamp)) { return $false }
    $got = (Get-Content -LiteralPath $Stamp -Raw).Trim()
    return ($got -eq $Ver)
}

function Copy-OpenSslTree {
    param([string]$From, [string]$To)
    if (Test-Path -LiteralPath $To) {
        Remove-Item -LiteralPath $To -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $To) | Out-Null
    Copy-Item -Path $From -Destination $To -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $LibInstallSrc | Out-Null
if ($UseAgentCache) {
    New-Item -ItemType Directory -Force -Path $CacheSrcDir | Out-Null
}

# 1. Workspace already has the right version → done.
if (Test-OpenSslReady -Prefix $OpenSslPrefix -Lib $OpenSslLib -Stamp $OpenSslVersionStamp -Ver $OpenSslVer) {
    Write-Output "==> external_lib already provisioned (OpenSSL $OpenSslVer)"
    exit 0
}

# 2. Persistent cache hit (CI / HC_EXTERNAL_LIB_CACHE) → seed workspace.
if ($UseAgentCache -and (Test-OpenSslReady -Prefix $CacheDir -Lib $CachedLib -Stamp $CachedVersionStamp -Ver $OpenSslVer)) {
    Write-Output "==> seeding OpenSSL $OpenSslVer from cache $CacheDir -> $OpenSslPrefix"
    Copy-OpenSslTree -From $CacheDir -To $OpenSslPrefix
    if (Test-OpenSslReady -Prefix $OpenSslPrefix -Lib $OpenSslLib -Stamp $OpenSslVersionStamp -Ver $OpenSslVer) {
        Write-Output "==> external_lib OpenSSL ready ($OpenSslLib)"
        exit 0
    }
    Write-Output "warning: cache seed incomplete; rebuilding"
}

# Stale workspace tree (wrong/missing version) → clear before rebuild.
if (Test-Path -LiteralPath $OpenSslPrefix) {
    Remove-Item -LiteralPath $OpenSslPrefix -Recurse -Force
}

Write-Output "==> provisioning external_lib OpenSSL (libcrypto) for windows-msvc ($OpenSslVer) (arch=$Arch)"
if ($UseAgentCache) {
    Write-Output "    cache: $CacheDir"
} else {
    Write-Output "    cache: disabled (local workspace-only)"
}

$perl = Get-Command perl -ErrorAction SilentlyContinue
if (-not $perl) {
    throw "perl not found on PATH (required to Configure OpenSSL). Install Strawberry Perl, or seed via HC_EXTERNAL_LIB_CACHE / C:\external_lib (versioned openssl-* tree with lib\libcrypto.lib)."
}

# OpenSSL VC-WIN64A needs the x64 MSVC toolchain (cl/lib/nmake). A plain
# shell or an x86 Developer Prompt yields missing nmake or LNK1112.
function Import-VcVars64 {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) { return $false }
    $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsRoot) { return $false }
    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path -LiteralPath $vcvars)) { return $false }
    Write-Output "==> importing MSVC x64 env ($vcvars)"
    cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            Set-Item -LiteralPath "Env:$($Matches[1])" -Value $Matches[2]
        }
    }
    return $true
}

$needX64 = (-not (Get-Command nmake -ErrorAction SilentlyContinue)) -or
    ($env:VSCMD_ARG_TGT_ARCH -and $env:VSCMD_ARG_TGT_ARCH -ne "x64")
if ($needX64) {
    if (-not (Import-VcVars64)) {
        throw "nmake/x64 MSVC not available. Install VS C++ tools, or run from x64 Native Tools / Launch-VsDevShell.ps1 -Arch amd64."
    }
}

$nmake = Get-Command nmake -ErrorAction SilentlyContinue
if (-not $nmake) {
    throw "nmake not found on PATH after vcvars64 (run from a VS developer / msvc-dev-cmd environment)."
}
if ($env:VSCMD_ARG_TGT_ARCH -and $env:VSCMD_ARG_TGT_ARCH -ne "x64") {
    throw "VS target arch is still '$($env:VSCMD_ARG_TGT_ARCH)' after vcvars64 (need x64)."
}

$jom = Get-Command jom -ErrorAction SilentlyContinue
$makeCmd = if ($jom) { $jom.Source } else { $nmake.Source }
$makeName = if ($jom) { "jom" } else { "nmake" }

$OpenSslSrcName = "openssl-$OpenSslVer"
$OpenSslSrcDir = Join-Path $LibInstallSrc $OpenSslSrcName
$OpenSslTar = Join-Path $LibInstallSrc "$OpenSslSrcName.tar.gz"
$OpenSslUrl = "https://github.com/openssl/openssl/releases/download/$OpenSslSrcName/$OpenSslSrcName.tar.gz"

if ($UseAgentCache) {
    if (-not (Test-Path -LiteralPath $CacheTarball)) {
        Write-Output "==> downloading $OpenSslSrcName.tar.gz"
        Invoke-WebRequest -Uri $OpenSslUrl -OutFile $CacheTarball -UseBasicParsing
    }
    Copy-Item -LiteralPath $CacheTarball -Destination $OpenSslTar -Force
} elseif (-not (Test-Path -LiteralPath $OpenSslTar)) {
    Write-Output "==> downloading $OpenSslSrcName.tar.gz"
    Invoke-WebRequest -Uri $OpenSslUrl -OutFile $OpenSslTar -UseBasicParsing
}

if (Test-Path -LiteralPath $OpenSslSrcDir) {
    Remove-Item -LiteralPath $OpenSslSrcDir -Recurse -Force
}
Write-Output "==> extracting $OpenSslTar"
& tar -xzf $OpenSslTar -C $LibInstallSrc
if ($LASTEXITCODE -ne 0) { throw "OpenSSL extract failed (exit $LASTEXITCODE)" }
if (-not (Test-Path -LiteralPath $OpenSslSrcDir)) {
    throw "OpenSSL extract did not produce $OpenSslSrcDir"
}

Push-Location $OpenSslSrcDir
try {
    # /FS: parallel cl (jom) may share one /Fd PDB; without it MSVC emits C1041.
    # Native Windows build — keep AES-NI / SHA asm (do not pass no-asm).
    & perl Configure VC-WIN64A -static no-apps /FS --prefix=$OpenSslPrefix `
        "--openssldir=$OpenSslPrefix\ssl"
    if ($LASTEXITCODE -ne 0) { throw "OpenSSL Configure failed (exit $LASTEXITCODE)" }
    & perl -I. -Mconfigdata -e "die qq(OpenSSL asm disabled on native Windows build; digests need platform asm`n) if `$disabled{asm}; die qq(OpenSSL Configure left asm_arch empty`n) unless `$target{asm_arch}; print qq(OpenSSL asm enabled (asm_arch=`$target{asm_arch})`n)"
    if ($LASTEXITCODE -ne 0) { throw "OpenSSL asm check failed (exit $LASTEXITCODE)" }
    if ($jom) {
        Write-Output "==> building OpenSSL with jom -j$env:NUMBER_OF_PROCESSORS"
        & $makeCmd "-j$env:NUMBER_OF_PROCESSORS"
    } else {
        Write-Output "==> building OpenSSL with nmake"
        & $makeCmd
    }
    if ($LASTEXITCODE -ne 0) { throw "OpenSSL $makeName failed (exit $LASTEXITCODE)" }
    # install_sw must be serial: parallel jom re-enters `depend` and fails
    # with Error 13 / build_inst_programs Error 2 on this runner.
    Write-Output "==> installing OpenSSL with nmake install_sw"
    & $nmake.Source install_sw
    if ($LASTEXITCODE -ne 0) { throw "OpenSSL nmake install_sw failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $OpenSslLib)) {
    throw "OpenSSL provisioning did not produce $OpenSslLib"
}
Set-Content -LiteralPath $OpenSslVersionStamp -Value $OpenSslVer

# Write-back only on CI / when HC_EXTERNAL_LIB_CACHE is set.
if ($UseAgentCache) {
    Write-Output "==> caching OpenSSL $OpenSslVer -> $CacheDir"
    Copy-OpenSslTree -From $OpenSslPrefix -To $CacheDir
    Set-Content -LiteralPath $CachedVersionStamp -Value $OpenSslVer
}

Write-Output "==> external_lib OpenSSL ready ($OpenSslLib)"
# Explicit exit so the caller sees 0: $LASTEXITCODE tracks native exes and can
# otherwise retain a stale non-zero from earlier CI/shell steps.
exit 0
