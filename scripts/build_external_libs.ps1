#Requires -Version 5.1
<#
.SYNOPSIS
  Provisions Windows C deps for the Zig build: OpenSSL headers + a static
  Apache APR archive (apr-1.lib) that lld-link can consume.

.DESCRIPTION
  Mirrors scripts/build_external_libs.sh (the Linux provisioner) and the legacy
  CI step that copied c:\external_lib into the workspace.

  OpenSSL: build.zig only needs headers under external_lib\openssl\include
  (whirlpool is compiled from vendored sources, not linked against
  libcrypto.lib). Headers are not in git (external_lib/ is gitignored); on the
  self-hosted runner they live in the persistent cache at C:\external_lib
  (override with HC_EXTERNAL_LIB_CACHE). This script seeds the workspace from
  that cache when missing.

  APR: lld-link cannot consume LTCG bitcode archives — only native COFF. The
  prebuilt apr-1.lib the CI used to cache was built with /GL (~16 MB bitcode);
  this script rebuilds APR via cmake + MSVC WITHOUT /GL (~700 KB COFF).

  Idempotent: skips work when headers and a non-LTCG apr-1.lib are present.

.PARAMETER Arch
  Target arch (default x86_64). APR's cmake build is currently x64-only here.

.PARAMETER AprVer
  APR source version to fetch (default 1.7.6, matching the CI).
#>
[CmdletBinding()]
param(
    [string]$Arch = "x86_64",
    [string]$AprVer = "1.7.6"
)

# See windows_build.ps1: native commands (cmake) write progress to stderr, which
# PowerShell 5.1 would turn into a terminating error under Stop. Rely on the
# explicit $LASTEXITCODE checks below instead.
$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LibInstallSrc = Join-Path $Root "external_lib\src"
$AprPrefix = Join-Path $Root "external_lib\apr"
$AprLib = Join-Path $AprPrefix "lib\apr-1.lib"
$OpenSslInclude = Join-Path $Root "external_lib\openssl\include\openssl"
$OpenSslMarker = Join-Path $OpenSslInclude "whrlpool.h"
$CacheRoot = if ($env:HC_EXTERNAL_LIB_CACHE) { $env:HC_EXTERNAL_LIB_CACHE } else { "C:\external_lib" }

# Seed OpenSSL headers from the runner cache (legacy CI: xcopy c:\external_lib).
# Must run before the APR early-exit so a warm APR cache does not skip this.
if (-not (Test-Path -LiteralPath $OpenSslMarker)) {
    $CachedOpenSsl = Join-Path $CacheRoot "openssl"
    $CachedMarker = Join-Path $CachedOpenSsl "include\openssl\whrlpool.h"
    if (-not (Test-Path -LiteralPath $CachedMarker)) {
        throw @"
OpenSSL headers missing at $OpenSslMarker
and no cache at $CachedMarker.
Populate C:\external_lib\openssl\include (or set HC_EXTERNAL_LIB_CACHE), then re-run.
"@
    }
    $DstOpenSsl = Join-Path $Root "external_lib\openssl"
    Write-Output "==> seeding OpenSSL headers from $CachedOpenSsl -> $DstOpenSsl"
    New-Item -ItemType Directory -Force -Path $DstOpenSsl | Out-Null
    Copy-Item -Path (Join-Path $CachedOpenSsl "*") -Destination $DstOpenSsl -Recurse -Force
    if (-not (Test-Path -LiteralPath $OpenSslMarker)) {
        throw "OpenSSL seed from $CachedOpenSsl did not produce $OpenSslMarker"
    }
} else {
    Write-Output "==> external_lib OpenSSL headers present"
}

# Idempotent: skip if a non-LTCG apr-1.lib is already present. The LTCG archive
# is ~16 MB of bitcode; the native COFF static archive is well under 5 MB.
$NonLtcgSizeThreshold = 5MB
if ((Test-Path $AprLib) -and ((Get-Item $AprLib).Length -lt $NonLtcgSizeThreshold)) {
    Write-Output "==> external_lib APR already provisioned (non-LTCG apr-1.lib present)"
    exit 0
}

Write-Output "==> provisioning external_lib APR for windows-msvc ($Arch)"

New-Item -ItemType Directory -Force -Path $LibInstallSrc | Out-Null
New-Item -ItemType Directory -Force -Path $AprPrefix | Out-Null

$AprSrcDir = Join-Path $LibInstallSrc "apr-$AprVer"
$AprZip = Join-Path $LibInstallSrc "apr-$AprVer-win32-src.zip"

if (-not (Test-Path $AprSrcDir)) {
    if (-not (Test-Path $AprZip)) {
        Write-Output "==> downloading apr-$AprVer-win32-src.zip"
        Invoke-WebRequest -Uri "https://dlcdn.apache.org/apr/apr-$AprVer-win32-src.zip" `
            -OutFile $AprZip -UseBasicParsing
    }
    Expand-Archive -Path $AprZip -DestinationPath $LibInstallSrc -Force
}

$BuildDir = Join-Path $LibInstallSrc "apr-$AprVer-build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# Configure WITHOUT /GL (LTCG) and /Qpar so lld-link can consume the archive.
# /MT links the static CRT (matches the static APR link model). Flags mirror the
# CI's MSVC build minus the LTCG-related switches.
$CFlags = "/MT /O2 /Ob2 /Oi /Ot /Zc:wchar_t /Zc:inline /Zc:preprocessor /DNDEBUG"

Push-Location $BuildDir
try {
    & cmake -G "Visual Studio 17 2022" -A x64 `
        "-DCMAKE_INSTALL_PREFIX=$AprPrefix" `
        "-DCMAKE_BUILD_TYPE=Release" `
        "-DCMAKE_C_FLAGS_RELEASE=$CFlags" `
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded" `
        -S $AprSrcDir -B .
    if ($LASTEXITCODE -ne 0) { throw "APR cmake configure failed (exit $LASTEXITCODE)" }

    & cmake --build . --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "APR build failed (exit $LASTEXITCODE)" }

    & cmake --install . --config Release
    if ($LASTEXITCODE -ne 0) { throw "APR install failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}

if (-not (Test-Path $AprLib)) { throw "APR install did not produce $AprLib" }
$SizeMB = [math]::Round((Get-Item $AprLib).Length / 1MB, 2)
Write-Output "==> external_lib APR provisioning complete ($AprLib, $SizeMB MB)"
