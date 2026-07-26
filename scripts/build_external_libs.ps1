#Requires -Version 5.1
<#
.SYNOPSIS
  Provisions the C dependency the Zig build (build.zig) cannot build itself on
  Windows: a static Apache APR archive (apr-1.lib) that lld-link can consume.

.DESCRIPTION
  Mirrors scripts/build_external_libs.sh (the Linux provisioner). The Zig build
  links APR as an object file into the hc/bf modules, but lld-link (zig's linker
  for windows-msvc) cannot consume LTCG bitcode archives — only native COFF. The
  prebuilt apr-1.lib the CI used to produce was built with /GL (LTCG) and is
  ~16 MB of bitcode; this script rebuilds APR via cmake + the MSVC toolchain
  WITHOUT /GL, yielding a ~700 KB native COFF static archive.

  OpenSSL headers are already vendored under external_lib/openssl/include (the
  Zig build consumes headers only — whirlpool is compiled from vendored sources,
  not linked against libcrypto.lib), so only APR is provisioned here.

  Idempotent: skips the rebuild when a non-LTCG apr-1.lib is already present.

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
