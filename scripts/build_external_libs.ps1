#Requires -Version 5.1
<#
.SYNOPSIS
  Provisions Windows C deps for the Zig build: OpenSSL headers + a static
  Apache APR archive (apr-1.lib) that lld-link can consume.

.DESCRIPTION
  Mirrors scripts/build_external_libs.sh (the Linux provisioner): idempotent
  download + build when artifacts are missing. Workspace layout stays Windows-
  specific (external_lib\{apr,openssl}\..., no lib/ parent).

  OpenSSL: build.zig only needs headers under external_lib\openssl\include
  (whirlpool is compiled from vendored sources, not linked against
  libcrypto.lib). On miss: optional seed from HC_EXTERNAL_LIB_CACHE /
  C:\external_lib, else download openssl sources, run Configure (generates
  configuration.h), and install public headers. Perl is required only on the
  download path.

  APR: lld-link cannot consume LTCG bitcode archives — only native COFF. The
  prebuilt apr-1.lib the CI used to cache was built with /GL (~16 MB bitcode);
  this script rebuilds APR via cmake + MSVC WITHOUT /GL (~700 KB COFF).

  Idempotent: skips work when headers and a non-LTCG apr-1.lib are present.

.PARAMETER Arch
  Target arch (default x86_64). APR's cmake build is currently x64-only here.

.PARAMETER AprVer
  APR source version to fetch (default 1.7.6, matching Linux).

.PARAMETER OpenSslVer
  OpenSSL source version to fetch (default 4.0.0, matching Linux).
#>
[CmdletBinding()]
param(
    [string]$Arch = "x86_64",
    [string]$AprVer = "1.7.6",
    [string]$OpenSslVer = "4.0.0"
)

# See windows_build.ps1: native commands (cmake) write progress to stderr, which
# PowerShell 5.1 would turn into a terminating error under Stop. Rely on the
# explicit $LASTEXITCODE checks below instead.
$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LibInstallSrc = Join-Path $Root "external_lib\src"
$AprPrefix = Join-Path $Root "external_lib\apr"
$AprLib = Join-Path $AprPrefix "lib\apr-1.lib"
$OpenSslPrefix = Join-Path $Root "external_lib\openssl"
$OpenSslInclude = Join-Path $OpenSslPrefix "include\openssl"
$OpenSslMarker = Join-Path $OpenSslInclude "whrlpool.h"
$CacheRoot = if ($env:HC_EXTERNAL_LIB_CACHE) { $env:HC_EXTERNAL_LIB_CACHE } else { "C:\external_lib" }

New-Item -ItemType Directory -Force -Path $LibInstallSrc | Out-Null

# ---- OpenSSL (headers only; same download-on-miss behavior as APR) ----
if (Test-Path -LiteralPath $OpenSslMarker) {
    Write-Output "==> external_lib OpenSSL headers present"
} else {
    $CachedOpenSsl = Join-Path $CacheRoot "openssl"
    $CachedMarker = Join-Path $CachedOpenSsl "include\openssl\whrlpool.h"
    if (Test-Path -LiteralPath $CachedMarker) {
        Write-Output "==> seeding OpenSSL headers from $CachedOpenSsl -> $OpenSslPrefix"
        New-Item -ItemType Directory -Force -Path $OpenSslPrefix | Out-Null
        Copy-Item -Path (Join-Path $CachedOpenSsl "*") -Destination $OpenSslPrefix -Recurse -Force
    } else {
        Write-Output "==> provisioning external_lib OpenSSL headers for windows-msvc ($OpenSslVer)"

        $perl = Get-Command perl -ErrorAction SilentlyContinue
        if (-not $perl) {
            throw "perl not found on PATH (required to Configure OpenSSL). Install Strawberry Perl, or seed headers via HC_EXTERNAL_LIB_CACHE / C:\external_lib\openssl\include."
        }

        $OpenSslSrcName = "openssl-$OpenSslVer"
        $OpenSslSrcDir = Join-Path $LibInstallSrc $OpenSslSrcName
        $OpenSslTar = Join-Path $LibInstallSrc "$OpenSslSrcName.tar.gz"
        $OpenSslUrl = "https://github.com/openssl/openssl/releases/download/$OpenSslSrcName/$OpenSslSrcName.tar.gz"

        if (-not (Test-Path -LiteralPath $OpenSslSrcDir)) {
            if (-not (Test-Path -LiteralPath $OpenSslTar)) {
                Write-Output "==> downloading $OpenSslSrcName.tar.gz"
                Invoke-WebRequest -Uri $OpenSslUrl -OutFile $OpenSslTar -UseBasicParsing
            }
            Write-Output "==> extracting $OpenSslTar"
            # Windows 10+ ships bsdtar; matches Linux tar -xzf.
            & tar -xzf $OpenSslTar -C $LibInstallSrc
            if ($LASTEXITCODE -ne 0) { throw "OpenSSL extract failed (exit $LASTEXITCODE)" }
            if (-not (Test-Path -LiteralPath $OpenSslSrcDir)) {
                throw "OpenSSL extract did not produce $OpenSslSrcDir"
            }
        }

        # Configure generates include/openssl/configuration.h. no-asm avoids a
        # NASM dependency; we never link libcrypto on Windows (vendored whirlpool).
        Push-Location $OpenSslSrcDir
        try {
            & perl Configure VC-WIN64A no-shared no-apps no-tests no-asm `
                "--prefix=$OpenSslPrefix" "--openssldir=$OpenSslPrefix\ssl"
            if ($LASTEXITCODE -ne 0) { throw "OpenSSL Configure failed (exit $LASTEXITCODE)" }
        }
        finally {
            Pop-Location
        }

        $GeneratedConf = Join-Path $OpenSslSrcDir "include\openssl\configuration.h"
        if (-not (Test-Path -LiteralPath $GeneratedConf)) {
            throw "OpenSSL Configure did not produce $GeneratedConf"
        }

        New-Item -ItemType Directory -Force -Path $OpenSslInclude | Out-Null
        Copy-Item -Path (Join-Path $OpenSslSrcDir "include\openssl\*") `
            -Destination $OpenSslInclude -Recurse -Force
    }

    if (-not (Test-Path -LiteralPath $OpenSslMarker)) {
        throw "OpenSSL provisioning did not produce $OpenSslMarker"
    }
    Write-Output "==> external_lib OpenSSL headers ready ($OpenSslMarker)"
}

# ---- APR (static non-LTCG archive for lld-link) ----
# Idempotent: skip if a non-LTCG apr-1.lib is already present. The LTCG archive
# is ~16 MB of bitcode; the native COFF static archive is well under 5 MB.
$NonLtcgSizeThreshold = 5MB
if ((Test-Path $AprLib) -and ((Get-Item $AprLib).Length -lt $NonLtcgSizeThreshold)) {
    Write-Output "==> external_lib APR already provisioned (non-LTCG apr-1.lib present)"
    exit 0
}

Write-Output "==> provisioning external_lib APR for windows-msvc ($Arch)"

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
