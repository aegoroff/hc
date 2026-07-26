#Requires -Version 5.1
<#
.SYNOPSIS
  Provisions Windows C deps for the Zig build: OpenSSL headers only.

.DESCRIPTION
  Mirrors scripts/build_external_libs.sh (the Linux provisioner): idempotent
  download + install when artifacts are missing. Workspace layout stays Windows-
  specific (external_lib\openssl\..., no lib/ parent).

  OpenSSL: build.zig only needs headers under external_lib\openssl\include
  (whirlpool is compiled from vendored sources, not linked against
  libcrypto.lib). On miss: optional seed from HC_EXTERNAL_LIB_CACHE /
  C:\external_lib, else download openssl sources, run Configure (generates
  configuration.h), and install public headers. Perl is required only on the
  download path.

  Idempotent: skips work when whrlpool.h is present.

.PARAMETER Arch
  Target arch (default x86_64). Reserved for future use; OpenSSL headers are
  arch-independent for our purposes.

.PARAMETER OpenSslVer
  OpenSSL source version to fetch (default 4.0.0, matching Linux).
#>
[CmdletBinding()]
param(
    [string]$Arch = "x86_64",
    [string]$OpenSslVer = "4.0.0"
)

# See windows_build.ps1: native commands write progress to stderr, which
# PowerShell 5.1 would turn into a terminating error under Stop. Rely on the
# explicit $LASTEXITCODE checks below instead.
$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LibInstallSrc = Join-Path $Root "external_lib\src"
$OpenSslPrefix = Join-Path $Root "external_lib\openssl"
$OpenSslInclude = Join-Path $OpenSslPrefix "include\openssl"
$OpenSslMarker = Join-Path $OpenSslInclude "whrlpool.h"
$CacheRoot = if ($env:HC_EXTERNAL_LIB_CACHE) { $env:HC_EXTERNAL_LIB_CACHE } else { "C:\external_lib" }

New-Item -ItemType Directory -Force -Path $LibInstallSrc | Out-Null

if (Test-Path -LiteralPath $OpenSslMarker) {
    Write-Output "==> external_lib OpenSSL headers present"
    exit 0
}

$CachedOpenSsl = Join-Path $CacheRoot "openssl"
$CachedMarker = Join-Path $CachedOpenSsl "include\openssl\whrlpool.h"
if (Test-Path -LiteralPath $CachedMarker) {
    Write-Output "==> seeding OpenSSL headers from $CachedOpenSsl -> $OpenSslPrefix"
    New-Item -ItemType Directory -Force -Path $OpenSslPrefix | Out-Null
    Copy-Item -Path (Join-Path $CachedOpenSsl "*") -Destination $OpenSslPrefix -Recurse -Force
} else {
    Write-Output "==> provisioning external_lib OpenSSL headers for windows-msvc ($OpenSslVer) (arch=$Arch)"

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
