#Requires -Version 5.1
<#
.SYNOPSIS
  Provisions Windows C deps for the Zig build: static OpenSSL libcrypto + headers.

.DESCRIPTION
  Mirrors scripts/build_external_libs.sh (the Linux provisioner): idempotent
  download + install when artifacts are missing. Workspace layout stays Windows-
  specific (external_lib\openssl\..., no lib/ parent).

  OpenSSL: build.zig links external_lib\openssl\lib\libcrypto.lib for
  MD5/SHA*/RIPEMD160/WHIRLPOOL (parity with CMake). On miss: optional seed from
  HC_EXTERNAL_LIB_CACHE / C:\external_lib when libcrypto.lib is present, else
  download openssl sources, Configure VC-WIN64A, nmake, nmake install_sw.
  Perl is required only on the download path.

  Idempotent: skips work when libcrypto.lib is present.

.PARAMETER Arch
  Target arch (default x86_64). Reserved for future use.

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
$OpenSslLib = Join-Path $OpenSslPrefix "lib\libcrypto.lib"
$CacheRoot = if ($env:HC_EXTERNAL_LIB_CACHE) { $env:HC_EXTERNAL_LIB_CACHE } else { "C:\external_lib" }

New-Item -ItemType Directory -Force -Path $LibInstallSrc | Out-Null

if (Test-Path -LiteralPath $OpenSslLib) {
    Write-Output "==> external_lib OpenSSL libcrypto present"
    exit 0
}

$CachedOpenSsl = Join-Path $CacheRoot "openssl"
$CachedLib = Join-Path $CachedOpenSsl "lib\libcrypto.lib"
if (Test-Path -LiteralPath $CachedLib) {
    Write-Output "==> seeding OpenSSL from $CachedOpenSsl -> $OpenSslPrefix"
    New-Item -ItemType Directory -Force -Path $OpenSslPrefix | Out-Null
    Copy-Item -Path (Join-Path $CachedOpenSsl "*") -Destination $OpenSslPrefix -Recurse -Force
} else {
    Write-Output "==> provisioning external_lib OpenSSL (libcrypto) for windows-msvc ($OpenSslVer) (arch=$Arch)"

    $perl = Get-Command perl -ErrorAction SilentlyContinue
    if (-not $perl) {
        throw "perl not found on PATH (required to Configure OpenSSL). Install Strawberry Perl, or seed via HC_EXTERNAL_LIB_CACHE / C:\external_lib\openssl (with lib\libcrypto.lib)."
    }

    $nmake = Get-Command nmake -ErrorAction SilentlyContinue
    if (-not $nmake) {
        throw "nmake not found on PATH (run from a VS developer / msvc-dev-cmd environment)."
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
        & tar -xzf $OpenSslTar -C $LibInstallSrc
        if ($LASTEXITCODE -ne 0) { throw "OpenSSL extract failed (exit $LASTEXITCODE)" }
        if (-not (Test-Path -LiteralPath $OpenSslSrcDir)) {
            throw "OpenSSL extract did not produce $OpenSslSrcDir"
        }
    }

    Push-Location $OpenSslSrcDir
    try {
        & perl Configure VC-WIN64A -static no-apps --prefix=$OpenSslPrefix `
            "--openssldir=$OpenSslPrefix\ssl"
        if ($LASTEXITCODE -ne 0) { throw "OpenSSL Configure failed (exit $LASTEXITCODE)" }
        & nmake
        if ($LASTEXITCODE -ne 0) { throw "OpenSSL nmake failed (exit $LASTEXITCODE)" }
        & nmake install_sw
        if ($LASTEXITCODE -ne 0) { throw "OpenSSL nmake install_sw failed (exit $LASTEXITCODE)" }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $OpenSslLib)) {
    throw "OpenSSL provisioning did not produce $OpenSslLib"
}
Write-Output "==> external_lib OpenSSL ready ($OpenSslLib)"
# Explicit exit so the caller sees 0: $LASTEXITCODE tracks native exes and can
# otherwise retain a stale non-zero from earlier CI/shell steps.
exit 0
