<#
.SYNOPSIS
  Hybrid build under `zig build` for Windows: builds hc/l2h for the
  x86_64-windows-msvc target, runs unit tests + the C# black-box regression,
  and produces a cpack-equivalent TGZ artefact. Mirrors linux_build.sh.

.DESCRIPTION
  Provisioning: scripts/build_external_libs.ps1 downloads/installs OpenSSL
  headers on first run (optional seed from C:\external_lib /
  HC_EXTERNAL_LIB_CACHE when present). Idempotent afterwards.

  C dependencies are prebuilt MSVC COFF artifacts; the build targets
  x86_64-windows-msvc so lld-link can link them. CUDA is required for GPU
  parity with the Linux gnu build: build.zig auto-detects nvcc (CUDA_PATH /
  CUDA_PATH_V* / stock Program Files install). Missing toolkit is a hard fail
  (pass -Dcuda=false only for intentional CPU-only tooling builds).

.PARAMETER Arch
  Target arch (default x86_64).

.EXAMPLE
  pwsh ./windows_build.ps1
#>
[CmdletBinding()]
param(
    [string]$Arch = "x86_64"
)

# NOTE: deliberately NOT $ErrorActionPreference="Stop" — PowerShell 5.1 treats
# any native-command stderr line (zig/cmake/tar write progress there) as a
# terminating error under Stop, throwing even on a successful (exit 0) run.
# Bash's `set -e` only checks exit codes; we mirror that with explicit
# $LASTEXITCODE checks after every native invocation below.
$ErrorActionPreference = "Continue"

$Version = if ($env:HC_VERSION) { $env:HC_VERSION } else { "6.0.0" }
$BuildConf = "Release"
$ZigOptimize = "ReleaseFast"
$Triple = "$Arch-windows-msvc"

$OutDir = "zig-out"
$BinDir = "bin"
$ScriptDir = $PSScriptRoot
# ArchLinux.cs / ArchWindows.cs resolve hc via PROJECT_BASE_PATH\x64\Release\hc.exe
# when set; default to the repo root so local runs match CI.
if (-not $env:PROJECT_BASE_PATH) { $env:PROJECT_BASE_PATH = $ScriptDir }

Set-Location $ScriptDir
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

# 1. Provision OpenSSL headers (idempotent; no-op when whrlpool.h is present).
# Clear stale $LASTEXITCODE (native-exe residue from prior CI steps); the child
# script ends with `exit 0` on success so a real failure still surfaces here.
$global:LASTEXITCODE = 0
& (Join-Path $ScriptDir "scripts\build_external_libs.ps1") -Arch $Arch
if ($LASTEXITCODE -ne 0) { throw "external_lib provisioning failed" }

# 2. CUDA: normalize CUDA_PATH from versioned NVIDIA installer vars when unset
#    (e.g. CUDA_PATH_V13_2), then require nvcc before zig build. Mirrors
#    linux_build.sh gnu (auto-detect); Windows hard-fails without a toolkit.
if (-not $env:CUDA_PATH) {
    $versioned = Get-ChildItem Env:CUDA_PATH_V* -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if ($versioned) {
        $env:CUDA_PATH = $versioned.Value
        Write-Output "==> CUDA_PATH unset; using $($versioned.Name)=$($env:CUDA_PATH)"
    }
}
$nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvccCmd -and $env:CUDA_PATH) {
    $nvccCandidate = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path -LiteralPath $nvccCandidate) {
        $env:Path = "$(Join-Path $env:CUDA_PATH 'bin');$env:Path"
        $nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
    }
}
if (-not $nvccCmd) {
    throw "nvcc not found. Install the CUDA toolkit and set CUDA_PATH (or CUDA_PATH_V*), or build with zig -Dcuda=false for a CPU-only stub."
}
Write-Output "==> CUDA: $($nvccCmd.Source)"

# 3. zig build (x86_64-windows-msvc target; -Dtarget kept explicit for clarity,
#    matching linux_build.sh even though it is now the native default).
#    CUDA is auto-detected by build.zig (no -Dcuda=false).
Write-Output "==> zig build -Dtarget=$Triple -Doptimize=$ZigOptimize -Dversion=$Version"
$BuildArgs = @("build", "-Dtarget=$Triple", "-Doptimize=$ZigOptimize", "-Dversion=$Version")
& zig @BuildArgs
if ($LASTEXITCODE -ne 0) { throw "zig build failed" }

# Expose artefacts under bin/ for the legacy packaging layout.
Copy-Item "$OutDir\bin\hc.exe" "$BinDir\hc.exe" -Force
Copy-Item "$OutDir\bin\l2h.exe" "$BinDir\l2h.exe" -Force -ErrorAction SilentlyContinue
Copy-Item "LICENSE.txt" "$BinDir\LICENSE.txt" -Force -ErrorAction SilentlyContinue

# 4. Unit tests (full parity with linux_build.sh — includes brute_force_test).
$TestFlags = @("test", "-Dtarget=$Triple")
Write-Output "==> zig build $($TestFlags -join ' ')"
& zig build @TestFlags --summary new
if ($LASTEXITCODE -ne 0) { throw "zig build test failed" }

# 5. C# black-box regression (parity with linux_build.sh's `dotnet test`).
#    ArchWindows.cs resolves hc via %PROJECT_BASE_PATH%\x64\Release\hc.exe — copy
#    the zig-built binary there. Run the _tst.net project (3217 scenarios:
#    string/file/dir/crack/gost). The _tst.pgo project is skipped: it is a
#    profile-guided-optimization artefact test whose HaveCount(3) on the number
#    of .exe files in the working dir is brittle and was never exercised by the
#    old Windows CI (only Linux ran dotnet test, where _tst.pgo has no concrete
#    test class).
if ($Arch -eq "x86_64") {
    $CompatDir = Join-Path $ScriptDir "x64\$BuildConf"
    New-Item -ItemType Directory -Force -Path $CompatDir | Out-Null
    Copy-Item "$OutDir\bin\hc.exe" (Join-Path $CompatDir "hc.exe") -Force
    Copy-Item "$OutDir\bin\l2h.exe" (Join-Path $CompatDir "l2h.exe") -Force -ErrorAction SilentlyContinue
    Write-Output "==> dotnet test -c $BuildConf src\_tst.net  (hc -> $CompatDir\hc.exe)"
    & dotnet test -c $BuildConf (Join-Path $ScriptDir "src\_tst.net\_tst.net.csproj")
    if ($LASTEXITCODE -ne 0) { throw "dotnet test failed" }
}

# 6. TGZ packaging (replaces cpack TGZ: hc + l2h + LICENSE per triple).
$PkgName = "hc-$Version-$Arch-pc-windows-msvc"
$Stage = Join-Path $env:TEMP "hc-pkg-$(Get-Random)"
New-Item -ItemType Directory -Force -Path (Join-Path $Stage $PkgName) | Out-Null
$PkgRoot = Join-Path $Stage $PkgName
Copy-Item "$OutDir\bin\hc.exe" $PkgRoot -Force
Copy-Item "$OutDir\bin\l2h.exe" $PkgRoot -Force -ErrorAction SilentlyContinue
Copy-Item "LICENSE.txt" $PkgRoot -Force -ErrorAction SilentlyContinue

$Tarball = Join-Path $BinDir "$PkgName.tar.gz"
# bsdtar (shipped with Windows 10+) honours the same -C <root> <name> layout as
# the GNU tar invocation in linux_build.sh.
& tar -C $Stage -czvf $Tarball $PkgName
if ($LASTEXITCODE -ne 0) { throw "packaging failed" }
Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
Write-Output "Package: $Tarball"
