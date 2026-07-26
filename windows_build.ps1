<#
.SYNOPSIS
  Hybrid build under `zig build` for Windows: builds hc/l2h for the
  x86_64-windows-msvc target, runs unit tests + the C# black-box regression,
  and produces a cpack-equivalent TGZ artefact. Mirrors linux_build.sh.

.DESCRIPTION
  Provisioning: scripts/build_external_libs.ps1 rebuilds APR without /GL on
  first run (lld-link cannot consume the LTCG bitcode archive the CI used to
  cache) and is a no-op afterwards. OpenSSL headers are already vendored.

  C dependencies are prebuilt MSVC COFF artifacts; the build targets
  x86_64-windows-msvc so lld-link can link them. CUDA is CPU-stubbed this pass
  (-Dcuda=false) — wiring Windows nvcc host objects is tracked separately.

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

# 1. Provision APR (idempotent; no-op when a non-LTCG apr-1.lib is present).
& (Join-Path $ScriptDir "scripts\build_external_libs.ps1") -Arch $Arch
if ($LASTEXITCODE -ne 0) { throw "external_lib provisioning failed" }

# 2. CUDA: nvcc host objects for windows-msvc are not wired this pass. Force the
#    CPU stub (GPU-accelerated hashes fall back to CPU). build.zig still warns
#    if nvcc is missing; -Dcuda=false silences it.
$CudaFlag = "-Dcuda=false"

# 3. zig build (x86_64-windows-msvc target; -Dtarget kept explicit for clarity,
#    matching linux_build.sh even though it is now the native default).
Write-Output "==> zig build -Dtarget=$Triple -Doptimize=$ZigOptimize -Dversion=$Version $CudaFlag"
$BuildArgs = @("build", "-Dtarget=$Triple", "-Doptimize=$ZigOptimize", "-Dversion=$Version", $CudaFlag)
& zig @BuildArgs
if ($LASTEXITCODE -ne 0) { throw "zig build failed" }

# Expose artefacts under bin/ for the legacy packaging layout.
Copy-Item "$OutDir\bin\hc.exe" "$BinDir\hc.exe" -Force
Copy-Item "$OutDir\bin\l2h.exe" "$BinDir\l2h.exe" -Force -ErrorAction SilentlyContinue
Copy-Item "LICENSE.txt" "$BinDir\LICENSE.txt" -Force -ErrorAction SilentlyContinue

# 4. Unit tests (full parity with linux_build.sh — includes brute_force_test).
$TestFlags = @("test", "-Dtarget=$Triple", $CudaFlag)
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
