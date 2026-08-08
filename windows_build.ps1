<#
.SYNOPSIS
  Hybrid build under `zig build` for Windows: builds hc/l2h for the
  x86_64-windows-msvc target, runs unit tests + the pytest black-box regression,
  produces a TGZ artefact (hc + l2h + LICENSE) and the NSIS installer
  (hc.setup.*.exe; hc only). Also builds a portable core2 TGZ (no installer)
  for older CPUs without SSE4.2 / SHA-NI.

.DESCRIPTION
  Provisioning: scripts/build_external_libs.ps1 downloads/builds OpenSSL
  (static libcrypto + headers) into the workspace and writes back a
  versioned tree under C:\external_lib / HC_EXTERNAL_LIB_CACHE so the next
  CI job can seed without rebuilding. Idempotent afterwards.

  C dependencies are prebuilt MSVC COFF artifacts; the build targets
  x86_64-windows-msvc so lld-link can link them. CUDA is required for GPU
  parity with the Linux gnu build: build.zig auto-detects nvcc (CUDA_PATH /
  CUDA_PATH_V* / stock Program Files install). Missing toolkit is a hard fail
  (pass -Dcuda=false only for intentional CPU-only tooling builds).

  Default CPU is haswell (SSE4.2 CRC32C HW + OpenSSL SHA-NI at runtime). A
  second `-Dcpu=core2` build ships as an archive only: soft CRC32C and OpenSSL
  software digests for older Intel CPUs; no NSIS installer.

  NSIS builds src/Install mainHLINQ.nsi after staging hc.exe to
  src/Binplace-x64/Release — same layout as the former msbuild Setup target.
  makensis is resolved from NSIS_ROOT (if valid), PATH (scoop shims), then
  stock Program Files / scoop app dirs.

.PARAMETER Arch
  Target arch for the Zig triple (default x86_64). Aliases: x64, amd64.

.EXAMPLE
  pwsh ./windows_build.ps1
#>
[CmdletBinding()]
param(
    [string]$Arch = "x86_64"
)

# Zig CPU names only (x86_64). Accept common Windows/VS aliases.
switch -Regex ($Arch.ToLowerInvariant()) {
    '^(x64|amd64)$' { $Arch = "x86_64" }
    '^(x86_64|aarch64|x86)$' { }
    default { throw "unsupported -Arch '$Arch' (use x86_64; aliases: x64, amd64)" }
}

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
# pytest runner resolves hc via %PROJECT_BASE_PATH%\x64\Release\hc.exe
# when set; default to the repo root so local runs match CI.
if (-not $env:PROJECT_BASE_PATH) { $env:PROJECT_BASE_PATH = $ScriptDir }

Set-Location $ScriptDir
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$TestResultsDir = Join-Path $ScriptDir "test-results"
New-Item -ItemType Directory -Force -Path $TestResultsDir | Out-Null

function Append-ZigSummary {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$LogFile
    )
    if (-not $env:GITHUB_STEP_SUMMARY) { return }
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("## $Title")
    [void]$sb.AppendLine()
    [void]$sb.AppendLine('```')
    $content = @()
    if (Test-Path -LiteralPath $LogFile) {
        $content = Get-Content -LiteralPath $LogFile
    }
    $summaryIdx = ($content | Select-String -Pattern '^Build Summary:' |
        Select-Object -First 1).LineNumber
    if ($summaryIdx) {
        $slice = $content[($summaryIdx - 1)..($content.Count - 1)]
        if ($slice.Count -gt 80) { $slice = $slice[($slice.Count - 80)..($slice.Count - 1)] }
        foreach ($line in $slice) { [void]$sb.AppendLine($line) }
    } else {
        $tail = if ($content.Count -gt 40) { $content[($content.Count - 40)..($content.Count - 1)] } else { $content }
        foreach ($line in $tail) { [void]$sb.AppendLine($line) }
    }
    [void]$sb.AppendLine('```')
    [void]$sb.AppendLine()
    Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Value $sb.ToString() -Encoding utf8
}

# 1. Provision OpenSSL libcrypto (idempotent; no-op when libcrypto.lib is present).
# Clear stale $LASTEXITCODE (native-exe residue from prior CI steps); the child
# script ends with `exit 0` on success so a real failure still surfaces here.
$global:LASTEXITCODE = 0
& (Join-Path $ScriptDir "scripts\build_external_libs.ps1") -Arch $Arch
if ($LASTEXITCODE -ne 0) { throw "external_lib provisioning failed" }

# 2. MSVC host compiler for nvcc: CUDA on Windows always shells out to cl.exe.
#    CI gets this from step-security/msvc-dev-cmd; local shells often do not.
#    build_external_libs.ps1 only imports vcvars when rebuilding OpenSSL, so a
#    warm external_lib cache leaves cl.exe off PATH — nvcc then exits 1 with
#    "Cannot find compiler 'cl.exe' in PATH" and zig only reports exit code 1.
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
$clCmd = Get-Command cl -ErrorAction SilentlyContinue
$needX64 = (-not $clCmd) -or
    ($env:VSCMD_ARG_TGT_ARCH -and $env:VSCMD_ARG_TGT_ARCH -ne "x64")
if ($needX64) {
    if (-not (Import-VcVars64)) {
        throw "cl.exe (MSVC x64) not on PATH. Install VS C++ tools, run from x64 Native Tools / Launch-VsDevShell.ps1 -Arch amd64, or use CI's msvc-dev-cmd."
    }
    $clCmd = Get-Command cl -ErrorAction SilentlyContinue
}
if (-not $clCmd) {
    throw "cl.exe not found after vcvars64 (required by nvcc on Windows)."
}
Write-Output "==> MSVC: $($clCmd.Source)"

# 3. CUDA: normalize CUDA_PATH from versioned NVIDIA installer vars when unset
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

# 4. zig build (x86_64-windows-msvc target; -Dtarget kept explicit for clarity,
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

# 5. Unit tests (full parity with linux_build.sh — includes brute_force_test + l2h).
#    Same -Doptimize as the product build. `test` already pulls in l2h
#    (`test-l2h` remains for local focused runs). Capture logs under
#    test-results/ and append Build Summary to Job Summary in CI.
$TestFlags = @("test", "-Dtarget=$Triple", "-Doptimize=$ZigOptimize")
Write-Output "==> zig build $($TestFlags -join ' ') --summary new"
$zigTestLog = Join-Path $TestResultsDir "zig-test.log"
$zigTestOut = & zig build @TestFlags --summary new 2>&1
$zigTestStatus = $LASTEXITCODE
$zigTestOut | ForEach-Object { Write-Output $_ }
$zigTestOut | Set-Content -LiteralPath $zigTestLog -Encoding utf8
Append-ZigSummary -Title "Zig: zig-test ($Triple)" -LogFile $zigTestLog
if ($zigTestStatus -ne 0) { throw "zig build test failed" }

# 6. pytest black-box regression (parity with linux_build.sh).
#    runner.py resolves hc via %PROJECT_BASE_PATH%\x64\Release\hc.exe — copy
#    the zig-built binary there. JUnit XML for dorny/test-reporter.
if ($Arch -eq "x86_64") {
    $CompatDir = Join-Path $ScriptDir "x64\$BuildConf"
    New-Item -ItemType Directory -Force -Path $CompatDir | Out-Null
    Copy-Item "$OutDir\bin\hc.exe" (Join-Path $CompatDir "hc.exe") -Force
    Copy-Item "$OutDir\bin\l2h.exe" (Join-Path $CompatDir "l2h.exe") -Force -ErrorAction SilentlyContinue
    Write-Output "==> pytest src/_tst.py  (hc -> $CompatDir\hc.exe)"
    # Prefer real interpreters (scoop `python`) over the Windows `py` launcher:
    # py.exe is often on PATH but prints "No installed Python found!" when no
    # install is registered with the launcher — even if scoop python works.
    $PyLauncher = $null
    foreach ($cand in @("python", "python3", "py")) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        & $cmd.Source -c "import venv" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PyLauncher = $cmd.Source
            Write-Output "==> Python for pytest: $PyLauncher"
            break
        }
    }
    if (-not $PyLauncher) { throw "Python 3 not found (need python/python3/py with venv for pytest black-box)" }
    $VenvDir = Join-Path $ScriptDir ".venv-tst"
    $VenvPy = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $VenvPy)) {
        & $PyLauncher -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed ($PyLauncher)" }
    }
    & $VenvPy -m pip install -q -r (Join-Path $ScriptDir "src\_tst.py\requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install pytest failed" }
    $env:HC_TEST_DIR = Join-Path $TestResultsDir "_tst.py-workdir"
    # PS 5.1 mangles `--junitxml=(Join-Path …)` into a bare path arg; pass as
    # two argv tokens so pytest gets an option, not a collection path.
    $junitXml = Join-Path $TestResultsDir "pytest-windows.xml"
    # Parallel via xdist; file → group "file", crack → group "crack" (GPU VRAM),
    # each group serial on one worker (--dist loadgroup).
    $pytestArgs = @(
        (Join-Path $ScriptDir "src\_tst.py"),
        "-n", "auto",
        "--dist", "loadgroup",
        "--junitxml", $junitXml
    )
    & $VenvPy -m pytest @pytestArgs
    if ($LASTEXITCODE -ne 0) { throw "pytest black-box failed" }
}

# 7. TGZ packaging: one archive with hc + l2h + LICENSE.
# Flat layout (binaries + LICENSE at archive root) matches historical releases
# and scoop/AUR expectations; both tools ship in the same package.
# NSIS installer (below) remains hc-only.
$PkgName = "hc-$Version-$Arch-pc-windows-msvc"
$Stage = Join-Path $env:TEMP "hc-pkg-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
try {
    Copy-Item "$OutDir\bin\hc.exe" (Join-Path $Stage "hc.exe") -Force
    $Members = @("hc.exe")
    $L2hExe = "$OutDir\bin\l2h.exe"
    if (Test-Path -LiteralPath $L2hExe) {
        Copy-Item $L2hExe (Join-Path $Stage "l2h.exe") -Force
        $Members += "l2h.exe"
    }
    if (Test-Path -LiteralPath "LICENSE.txt") {
        Copy-Item "LICENSE.txt" (Join-Path $Stage "LICENSE.txt") -Force
        $Members += "LICENSE.txt"
    }
    $Tarball = Join-Path $BinDir "$PkgName.tar.gz"
    # bsdtar (Windows 10+) matches the GNU tar -C layout in linux_build.sh.
    & tar -C $Stage -czvf $Tarball @Members
    if ($LASTEXITCODE -ne 0) { throw "packaging failed" }
    Write-Output "Package: $Tarball"
} finally {
    Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
}

# 8. NSIS installer (parity with msbuild Setup target in src/hc.xml).
#    Stages Binplace-x64\Release\hc.exe, renders Readme from docs/*.st, runs
#    makensis → src\Install\Release\hc.setup.<PRODUCT_VERSION>.exe.
if ($Arch -eq "x86_64") {
    function Resolve-Makensis {
        $candidates = @()
        if ($env:NSIS_ROOT) {
            $candidates += (Join-Path $env:NSIS_ROOT "makensis.exe")
        }
        $onPath = Get-Command makensis -ErrorAction SilentlyContinue
        if ($onPath) { $candidates += $onPath.Source }
        $candidates += @(
            "C:\Program Files (x86)\NSIS\makensis.exe",
            "C:\Program Files\NSIS\makensis.exe"
        )
        if ($env:SCOOP) {
            $candidates += (Join-Path $env:SCOOP "apps\nsis\current\makensis.exe")
        }
        $candidates += (Join-Path $env:USERPROFILE "scoop\apps\nsis\current\makensis.exe")

        foreach ($p in $candidates) {
            if ($p -and (Test-Path -LiteralPath $p)) { return $p }
        }
        throw "makensis not found (set NSIS_ROOT, install NSIS, or put makensis on PATH)"
    }
    $Makensis = Resolve-Makensis
    Write-Output "==> NSIS: $Makensis"

    # VIProductVersion needs four numeric components (X.X.X.X). HC_VERSION may be
    # SemVer with prerelease (6.0.0-beta1) or CI metadata (6.0.0-master.561);
    # strip that and keep major.minor.patch + Revision (same rules as hc.xml).
    $Revision = if ($env:Revision -match '^\d+$') { $env:Revision } else { "0" }
    $versionCore = ($Version -split '[-+]', 2)[0]
    $verParts = @($versionCore.Split('.') | Where-Object { $_ -match '^\d+$' })
    while ($verParts.Count -lt 3) { $verParts += '0' }
    $ProductVersion = "$($verParts[0]).$($verParts[1]).$($verParts[2]).$Revision"

    $BinplaceDir = Join-Path $ScriptDir "src\Binplace-x64\$BuildConf"
    New-Item -ItemType Directory -Force -Path $BinplaceDir | Out-Null
    Copy-Item "$OutDir\bin\hc.exe" (Join-Path $BinplaceDir "hc.exe") -Force

    $DocsDir = Join-Path $ScriptDir "docs"
    function Expand-ReadmeTemplate {
        param(
            [Parameter(Mandatory = $true)][string]$TemplatePath,
            [Parameter(Mandatory = $true)][string]$OutPath,
            [Parameter(Mandatory = $true)][string]$LangName,
            [Parameter(Mandatory = $true)][string]$AppName
        )
        $text = [System.IO.File]::ReadAllText($TemplatePath)
        $text = $text.Replace("{{langName}}", $LangName).Replace("{{appName}}", $AppName)
        [System.IO.File]::WriteAllText($OutPath, $text)
    }
    Expand-ReadmeTemplate `
        -TemplatePath (Join-Path $DocsDir "Readme.hc.en.st") `
        -OutPath (Join-Path $DocsDir "Readme.hc.en.txt") `
        -LangName "Hash Calculator" -AppName "hc"
    Expand-ReadmeTemplate `
        -TemplatePath (Join-Path $DocsDir "Readme.hc.ru.st") `
        -OutPath (Join-Path $DocsDir "Readme.hc.ru.txt") `
        -LangName "Хэш калькулятор" -AppName "hc"

    $InstallDir = Join-Path $ScriptDir "src\Install"
    $InstallOut = Join-Path $InstallDir $BuildConf
    New-Item -ItemType Directory -Force -Path $InstallOut | Out-Null

    # CodeSigner stub: NSIS !system requires exit 0; unsigned CI uses echo (hc.xml EchoCommand).
    $CodeSigner = Join-Path $ScriptDir "src\tmp.bat"
    Set-Content -Path $CodeSigner -Value "@echo off" -Encoding ASCII

    Write-Output "==> NSIS makensis PRODUCT_VERSION=$ProductVersion"
    Push-Location $InstallDir
    try {
        & $Makensis `
            "/DConfiguration=$BuildConf" `
            "/DPRODUCT_VERSION=$ProductVersion" `
            "/DCodeSigner=$CodeSigner" `
            "mainHLINQ.nsi"
        if ($LASTEXITCODE -ne 0) { throw "makensis failed" }
    } finally {
        Pop-Location
        Remove-Item -Force $CodeSigner -ErrorAction SilentlyContinue
    }

    $SetupExe = Get-ChildItem -Path $InstallOut -Filter "hc.setup.*.exe" -ErrorAction Stop |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    Copy-Item $SetupExe.FullName (Join-Path $BinDir $SetupExe.Name) -Force
    Write-Output "Installer: $($SetupExe.FullName)"
} else {
    Write-Output "==> NSIS installer skipped (arch=$Arch; only x86_64)"
}

# 9. Portable core2 build (x86_64 only): soft CRC32C + OpenSSL software digests
#    for CPUs without SSE4.2 / SHA-NI. Archive only — no tests re-run, no NSIS.
if ($Arch -eq "x86_64") {
    Write-Output "==> zig build -Dtarget=$Triple -Dcpu=core2 -Doptimize=$ZigOptimize -Dversion=$Version"
    $Core2Args = @("build", "-Dtarget=$Triple", "-Dcpu=core2", "-Doptimize=$ZigOptimize", "-Dversion=$Version")
    & zig @Core2Args
    if ($LASTEXITCODE -ne 0) { throw "zig build (core2) failed" }

    $PkgNameCore2 = "hc-$Version-$Arch-pc-windows-msvc-core2"
    $StageCore2 = Join-Path $env:TEMP "hc-pkg-core2-$(Get-Random)"
    New-Item -ItemType Directory -Force -Path $StageCore2 | Out-Null
    try {
        Copy-Item "$OutDir\bin\hc.exe" (Join-Path $StageCore2 "hc.exe") -Force
        $MembersCore2 = @("hc.exe")
        $L2hExeCore2 = "$OutDir\bin\l2h.exe"
        if (Test-Path -LiteralPath $L2hExeCore2) {
            Copy-Item $L2hExeCore2 (Join-Path $StageCore2 "l2h.exe") -Force
            $MembersCore2 += "l2h.exe"
        }
        if (Test-Path -LiteralPath "LICENSE.txt") {
            Copy-Item "LICENSE.txt" (Join-Path $StageCore2 "LICENSE.txt") -Force
            $MembersCore2 += "LICENSE.txt"
        }
        $TarballCore2 = Join-Path $BinDir "$PkgNameCore2.tar.gz"
        & tar -C $StageCore2 -czvf $TarballCore2 @MembersCore2
        if ($LASTEXITCODE -ne 0) { throw "core2 packaging failed" }
        Write-Output "Package (core2): $TarballCore2"
    } finally {
        Remove-Item -Recurse -Force $StageCore2 -ErrorAction SilentlyContinue
    }
} else {
    Write-Output "==> core2 portable build skipped (arch=$Arch; only x86_64)"
}
