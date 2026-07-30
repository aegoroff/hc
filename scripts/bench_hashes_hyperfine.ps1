#Requires -Version 5.1
<#
.SYNOPSIS
  Compare performance of two hc builds with hyperfine (Windows).

.DESCRIPTION
  Mirrors scripts/bench_hashes_poop.sh: for each common hash algorithm, run
  hyperfine comparing <new> vs <old>.

  Expects:
    <base_dir>\<new_subdir>\<binary>
    <base_dir>\<old_subdir>\<binary>

  Env (optional):
    BINARY      executable name under each subdir (default: hc.exe)
    HASHES      space-separated hash list (default: intersection of both builds)
    BENCH_ARGS  args after "<bin> <hash>" (default: hash -p --noprobe)
    RUNS        hyperfine --runs (default: unset = auto)
    WARMUP      hyperfine --warmup (default: unset)
    OUT_DIR     write per-hash logs here (default: unset = stdout only)

.PARAMETER BaseDir
  Directory that contains version subdirectories.

.PARAMETER NewSub
  Subdirectory with the new binary (default: msvc).

.PARAMETER OldSub
  Subdirectory with the old binary (default: old).

.EXAMPLE
  .\scripts\bench_hashes_hyperfine.ps1 C:\builds\hc_x86_64 msvc old

.EXAMPLE
  $env:BENCH_ARGS = 'file -p --noprobe C:\data\big.bin'
  .\scripts\bench_hashes_hyperfine.ps1 C:\builds\hc_x86_64
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$BaseDir,

    [Parameter(Position = 1)]
    [string]$NewSub = "msvc",

    [Parameter(Position = 2)]
    [string]$OldSub = "old"
)

# Native stderr must not become terminating errors under PS 5.1 (see windows_build.ps1).
$ErrorActionPreference = "Continue"

# Quote for cmd.exe (hyperfine's default shell on Windows).
function Quote-CmdArg([string]$Arg) {
    if ($Arg -notmatch '[\s&<>|^()"]') { return $Arg }
    return '"' + ($Arg -replace '"', '""') + '"'
}

function Format-Command([string[]]$Parts) {
    ($Parts | ForEach-Object { Quote-CmdArg $_ }) -join ' '
}

function Get-HashList([string]$Bin) {
    $lines = & $Bin --help 2>&1 | ForEach-Object { "$_" }
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($line in $lines) {
        if ($line -match '^\s+([a-z][a-z0-9-]*)\s*$') {
            $name = $Matches[1]
            if ($name -ne 'default' -and $name -ne 'help') {
                [void]$names.Add($name)
            }
        }
    }
    return @($names | Sort-Object -Unique)
}

if (-not (Test-Path -LiteralPath $BaseDir)) {
    Write-Error "error: base directory not found: $BaseDir"
    exit 1
}
$BaseDir = (Resolve-Path -LiteralPath $BaseDir).Path

$Binary = if ($env:BINARY) { $env:BINARY } else { "hc.exe" }
$BenchArgs = if ($env:BENCH_ARGS) { $env:BENCH_ARGS } else { "hash -p --noprobe" }
$OutDir = $env:OUT_DIR
$Runs = $env:RUNS
$Warmup = $env:WARMUP

$NewBin = Join-Path (Join-Path $BaseDir $NewSub) $Binary
$OldBin = Join-Path (Join-Path $BaseDir $OldSub) $Binary

foreach ($bin in @($NewBin, $OldBin)) {
    if (-not (Test-Path -LiteralPath $bin)) {
        Write-Error "error: executable not found: $bin"
        exit 1
    }
}

$hyperfine = Get-Command hyperfine -ErrorAction SilentlyContinue
if (-not $hyperfine) {
    Write-Error "error: hyperfine not found in PATH (https://github.com/sharkdp/hyperfine)"
    exit 1
}

$NewHashes = Get-HashList $NewBin
$OldHashes = Get-HashList $OldBin

if ($env:HASHES) {
    $Selected = @($env:HASHES -split '\s+' | Where-Object { $_ })
} else {
    $Selected = @($NewHashes | Where-Object { $OldHashes -contains $_ })
}

if ($Selected.Count -eq 0) {
    Write-Error "error: no hashes to benchmark"
    exit 1
}

$OnlyNew = @($NewHashes | Where-Object { $OldHashes -notcontains $_ })
$OnlyOld = @($OldHashes | Where-Object { $NewHashes -notcontains $_ })

Write-Output "=== hc benchmark (hyperfine) ==="
Write-Output "new: $NewBin"
Write-Output "old: $OldBin"
if ($Runs) {
    Write-Output "runs: $Runs"
} else {
    Write-Output "runs: auto"
}
if ($Warmup) {
    Write-Output "warmup: $Warmup"
}
Write-Output "bench args: $BenchArgs"
Write-Output "hashes: $($Selected.Count)"
if ($OnlyNew.Count -gt 0) {
    Write-Output ("skipped (only in new): " + ($OnlyNew -join ' '))
}
if ($OnlyOld.Count -gt 0) {
    Write-Output ("skipped (only in old): " + ($OnlyOld -join ' '))
}
Write-Output ""

if ($OutDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

$BenchArr = if ($BenchArgs.Trim()) { @($BenchArgs -split '\s+' | Where-Object { $_ }) } else { @() }

$failed = 0
foreach ($hash in $Selected) {
    Write-Output "------------------------------------------------------------"
    Write-Output "### $hash"
    Write-Output "------------------------------------------------------------"

    $newCmd = Format-Command (@($NewBin, $hash) + $BenchArr)
    $oldCmd = Format-Command (@($OldBin, $hash) + $BenchArr)

    $hfArgs = [System.Collections.Generic.List[string]]::new()
    [void]$hfArgs.Add('--command-name')
    [void]$hfArgs.Add('new')
    [void]$hfArgs.Add('--command-name')
    [void]$hfArgs.Add('old')
    if ($Runs) {
        [void]$hfArgs.Add('--runs')
        [void]$hfArgs.Add("$Runs")
    }
    if ($Warmup) {
        [void]$hfArgs.Add('--warmup')
        [void]$hfArgs.Add("$Warmup")
    }
    [void]$hfArgs.Add($newCmd)
    [void]$hfArgs.Add($oldCmd)

    if ($OutDir) {
        $outFile = Join-Path $OutDir "$hash.txt"
        & hyperfine @hfArgs 2>&1 | Tee-Object -FilePath $outFile
    } else {
        & hyperfine @hfArgs
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "hyperfine failed for $hash"
        $failed++
    }
    Write-Output ""
}

Write-Output "=== done: $($Selected.Count) hashes, $failed failures ==="
if ($failed -gt 0) { exit 1 } else { exit 0 }
