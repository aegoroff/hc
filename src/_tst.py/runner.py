"""Spawn the hc CLI and capture filtered stdout lines (ProcessRunner parity)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

GPU_DIAGNOSTICS = (
    "GPU present but driver's CUDA version",
    "GPU unavailable (driver/toolkit); using CPU only",
)


def strip_gpu_diagnostics(line: str) -> str | None:
    for diagnostic in GPU_DIAGNOSTICS:
        idx = line.find(diagnostic)
        if idx < 0:
            continue
        if idx == 0:
            return None
        line = line[:idx].rstrip()
        return None if not line.strip() else line
    return line


def resolve_hc_path() -> Path:
    """Resolve hc the same way Architecture + ArchLinux/ArchWindows did."""
    base = Path(os.environ.get("PROJECT_BASE_PATH") or Path.cwd())
    configuration = "Debug" if os.environ.get("HC_TEST_DEBUG") else "Release"
    if os.name == "nt":
        candidate = base / "x64" / configuration / "hc.exe"
    else:
        if os.environ.get("PROJECT_BASE_PATH"):
            candidate = base / f"build-x86_64-linux-gnu-{configuration}" / "hc"
        else:
            candidate = base / "build" / "hc"
    if candidate.is_file():
        return candidate.resolve()
    # Fallback: zig-out after a local `zig build`
    zig = base / "zig-out" / "bin" / ("hc.exe" if os.name == "nt" else "hc")
    if zig.is_file():
        return zig.resolve()
    raise FileNotFoundError(
        f"hc executable not found (tried {candidate} and {zig}); "
        "set PROJECT_BASE_PATH or build hc first"
    )


class ProcessRunner:
    def __init__(self, exe: Path | None = None) -> None:
        self.exe = Path(exe) if exe else resolve_hc_path()
        if not self.exe.is_file():
            raise FileNotFoundError(self.exe)

    @property
    def test_exe_path(self) -> Path:
        return self.exe

    def run(self, *command_line: str) -> list[str]:
        proc = subprocess.run(
            [str(self.exe), *command_line],
            cwd=str(self.exe.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        lines: list[str] = []
        for raw in proc.stdout.splitlines():
            if not raw or raw.isspace():
                continue
            cleaned = strip_gpu_diagnostics(raw)
            if cleaned is not None:
                lines.append(cleaned)
        if not lines and proc.returncode != 0:
            err = (proc.stderr or "").strip()
            raise RuntimeError(
                f"hc exited {proc.returncode} with empty stdout"
                + (f": {err}" if err else "")
            )
        return lines
