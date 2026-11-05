import os

import pytest

from runner import resolve_hc_path


def test_explicit_executable_overrides_local_build(tmp_path, monkeypatch):
    # Arrange
    executable = tmp_path / "chosen hc"
    executable.touch()
    local = tmp_path / "zig-out" / "bin" / ("hc.exe" if os.name == "nt" else "hc")
    local.parent.mkdir(parents=True)
    local.touch()
    monkeypatch.setenv("PROJECT_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("HC_TEST_EXE", str(executable))

    # Act
    actual = resolve_hc_path()

    # Assert
    assert actual == executable.resolve()


def test_missing_explicit_executable_does_not_fall_back(tmp_path, monkeypatch):
    # Arrange
    local = tmp_path / "zig-out" / "bin" / ("hc.exe" if os.name == "nt" else "hc")
    local.parent.mkdir(parents=True)
    local.touch()
    monkeypatch.setenv("PROJECT_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("HC_TEST_EXE", str(tmp_path / "missing"))

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_hc_path()


def test_default_executable_ignores_old_builds(tmp_path, monkeypatch):
    # Arrange
    for directory in ("build", "x64/Release", "build-x86_64-linux-gnu-Release", "zig-out/bin"):
        folder = tmp_path / directory
        folder.mkdir(parents=True)
        for name in ("hc", "hc.exe"):
            (folder / name).touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROJECT_BASE_PATH", raising=False)
    monkeypatch.delenv("HC_TEST_EXE", raising=False)

    # Act
    actual = resolve_hc_path()

    # Assert
    assert actual == tmp_path / "zig-out" / "bin" / ("hc.exe" if os.name == "nt" else "hc")
