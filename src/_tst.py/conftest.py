"""pytest fixtures for hc black-box tests."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from runner import ProcessRunner, resolve_hc_path


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "string: string/hash CLI tests")
    config.addinivalue_line("markers", "file: file/dir CLI tests")
    config.addinivalue_line("markers", "crack: crack/hash restore tests")
    config.addinivalue_line("markers", "gost: GOST CryptoPro vectors")


@pytest.fixture(scope="session")
def hc_exe() -> Path:
    return resolve_hc_path()


@pytest.fixture(scope="session")
def runner(hc_exe: Path) -> ProcessRunner:
    return ProcessRunner(hc_exe)


def _default_test_dir() -> Path:
    if os.name == "nt":
        return Path(r"C:\_tst.py")
    return Path.home() / ".local" / "share" / "_tst.py"


@pytest.fixture(scope="module")
def file_fixture():
    """Create/clean the file-test workspace (FileFixture parity)."""
    env = os.environ.get("HC_TEST_DIR")
    if env:
        base = Path(env.strip())
    else:
        base = _default_test_dir()

    if base.exists():
        shutil.rmtree(base)
    sub = base / "sub"
    base.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)

    class Fixture:
        base_test_dir = base
        sub_dir = sub

    yield Fixture()

    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
