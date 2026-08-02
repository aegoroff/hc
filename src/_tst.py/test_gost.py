"""GOST CryptoPro string vectors (GostTests parity)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from runner import ProcessRunner

_VECTORS_PATH = Path(__file__).resolve().parent / "gost_tv_cryptopro.txt"
_GOST_RE = re.compile(r'^GOST\("(.*?)"\)$')


def _load_gost_vectors() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in _VECTORS_PATH.read_text(encoding="utf-8").splitlines():
        # Split on the LAST '=' so messages like 'length = 50' stay intact.
        eq = line.rfind("=")
        if eq < 0:
            continue
        left, right = line[:eq], line[eq + 1 :]
        expected = right.strip()
        match = _GOST_RE.match(left.strip())
        if not match:
            continue
        rows.append((match.group(1), expected))
    return rows


_GOST_VECTORS = _load_gost_vectors()
_GOST_IDS = [
    "empty" if s == "" else (s[:24] + "…" if len(s) > 24 else s) for s, _ in _GOST_VECTORS
]


@pytest.mark.gost
@pytest.mark.parametrize(
    "test_string,expected",
    _GOST_VECTORS,
    ids=_GOST_IDS,
)
def test_calc_string_gost(
    runner: ProcessRunner, test_string: str, expected: str
) -> None:
    # Arrange
    expectation = expected.lower()

    # Act
    results = runner.run("gost", "string", "-s", test_string)

    # Assert
    assert len(results) == 1
    assert results[0].lower() == expectation
