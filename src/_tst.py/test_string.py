"""String / hash CLI black-box tests (StringTests + CmdStringTests parity)."""
from __future__ import annotations

import base64
import re

import pytest

from hashes import HASH_CRACK_IDS, HASH_IDS, HASHES, HASHES_CRACK, Hash
from runner import ProcessRunner

SOURCE_OPT = "-s"
BASE64_OPT = "-b"
LOW_CASE_OPT = "-l"
NO_PROBE_OPT = "--noprobe"
MAX_OPT = "-x"
MIN_OPT = "-n"
DICT_OPT = "-a"
PERF_OPT = "-p"
STRING_CMD = "string"
HASH_CMD = "hash"
RESTORED_STRING_TEMPLATE = "Initial string is: {0}"
NOTHING_FOUND = "Nothing found"

NON_DEFAULT_DICTS = ("123", "0-9", "0-9a-z", "0-9A-Z")
NON_DEFAULT_DICTS_FAIL = ("a-zA-Z", "a-z", "A-Z", "abc")
BAD_THREADS = ("-1", "10000")


def _cartesian(hashes: list[Hash], extras: tuple[str, ...]):
    for h in hashes:
        for item in extras:
            yield h, item


@pytest.mark.string
@pytest.mark.parametrize("h", HASHES, ids=HASH_IDS)
def test_calc_string_full_string(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(h.algorithm, STRING_CMD, SOURCE_OPT, h.initial_string)

    # Assert
    assert len(results) == 1
    assert results[0] == h.hash_string


@pytest.mark.string
@pytest.mark.parametrize("h", HASHES, ids=HASH_IDS)
def test_calc_string_as_base64(runner: ProcessRunner, h: Hash) -> None:
    # Arrange
    expected = base64.b64encode(bytes.fromhex(h.hash_string)).decode("ascii")

    # Act
    results = runner.run(h.algorithm, STRING_CMD, BASE64_OPT, SOURCE_OPT, h.initial_string)

    # Assert
    assert len(results) == 1
    assert results[0] == expected


@pytest.mark.string
@pytest.mark.parametrize("h", HASHES, ids=HASH_IDS)
def test_calc_string_low_case(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(h.algorithm, STRING_CMD, SOURCE_OPT, h.initial_string, LOW_CASE_OPT)

    # Assert
    assert len(results) == 1
    assert results[0] == h.hash_string.lower()


@pytest.mark.string
@pytest.mark.parametrize("h", HASHES, ids=HASH_IDS)
def test_calc_string_empty(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(h.algorithm, STRING_CMD, SOURCE_OPT, "")

    # Assert
    assert len(results) == 1
    assert results[0] == h.empty_string_hash


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_default(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, HASH_CMD, NO_PROBE_OPT, SOURCE_OPT, h.hash_string, MAX_OPT, "3"
    )

    # Assert
    assert len(results) == 2
    assert results[1] == RESTORED_STRING_TEMPLATE.format(h.initial_string)


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_empty(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, HASH_CMD, NO_PROBE_OPT, SOURCE_OPT, h.empty_string_hash
    )

    # Assert
    assert results[0] == "Attempts: 0 Time 00:00:0.000 Speed: 0 attempts/second"
    assert results[1] == RESTORED_STRING_TEMPLATE.format("Empty string")
    assert len(results) == 2


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_low_case_hash(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.hash_string.lower(),
        MAX_OPT,
        "3",
        DICT_OPT,
        h.initial_string,
    )

    # Assert
    assert len(results) == 2
    assert results[1] == RESTORED_STRING_TEMPLATE.format(h.initial_string)


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize(
    "h,dict_",
    list(_cartesian(HASHES_CRACK, NON_DEFAULT_DICTS)),
    ids=[f"{h.algorithm}-{d}" for h, d in _cartesian(HASHES_CRACK, NON_DEFAULT_DICTS)],
)
def test_crack_string_non_default_dict_success(
    runner: ProcessRunner, h: Hash, dict_: str
) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.start_part_string_hash,
        DICT_OPT,
        dict_,
        MAX_OPT,
        "2",
        MIN_OPT,
        "2",
    )

    # Assert
    assert results[1] == RESTORED_STRING_TEMPLATE.format(h.initial_string[:2])
    assert len(results) == 2


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize(
    "h,dict_",
    list(_cartesian(HASHES_CRACK, NON_DEFAULT_DICTS_FAIL)),
    ids=[f"{h.algorithm}-{d}" for h, d in _cartesian(HASHES_CRACK, NON_DEFAULT_DICTS_FAIL)],
)
def test_crack_string_non_default_dict_failure(
    runner: ProcessRunner, h: Hash, dict_: str
) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.start_part_string_hash,
        DICT_OPT,
        dict_,
        MAX_OPT,
        "2",
        MIN_OPT,
        "2",
    )

    # Assert
    assert results[1] == NOTHING_FOUND
    assert len(results) == 2


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_too_short(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.hash_string,
        MAX_OPT,
        str(len(h.initial_string) - 1),
        DICT_OPT,
        h.initial_string,
    )

    # Assert
    assert len(results) == 2
    assert results[1] == NOTHING_FOUND


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_too_long_min(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.hash_string,
        MIN_OPT,
        str(len(h.initial_string) + 1),
        MAX_OPT,
        str(len(h.initial_string) + 2),
        DICT_OPT,
        h.initial_string,
    )

    # Assert
    assert len(results) == 2
    assert results[1] == NOTHING_FOUND


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_base64(runner: ProcessRunner, h: Hash) -> None:
    # Arrange
    b64 = base64.b64encode(bytes.fromhex(h.hash_string)).decode("ascii")

    # Act
    results = runner.run(
        h.algorithm, HASH_CMD, NO_PROBE_OPT, "-b", SOURCE_OPT, b64, MAX_OPT, "3"
    )

    # Assert
    assert results[1] == RESTORED_STRING_TEMPLATE.format(h.initial_string)
    assert len(results) == 2


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_single_thread(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.hash_string,
        MAX_OPT,
        "3",
        "-T",
        "1",
    )

    # Assert
    assert results[1] == RESTORED_STRING_TEMPLATE.format(h.initial_string)
    assert len(results) == 2


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize(
    "h,threads",
    list(_cartesian(HASHES_CRACK, BAD_THREADS)),
    ids=[f"{h.algorithm}-{t}" for h, t in _cartesian(HASHES_CRACK, BAD_THREADS)],
)
def test_crack_string_bad_thread_count(
    runner: ProcessRunner, h: Hash, threads: str
) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.hash_string,
        MAX_OPT,
        "3",
        "-T",
        threads,
    )

    # Assert
    assert results[2] == RESTORED_STRING_TEMPLATE.format(h.initial_string)
    assert len(results) == 3


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_single_char_max(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.middle_part_string_hash,
        MAX_OPT,
        "2",
    )

    # Assert
    assert results[1] == RESTORED_STRING_TEMPLATE.format("2")
    assert len(results) == 2


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_single_char_max_single_thread(
    runner: ProcessRunner, h: Hash
) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.middle_part_string_hash,
        MAX_OPT,
        "2",
        "-T",
        "1",
    )

    # Assert
    assert results[1] == RESTORED_STRING_TEMPLATE.format("2")
    assert len(results) == 2


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_single_char_max_non_default_dict(
    runner: ProcessRunner, h: Hash
) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        h.middle_part_string_hash,
        MAX_OPT,
        "2",
        DICT_OPT,
        "[0-9]",
    )

    # Assert
    assert results[1] == RESTORED_STRING_TEMPLATE.format("2")
    assert len(results) == 2


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
@pytest.mark.parametrize("h", HASHES_CRACK, ids=HASH_CRACK_IDS)
def test_crack_string_performance(runner: ProcessRunner, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, HASH_CMD, PERF_OPT, DICT_OPT, "12345", MAX_OPT, "5", MIN_OPT, "5"
    )

    # Assert
    assert len(results) == 3
    assert results[0].startswith("May take approximatelly:")
    assert results[2] == RESTORED_STRING_TEMPLATE.format("12345")


@pytest.mark.string
@pytest.mark.crack
@pytest.mark.xdist_group("crack")
def test_crack_string_non_ascii(runner: ProcessRunner) -> None:
    # Arrange
    algorithm = "md5"
    digest = "327108899019B3BCFFF1683FBFDAF226"

    # Act
    results = runner.run(
        algorithm,
        HASH_CMD,
        NO_PROBE_OPT,
        SOURCE_OPT,
        digest,
        DICT_OPT,
        "еграб",
        MIN_OPT,
        "6",
        MAX_OPT,
        "6",
    )

    # Assert
    assert len(results) == 2
    assert re.match(r"Initial string is: *", results[1])
