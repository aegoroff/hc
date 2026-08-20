"""File / dir CLI black-box tests (FileTests + CmdFileTests parity)."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

from hashes import (
    HASH_FILE_IDS,
    HASH_NO_CRC32_IDS,
    HASHES_FILE,
    HASHES_WITHOUT_CRC32,
    Hash,
)
from runner import ProcessRunner

# Shared HC_TEST_DIR / -o result / _big names: keep all file cases on one xdist
# worker (--dist loadgroup). string/gost stay free to parallelize.
pytestmark = pytest.mark.xdist_group("file")

EMPTY_NAME = "empty"
NOT_EMPTY_NAME = "notempty"
DIR_CMD = "dir"
FILE_CMD = "file"
STRING_CMD = "string"
INCLUDE_OPT = "-i"
EXCLUDE_OPT = "-e"
RECURSE_OPT = "-r"
HASH_OPT = "-m"
SEARCH_OPT = "-H"
LIMIT_OPT = "-z"
OFFSET_OPT = "-q"
TIME_OPT = "-t"
BASE64_OPT = "-b"
SOURCE_OPT = "-s"
FILE_RESULT_TPL = "{0} | {2} bytes | {1}"
FILE_ERROR_TPL = "{0} | {1}"
FILE_RESULT_TIME_RE = re.compile(
    r"^(.*?) \| \d bytes \| \d\.\d{3} sec \| ([0-9a-zA-Z]{8,128})$"
)
FILE_RESULT_SFV_TPL = "{0}    {1}"
FILE_RESULT_CHECKSUM_TPL = "{0} {1}"
FILE_SEARCH_TPL = "{0} | {1} bytes"
FILE_SEARCH_TIME_RE = re.compile(r"^(.*?) \| \d bytes \| \d\.\d{3} sec$")
INVALID_NUMBER_RE = re.compile(r"Invalid parameter --\w{3,6} (\w+)\. Must be number")


def _write_not_empty(path: Path, content: str, min_size: int = 0) -> None:
    data = content.encode("ascii")
    written = 0
    with path.open("wb") as fh:
        while True:
            fh.write(data)
            written += len(data)
            if written > min_size:
                break


def _write_empty(path: Path) -> None:
    path.write_bytes(b"")


@pytest.fixture(scope="module")
def files(file_fixture):
    """Populate empty/notempty (+ under sub/) like FileTests.Initialize."""
    base = file_fixture.base_test_dir
    sub = file_fixture.sub_dir
    empty = base / EMPTY_NAME
    not_empty = base / NOT_EMPTY_NAME
    _write_empty(empty)
    _write_not_empty(not_empty, "123")
    _write_empty(sub / EMPTY_NAME)
    _write_not_empty(sub / NOT_EMPTY_NAME, "123")

    class Paths:
        base_test_dir = base
        sub_dir = sub
        empty_file = empty
        not_empty_file = not_empty

        def create_not_empty(self, path: Path, content: str, min_size: int = 0) -> None:
            _write_not_empty(path, content, min_size)

        def create_empty(self, path: Path) -> None:
            _write_empty(path)

    return Paths()


def _cartesian(hashes: list[Hash], rows: list):
    for h in hashes:
        for row in rows:
            if isinstance(row, tuple):
                yield (h, *row)
            else:
                yield (h, row)


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_small(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(h.algorithm, FILE_CMD, SOURCE_OPT, str(files.not_empty_file))

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_RESULT_TPL.format(
        files.not_empty_file, h.hash_string, len(h.initial_string)
    )


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_big(runner: ProcessRunner, files, h: Hash) -> None:
    # Arrange
    big = Path(str(files.not_empty_file) + "_big")
    files.create_not_empty(big, h.initial_string, 2 * 1024 * 1024)
    try:
        # Act
        results = runner.run(h.algorithm, FILE_CMD, SOURCE_OPT, str(big))

        # Assert
        assert len(results) == 1
        assert " Mb (2" in results[0]
    finally:
        big.unlink(missing_ok=True)


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_dir_checksumfile(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, DIR_CMD, SOURCE_OPT, str(files.base_test_dir), "--checksumfile"
    )

    # Assert — set: dir walk order is libc/FS-dependent (gnu vs musl).
    assert set(results) == {
        FILE_RESULT_CHECKSUM_TPL.format(h.empty_string_hash, files.empty_file),
        FILE_RESULT_CHECKSUM_TPL.format(h.hash_string, files.not_empty_file),
    }


@pytest.mark.file
def test_calc_dir_sfv_crc32(runner: ProcessRunner, files) -> None:
    # Arrange
    h = next(x for x in HASHES_FILE if x.algorithm == "crc32")

    # Act
    results = runner.run(
        h.algorithm, DIR_CMD, SOURCE_OPT, str(files.base_test_dir), "--sfv"
    )

    # Assert — set: dir walk order is libc/FS-dependent (gnu vs musl).
    assert set(results) == {
        FILE_RESULT_SFV_TPL.format(files.empty_file.name, h.empty_string_hash),
        FILE_RESULT_SFV_TPL.format(files.not_empty_file.name, h.hash_string),
    }


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_limit_bigger_than_size(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, FILE_CMD, SOURCE_OPT, str(files.not_empty_file), LIMIT_OPT, "10"
    )

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_RESULT_TPL.format(
        files.not_empty_file, h.hash_string, len(h.initial_string)
    )


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_validate_success(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        FILE_CMD,
        SOURCE_OPT,
        str(files.not_empty_file),
        HASH_OPT,
        h.hash_string,
    )

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_RESULT_TPL.format(
        files.not_empty_file, "File is valid", len(h.initial_string)
    )


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_validate_failure(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        FILE_CMD,
        SOURCE_OPT,
        str(files.not_empty_file),
        HASH_OPT,
        h.trail_part_string_hash,
    )

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_RESULT_TPL.format(
        files.not_empty_file, "File is invalid", len(h.initial_string)
    )


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_with_time(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, FILE_CMD, SOURCE_OPT, str(files.not_empty_file), TIME_OPT
    )

    # Assert
    assert len(results) == 1
    assert FILE_RESULT_TIME_RE.match(results[0])


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_limit(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, FILE_CMD, SOURCE_OPT, str(files.not_empty_file), LIMIT_OPT, "2"
    )

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_RESULT_TPL.format(
        files.not_empty_file, h.start_part_string_hash, len(h.initial_string)
    )


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_offset(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, FILE_CMD, SOURCE_OPT, str(files.not_empty_file), OFFSET_OPT, "1"
    )

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_RESULT_TPL.format(
        files.not_empty_file, h.trail_part_string_hash, len(h.initial_string)
    )


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_limit_and_offset(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        FILE_CMD,
        SOURCE_OPT,
        str(files.not_empty_file),
        LIMIT_OPT,
        "1",
        OFFSET_OPT,
        "1",
    )

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_RESULT_TPL.format(
        files.not_empty_file, h.middle_part_string_hash, len(h.initial_string)
    )


_NEG_ROWS = [
    ("-10", "-10", LIMIT_OPT, "limit"),
    ("-10", "-10", OFFSET_OPT, "offset"),
]


@pytest.mark.file
@pytest.mark.parametrize(
    "h,value,expectation,option,option_name",
    list(_cartesian(HASHES_FILE, _NEG_ROWS)),
    ids=[
        f"{h.algorithm}-{option_name}-{value}"
        for h, value, expectation, option, option_name in _cartesian(HASHES_FILE, _NEG_ROWS)
    ],
)
def test_calc_file_invalid_numeric(
    runner: ProcessRunner,
    files,
    h: Hash,
    value: str,
    expectation: str,
    option: str,
    option_name: str,
) -> None:
    # Act
    results = runner.run(
        h.algorithm, FILE_CMD, SOURCE_OPT, str(files.not_empty_file), option, value
    )

    # Assert
    assert len(results) == 3
    assert (
        results[2]
        == f"Invalid {option_name} option must be positive but was {expectation}"
    )


_OVERFLOW_ROWS = [
    ("18446744073709551615", LIMIT_OPT, "limit"),
    ("18446744073709551615", OFFSET_OPT, "offset"),
    ("-10223372036854775808", LIMIT_OPT, "limit"),
    ("-10223372036854775808", OFFSET_OPT, "offset"),
]


@pytest.mark.file
@pytest.mark.parametrize(
    "h,value,option,option_name",
    list(_cartesian(HASHES_FILE, _OVERFLOW_ROWS)),
    ids=[
        f"{h.algorithm}-{option_name}-{value}"
        for h, value, option, option_name in _cartesian(HASHES_FILE, _OVERFLOW_ROWS)
    ],
)
def test_calc_file_overflow_numeric(
    runner: ProcessRunner, files, h: Hash, value: str, option: str, option_name: str
) -> None:
    # Arrange — values outside i64 used to clamp (u64 max → whole file; huge
    # negatives → minInt(i64) then "must be positive"). They must fail instead.

    # Act
    results = runner.run(
        h.algorithm, FILE_CMD, SOURCE_OPT, str(files.not_empty_file), option, value
    )

    # Assert
    assert results == [
        f"Invalid parameter --{option_name} {value}. Must be a 64-bit number"
    ]


_BAD_NUM_ROWS = [("a", "1"), ("a", "0"), ("a", "a")]


@pytest.mark.file
@pytest.mark.parametrize(
    "h,limit,offset",
    list(_cartesian(HASHES_FILE, _BAD_NUM_ROWS)),
    ids=[
        f"{h.algorithm}-{limit}-{offset}"
        for h, limit, offset in _cartesian(HASHES_FILE, _BAD_NUM_ROWS)
    ],
)
def test_calc_file_limit_offset_incorrect_numbers(
    runner: ProcessRunner, files, h: Hash, limit: str, offset: str
) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        FILE_CMD,
        SOURCE_OPT,
        str(files.not_empty_file),
        LIMIT_OPT,
        limit,
        OFFSET_OPT,
        offset,
    )

    # Assert
    assert INVALID_NUMBER_RE.match(results[0])


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_offset_greater_than_size(
    runner: ProcessRunner, files, h: Hash
) -> None:
    # Act
    results = runner.run(
        h.algorithm, FILE_CMD, SOURCE_OPT, str(files.not_empty_file), OFFSET_OPT, "4"
    )

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_ERROR_TPL.format(
        files.not_empty_file, "Offset is greater than file size"
    )


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_big_with_offset(runner: ProcessRunner, files, h: Hash) -> None:
    # Arrange
    big = Path(str(files.not_empty_file) + "_big")
    files.create_not_empty(big, h.initial_string, 2 * 1024 * 1024)
    try:
        # Act
        results = runner.run(
            h.algorithm, FILE_CMD, SOURCE_OPT, str(big), OFFSET_OPT, "1024"
        )

        # Assert
        assert len(results) == 1
        assert " Mb (2" in results[0]
    finally:
        big.unlink(missing_ok=True)


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_big_with_limit_and_offset(
    runner: ProcessRunner, files, h: Hash
) -> None:
    # Arrange
    big = Path(str(files.not_empty_file) + "_big")
    files.create_not_empty(big, h.initial_string, 2 * 1024 * 1024)
    try:
        # Act
        results = runner.run(
            h.algorithm,
            FILE_CMD,
            SOURCE_OPT,
            str(big),
            OFFSET_OPT,
            "1024",
            LIMIT_OPT,
            "1048500",
        )

        # Assert
        assert len(results) == 1
        assert " Mb (2" in results[0]
    finally:
        big.unlink(missing_ok=True)


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_unexist(runner: ProcessRunner, h: Hash) -> None:
    # Arrange
    unexist = "u"

    # Act
    results = runner.run(h.algorithm, FILE_CMD, SOURCE_OPT, unexist)

    # Assert
    assert len(results) == 1
    assert not re.match(rf"{unexist} \| .+ bytes \| .+", results[0])
    assert re.match(rf"{unexist} \| .+?", results[0])


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_file_empty(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(h.algorithm, FILE_CMD, SOURCE_OPT, str(files.empty_file))

    # Assert
    assert results[0] == FILE_RESULT_TPL.format(files.empty_file, h.empty_string_hash, 0)
    assert len(results) == 1


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_dir_single(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(h.algorithm, DIR_CMD, SOURCE_OPT, str(files.base_test_dir))

    # Assert — set: dir walk order is libc/FS-dependent (gnu vs musl).
    assert set(results) == {
        FILE_RESULT_TPL.format(files.empty_file, h.empty_string_hash, 0),
        FILE_RESULT_TPL.format(
            files.not_empty_file, h.hash_string, len(h.initial_string)
        ),
    }


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_dir_base64(runner: ProcessRunner, files, h: Hash) -> None:
    # Arrange
    string_results = runner.run(
        h.algorithm, STRING_CMD, SOURCE_OPT, h.initial_string, BASE64_OPT
    )
    empty_string_results = runner.run(h.algorithm, STRING_CMD, SOURCE_OPT, "", BASE64_OPT)

    # Act
    results = runner.run(
        h.algorithm, DIR_CMD, SOURCE_OPT, str(files.base_test_dir), BASE64_OPT
    )

    # Assert — set: dir walk order is libc/FS-dependent (gnu vs musl).
    assert set(results) == {
        FILE_RESULT_TPL.format(files.empty_file, empty_string_results[0], 0),
        FILE_RESULT_TPL.format(
            files.not_empty_file, string_results[0], len(h.initial_string)
        ),
    }


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_dir_output_to_file(runner: ProcessRunner, files, h: Hash) -> None:
    # Arrange
    save = "result"
    result_path = runner.test_exe_path.parent / save
    try:
        # Act
        results = runner.run(
            h.algorithm, DIR_CMD, SOURCE_OPT, str(files.base_test_dir), "-o", save
        )

        # Assert — set: dir walk order is libc/FS-dependent (gnu vs musl).
        assert set(results) == {
            FILE_RESULT_TPL.format(files.empty_file, h.empty_string_hash, 0),
            FILE_RESULT_TPL.format(
                files.not_empty_file, h.hash_string, len(h.initial_string)
            ),
        }
        assert result_path.is_file()
        # Binary decode preserves Windows CRLF from save.zig (legacy CRT text mode).
        # Path.read_text() would normalize \r\n → \n and break the os.linesep match
        # that the C# suite got via File.ReadAllText.
        content = result_path.read_bytes().decode("utf-8")
        from_console = os.linesep.join(results) + os.linesep
        assert from_console == content
    finally:
        result_path.unlink(missing_ok=True)


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_WITHOUT_CRC32, ids=HASH_NO_CRC32_IDS)
def test_calc_dir_sfv_unsupported(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm, DIR_CMD, SOURCE_OPT, str(files.base_test_dir), "--sfv"
    )

    # Assert
    assert results[0] == "error: unrecognized option 'sfv'"


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_dir_recursively_many_subs(runner: ProcessRunner, files, h: Hash) -> None:
    # Arrange
    sub2 = Path(str(files.sub_dir) + "2")
    sub2.mkdir(parents=True, exist_ok=True)
    files.create_empty(sub2 / EMPTY_NAME)
    files.create_not_empty(sub2 / NOT_EMPTY_NAME, h.initial_string)
    try:
        # Act
        results = runner.run(
            h.algorithm, DIR_CMD, SOURCE_OPT, str(files.base_test_dir), RECURSE_OPT
        )

        # Assert
        assert len(results) == 6
    finally:
        shutil.rmtree(sub2, ignore_errors=True)


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_dir_include_filter(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        DIR_CMD,
        SOURCE_OPT,
        str(files.base_test_dir),
        INCLUDE_OPT,
        EMPTY_NAME,
    )

    # Assert
    assert results[0] == FILE_RESULT_TPL.format(files.empty_file, h.empty_string_hash, 0)
    assert len(results) == 1


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_calc_dir_exclude_filter(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        DIR_CMD,
        SOURCE_OPT,
        str(files.base_test_dir),
        EXCLUDE_OPT,
        EMPTY_NAME,
    )

    # Assert
    assert results[0] == FILE_RESULT_TPL.format(
        files.not_empty_file, h.hash_string, len(h.initial_string)
    )
    assert len(results) == 1


_DIR_THEORY_ROWS = [
    (0, (INCLUDE_OPT, EMPTY_NAME, EXCLUDE_OPT, EMPTY_NAME)),
    (0, (EXCLUDE_OPT, f"{EMPTY_NAME};{NOT_EMPTY_NAME}")),
    (2, (INCLUDE_OPT, f"{EMPTY_NAME};{NOT_EMPTY_NAME}")),
    (2, (INCLUDE_OPT, EMPTY_NAME, RECURSE_OPT)),
    (2, (EXCLUDE_OPT, EMPTY_NAME, RECURSE_OPT)),
    (4, (RECURSE_OPT,)),
]


@pytest.mark.file
@pytest.mark.parametrize(
    "h,count_results,cmdline",
    list(_cartesian(HASHES_FILE, _DIR_THEORY_ROWS)),
    ids=[
        f"{h.algorithm}-{count}-{'-'.join(cmd)}"
        for h, count, cmd in _cartesian(HASHES_FILE, _DIR_THEORY_ROWS)
    ],
)
def test_calc_dir_different_options(
    runner: ProcessRunner, files, h: Hash, count_results: int, cmdline: tuple[str, ...]
) -> None:
    # Arrange
    args = [h.algorithm, DIR_CMD, SOURCE_OPT, str(files.base_test_dir), *cmdline]

    # Act
    results = runner.run(*args)

    # Assert
    assert len(results) == count_results


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_search_file_not_recursively(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        DIR_CMD,
        SOURCE_OPT,
        str(files.base_test_dir),
        SEARCH_OPT,
        h.hash_string,
    )

    # Assert
    assert len(results) == 1
    assert results[0] == FILE_SEARCH_TPL.format(
        files.not_empty_file, len(h.initial_string)
    )


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_search_file_not_recursively_timed(
    runner: ProcessRunner, files, h: Hash
) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        DIR_CMD,
        SOURCE_OPT,
        str(files.base_test_dir),
        SEARCH_OPT,
        h.hash_string,
        TIME_OPT,
    )

    # Assert
    assert len(results) == 1
    # Prefer timed form when present; plain search line is also accepted (C#
    # MatchRegex left `|` unescaped and matched either).
    assert FILE_SEARCH_TIME_RE.match(results[0]) or results[
        0
    ] == FILE_SEARCH_TPL.format(files.not_empty_file, len(h.initial_string))


@pytest.mark.file
@pytest.mark.parametrize("h", HASHES_FILE, ids=HASH_FILE_IDS)
def test_search_file_recursively(runner: ProcessRunner, files, h: Hash) -> None:
    # Act
    results = runner.run(
        h.algorithm,
        DIR_CMD,
        SOURCE_OPT,
        str(files.base_test_dir),
        SEARCH_OPT,
        h.hash_string,
        RECURSE_OPT,
    )

    # Assert
    assert len(results) == 2
