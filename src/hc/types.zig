//! Types and output constants shared by the hc modes and the l2h fuzz stub.
//! Kept free of C so the stub can import it.

const std = @import("std");
const lib = @import("lib");
const catalog = @import("hash_catalog");

pub const FILE_INFO_COLUMN_SEPARATOR = " | ";
pub const SFV_SEPARATOR = "    ";
/// One space — compatible with GNU `*sum -c` (text mode).
pub const CHECKSUM_SEPARATOR = " ";
pub const VALID = "File is valid";
pub const INVALID = "File is invalid";
pub const FILE_BIG_BUFFER_SIZE: usize = 1 * lib.BINARY_THOUSAND * lib.BINARY_THOUSAND;

pub const MAX_DIGEST_SIZE: usize = 64;
// Defined in hash_catalog.zig so the implementation constructors can
// comptime-assert every hash context fits this stack slot.
pub const MAX_CONTEXT_SIZE = catalog.MAX_CONTEXT_SIZE;
pub const MAX_CONTEXT_ALIGN = catalog.MAX_CONTEXT_ALIGN;

pub const OFFSET_TOO_BIG = "Offset is greater than file size";

/// Byte range hashed from a file. Same meaning as `hc --offset/--limit`
/// and l2h `File.offset(n)` / `File.limit(n)`. `limit <= 0` means the rest
/// of the file.
pub const FileWindow = struct {
    offset: i64 = 0,
    limit: i64 = std.math.maxInt(i64),
};

/// Raw digest of a file window, plus the file's full size.
pub const FileDigest = struct {
    bytes: [MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([MAX_DIGEST_SIZE]u8),
    len: usize,
    /// Full file size from stat, even when only a window was hashed. When
    /// stat reports 0 (pipes, procfs) it is the number of bytes up to EOF if
    /// the read reached it.
    file_size: u64,

    pub fn slice(self: *const FileDigest) []const u8 {
        return self.bytes[0..self.len];
    }
};

pub const FileDigestError = error{
    OffsetPastEof,
    OpenFailed,
    StatFailed,
    ReadFailed,
    OutOfMemory,
};

pub const PathError = error{
    OpenFailed,
    StatFailed,
    ReadFailed,
};

/// One step of a regular-file walk. `failed.path` is a hint (root on iterate
/// errors, subdirectory path on `enter` errors); `failed.err` is the I/O error.
pub const WalkStep = union(enum) {
    file: []u8,
    failed: struct { path: []u8, err: anyerror },
};

/// Crack knobs for `restore`. `min`/`max` of 0 mean `hc hash` defaults (1 and 10).
pub const RestoreOpts = struct {
    dictionary: ?[]const u8 = null,
    min: i32 = 0,
    max: i32 = 0,
    no_probe: bool = false,
    threads: u32 = 0,
};

/// Errors of `hash.restore`: its own range checks plus what the brute-force
/// crack can fail with (thread spawn, UTF-8/UTF-16 conversion of wide
/// passwords).
pub const RestoreError = error{
    InvalidRange,
    PassmaxTooBig,
    InvalidUtf8,
    OutOfMemory,
    WriteFailed,
} || std.Thread.SpawnError || std.unicode.Utf16LeToUtf8Error;

pub const RunError = error{
    OutOfMemory,
    WriteFailed,
    InvalidArgument,
    /// A file, directory walk, or `-o` save failed. The mode already printed
    /// the reason and processed everything else; the process exits with 1.
    ProcessingFailed,
};

pub const RunEnv = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    out: *std.Io.Writer,
};

pub const StringCtx = struct {
    string: []const u8,
    is_base64: bool = false,
    low_case: bool = false,
};

pub const HashCtx = struct {
    hash: ?[]const u8 = null,
    min: i32 = 0,
    max: i32 = 0,
    dictionary: ?[]const u8 = null,
    threads: i32 = 0,
    performance: bool = false,
    no_probe: bool = false,
    is_base64: bool = false,
};

/// Shared hashing options for file and directory modes.
pub const FileOptions = struct {
    save_result_path: ?[]const u8 = null,
    hash: ?[]const u8 = null,
    limit: i64 = std.math.maxInt(i64),
    offset: i64 = 0,
    show_time: bool = false,
    result_in_sfv: bool = false,
    is_verify: bool = false,
    is_base64: bool = false,
    low_case: bool = false,
};

pub const FileCtx = struct {
    opts: FileOptions,
    file_path: []const u8,
};

pub const DirCtx = struct {
    opts: FileOptions,
    dir_path: []const u8,
    include_pattern: ?[]const u8 = null,
    exclude_pattern: ?[]const u8 = null,
    recursively: bool = false,
    no_error_on_find: bool = false,
    search_hash: ?[]const u8 = null,
};

pub fn hashToHex(digest: []const u8, low_case: bool, out: []u8) []u8 {
    // Caller must size `out` to at least digest.len * 2 (same contract as before).
    return if (low_case)
        std.fmt.bufPrint(out, "{x}", .{digest}) catch unreachable
    else
        std.fmt.bufPrint(out, "{X}", .{digest}) catch unreachable;
}

pub fn hashToBase64(digest: []const u8, out: []u8) []u8 {
    const enc = std.base64.standard.Encoder;
    const len = enc.calcSize(digest.len);
    _ = enc.encode(out[0..len], digest);
    return out[0..len];
}

pub fn formatHash(
    digest: []const u8,
    low_case: bool,
    is_base64: bool,
    hex_buf: []u8,
) []const u8 {
    if (is_base64) {
        return hashToBase64(digest, hex_buf);
    }
    return hashToHex(digest, low_case, hex_buf);
}

/// Prints why a wide-string algorithm (e.g. `ntlm`) rejected the input and
/// returns `error.InvalidArgument`, which main maps to a silent exit 1.
pub fn reportInvalidUtf8(out: *std.Io.Writer, hash_def: *const catalog.HashDefinition) RunError!void {
    try out.print("string is not valid UTF-8, {s} requires UTF-8 input\n", .{hash_def.name});
    return error.InvalidArgument;
}

pub fn parseSearchHash(
    search_hash: []const u8,
    is_base64: bool,
    hash_def: *const catalog.HashDefinition,
    out: []u8,
) !void {
    if (is_base64) {
        const dec = std.base64.standard.Decoder;
        const expected_len = hash_def.hash_length;
        const decoded_size = dec.calcSizeForSlice(search_hash) catch return error.InvalidArgument;
        if (decoded_size != expected_len) return error.InvalidArgument;
        dec.decode(out[0..expected_len], search_hash) catch return error.InvalidArgument;
    } else {
        const expected_len = hash_def.hash_length;
        // Exact length only (no truncate); odd len must fail too.
        if (search_hash.len != expected_len * 2) return error.InvalidArgument;
        // Strict hex like the base64 branch.
        _ = std.fmt.hexToBytes(out[0..expected_len], search_hash) catch return error.InvalidArgument;
    }
}

const TEST_HASHES = catalog.bindStandIns();

fn testHash(name: []const u8) ?*const catalog.HashDefinition {
    return catalog.find(&TEST_HASHES, name);
}

test "hashToHex upper and lower" {
    // Arrange
    const digest = [_]u8{ 0xde, 0xad, 0xbe, 0xef };
    var buf: [8]u8 = undefined;

    // Act

    // Assert
    try std.testing.expectEqualStrings("DEADBEEF", hashToHex(&digest, false, &buf));
    try std.testing.expectEqualStrings("deadbeef", hashToHex(&digest, true, &buf));
}

test "hashToBase64 roundtrip" {
    // Arrange
    const digest = [_]u8{ 0xde, 0xad, 0xbe, 0xef };
    var buf: [8]u8 = undefined;

    // Act
    const enc = hashToBase64(&digest, &buf);

    // Assert
    try std.testing.expectEqualStrings("3q2+7w==", enc);
}

test "parseSearchHash hex" {
    // Arrange
    var out: [MAX_DIGEST_SIZE]u8 = std.mem.zeroes([MAX_DIGEST_SIZE]u8);
    const tiger = testHash("tiger").?;

    // Act
    try parseSearchHash("3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3", false, tiger, &out);

    // Assert
    try std.testing.expectEqual(@as(u8, 0x32), out[0]);
    try std.testing.expectEqual(@as(u8, 0x93), out[1]);
}

test "parseSearchHash hex rejects wrong length" {
    // Arrange
    var out: [MAX_DIGEST_SIZE]u8 = std.mem.zeroes([MAX_DIGEST_SIZE]u8);
    const tiger = testHash("tiger").?;

    // Act

    // Assert
    // Too short (50 hex chars for 24-byte tiger).
    try std.testing.expectError(
        error.InvalidArgument,
        parseSearchHash("3293ac630c13f0245f92bbb1766e16167a4e58492dde73", false, tiger, &out),
    );
    // Too long (50 hex chars would previously truncate via @min).
    try std.testing.expectError(
        error.InvalidArgument,
        parseSearchHash("3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3aa", false, tiger, &out),
    );
    // Odd length: len/2 == hash_length must still fail.
    try std.testing.expectError(
        error.InvalidArgument,
        parseSearchHash("3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3a", false, tiger, &out),
    );
}

test "parseSearchHash hex rejects non-hex" {
    // Arrange
    var out: [MAX_DIGEST_SIZE]u8 = std.mem.zeroes([MAX_DIGEST_SIZE]u8);
    const md5 = testHash("md5").?;

    // Act

    // Assert
    // Correct length (32) but non-hex.
    try std.testing.expectError(
        error.InvalidArgument,
        parseSearchHash("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", false, md5, &out),
    );
}
