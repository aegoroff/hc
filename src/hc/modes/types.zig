const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");

pub const FILE_INFO_COLUMN_SEPARATOR = " | ";
pub const SFV_SEPARATOR = "    ";
pub const VALID = "File is valid";
pub const INVALID = "File is invalid";
pub const FILE_BIG_BUFFER_SIZE: usize = 1 * lib.BINARY_THOUSAND * lib.BINARY_THOUSAND;

pub const MAX_DIGEST_SIZE: usize = 64;
pub const MAX_CONTEXT_SIZE: usize = 4096;

pub const HashAlgorithmName = []const u8;

pub const RunError = error{
    UnknownHash,
    NotImplemented,
    OutOfMemory,
    OpenFailed,
    StatFailed,
    ReadFailed,
    WriteFailed,
    StreamTooLong,
    InvalidArgument,
};

pub const RunEnv = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    out: *std.Io.Writer,
};

pub const BuiltinCtx = struct {
    is_print_low_case: bool = false,
    hash_algorithm: HashAlgorithmName,
};

pub const StringCtx = struct {
    builtin: *const BuiltinCtx,
    string: []const u8,
    is_base64: bool = false,
};

pub const HashCtx = struct {
    builtin: *const BuiltinCtx,
    hash: ?[]const u8 = null,
    min: i32 = 0,
    max: i32 = 0,
    dictionary: ?[]const u8 = null,
    threads: i32 = 0,
    performance: bool = false,
    no_probe: bool = false,
    is_base64: bool = false,
};

pub const FileCtx = struct {
    builtin: *const BuiltinCtx,
    file_path: []const u8,
    save_result_path: ?[]const u8 = null,
    hash: ?[]const u8 = null,
    limit: i64 = std.math.maxInt(i64),
    offset: i64 = 0,
    show_time: bool = false,
    result_in_sfv: bool = false,
    is_verify: bool = false,
    is_base64: bool = false,
};

pub const DirCtx = struct {
    builtin: *const BuiltinCtx,
    dir_path: []const u8,
    limit: i64 = std.math.maxInt(i64),
    offset: i64 = 0,
    hash: ?[]const u8 = null,
    show_time: bool = false,
    save_result_path: ?[]const u8 = null,
    result_in_sfv: bool = false,
    is_verify: bool = false,
    include_pattern: ?[]const u8 = null,
    exclude_pattern: ?[]const u8 = null,
    recursively: bool = false,
    no_error_on_find: bool = false,
    search_hash: ?[]const u8 = null,
    is_base64: bool = false,
};

pub fn hashToHex(digest: []const u8, low_case: bool, out: []u8) []u8 {
    const hex_chars_upper = "0123456789ABCDEF";
    const hex_chars_lower = "0123456789abcdef";
    const chars = if (low_case) hex_chars_lower else hex_chars_upper;
    var i: usize = 0;
    while (i < digest.len) : (i += 1) {
        out[i * 2] = chars[digest[i] >> 4];
        out[i * 2 + 1] = chars[digest[i] & 0x0f];
    }
    return out[0 .. digest.len * 2];
}

pub fn base64EncodedLen(n: usize) usize {
    return ((n + 2) / 3) * 4;
}

pub fn hashToBase64(digest: []const u8, out: []u8) []u8 {
    const len = base64EncodedLen(digest.len);
    const enc = std.base64.standard.Encoder;
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

pub fn parseSearchHash(
    search_hash: []const u8,
    is_base64: bool,
    hash_def: *const hashes.HashDefinition,
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

test "hashToHex upper and lower" {
    const digest = [_]u8{ 0xde, 0xad, 0xbe, 0xef };
    var buf: [8]u8 = undefined;
    try std.testing.expectEqualStrings("DEADBEEF", hashToHex(&digest, false, &buf));
    try std.testing.expectEqualStrings("deadbeef", hashToHex(&digest, true, &buf));
}

test "hashToBase64 roundtrip" {
    const digest = [_]u8{ 0xde, 0xad, 0xbe, 0xef };
    var buf: [8]u8 = undefined;
    const enc = hashToBase64(&digest, &buf);
    try std.testing.expectEqualStrings("3q2+7w==", enc);
}

test "parseSearchHash hex" {
    var out: [MAX_DIGEST_SIZE]u8 = std.mem.zeroes([MAX_DIGEST_SIZE]u8);
    const tiger = hashes.getHash("tiger").?;
    try parseSearchHash("3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3", false, tiger, &out);
    try std.testing.expectEqual(@as(u8, 0x32), out[0]);
    try std.testing.expectEqual(@as(u8, 0x93), out[1]);
}

test "parseSearchHash hex rejects wrong length" {
    var out: [MAX_DIGEST_SIZE]u8 = std.mem.zeroes([MAX_DIGEST_SIZE]u8);
    const tiger = hashes.getHash("tiger").?;
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
    var out: [MAX_DIGEST_SIZE]u8 = std.mem.zeroes([MAX_DIGEST_SIZE]u8);
    const md5 = hashes.getHash("md5").?;
    // Correct length (32) but non-hex.
    try std.testing.expectError(
        error.InvalidArgument,
        parseSearchHash("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", false, md5, &out),
    );
}
