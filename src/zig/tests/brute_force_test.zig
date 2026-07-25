//! GoogleTest BruteForceTest parity: crack the digest of "123" back to the
//! original string through the C brute-force engine.
//!
//! The C++ suite parameterizes 10 dictionary scenarios over all 50 algorithms
//! (500 instantiations). The brute-force engine behavior is algorithm-agnostic
//! (dict expansion, length bounds, threading, miss handling), so every scenario
//! is exercised here with representative algorithms of differing digest sizes,
//! and the core "find 123" path is additionally verified across several families.
//!
//! bf.crackHash(..., no_probe = true) skips the "123" probe in bf_crack_hash and
//! runs bf_brute_force(passmin, passmax, dict, ...) directly — the exact call the
//! C++ tests make. ntlm (use_wide_string) needs the ANSI→char16 conversion path
//! and is covered indirectly via HashTest; its wide brute-force input wiring is
//! out of scope for the standalone Zig test (see "unknowns").

const std = @import("std");
const bf = @import("bf");
const hashes = @import("hashes");

// Compute the upper-case hex of the digest of "123" for `h` into `hex_out`.
fn hexOf123(h: *const hashes.HashDefinition, hex_out: []u8) []const u8 {
    var digest: [64]u8 align(8) = std.mem.zeroes([64]u8);
    hashes.compute(h, "123", &digest);
    const n = h.hash_length;
    for (digest[0..n], 0..) |b, i| {
        _ = std.fmt.bufPrint(hex_out[i * 2 ..][0..2], "{X:0>2}", .{b}) catch unreachable;
    }
    return hex_out[0 .. n * 2];
}

/// Run bf.crackHash with no_probe = true against the digest of "123" for `algo`,
/// returning the restored password (caller frees) or null when nothing matches.
fn crack(
    allocator: std.mem.Allocator,
    algo: []const u8,
    dict: []const u8,
    passmin: u32,
    passmax: u32,
    num_threads: u32,
) !?[]u8 {
    const h = hashes.getHash(algo) orelse return error.UnknownHash;
    var hex_buf: [128]u8 = undefined;
    const hex = hexOf123(h, &hex_buf);

    var sink: [16]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&sink);
    const result = try bf.crackHash(allocator, &writer, dict, hex, passmin, passmax, h, true, num_threads, false, false);
    return result.password;
}

// --- TEST_P(BruteForceTest, BruteForce_CrackHash_RestoredStringAsSpecified) --

test "BruteForce_CrackHash_RestoredStringAsSpecified" {
    const pw = (try crack(std.testing.allocator, "md5", "12345", 1, 4, 1)) orelse return error.NoPassword;
    defer std.testing.allocator.free(pw);
    try std.testing.expectEqualStrings("123", pw);
}

// --- base64 transform round-trip -------------------------------------------

test "BruteForce_CrackHashWithBase64TransformStep_RestoredStringAsSpecified" {
    const h = hashes.getHash("md5") orelse return error.UnknownHash;

    var digest: [64]u8 align(8) = std.mem.zeroes([64]u8);
    hashes.compute(h, "123", &digest);
    const n = h.hash_length;

    // out_hash_to_base64_string(digest) -> base64
    var b64: [128]u8 = undefined;
    const enc = std.base64.standard.Encoder;
    const b64_len = enc.calcSize(n);
    const b64_str = enc.encode(b64[0..b64_len], digest[0..n]);

    // hsh_from_base64(base64) -> hex (decode then re-encode as hex)
    const dec = std.base64.standard.Decoder;
    var decoded: [64]u8 = undefined;
    const sz = dec.calcSizeForSlice(b64_str) catch return error.BadBase64;
    dec.decode(decoded[0..sz], b64_str) catch return error.BadBase64;

    var hex: [128]u8 = undefined;
    for (decoded[0..sz], 0..) |b, i| {
        _ = std.fmt.bufPrint(hex[i * 2 ..][0..2], "{X:0>2}", .{b}) catch unreachable;
    }

    var sink: [16]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&sink);
    const result = try bf.crackHash(std.testing.allocator, &writer, "12345", hex[0 .. sz * 2], 1, 4, h, true, 1, false, false);
    const pw = result.password orelse return error.NoPassword;
    defer std.testing.allocator.free(pw);
    try std.testing.expectEqualStrings("123", pw);
}

// --- dictionary template scenarios -----------------------------------------

test "BruteForce_CrackHashDigitsDictAsTemplate_RestoredStringAsSpecified" {
    const pw = (try crack(std.testing.allocator, "md5", "0-9", 1, 3, 1)) orelse return error.NoPassword;
    defer std.testing.allocator.free(pw);
    try std.testing.expectEqualStrings("123", pw);
}

test "BruteForce_CrackHashDigitsDictAsTemplateAndCustomChars_RestoredStringAsSpecified" {
    const pw = (try crack(std.testing.allocator, "md5", "0-9+-.#~&*", 1, 3, 1)) orelse return error.NoPassword;
    defer std.testing.allocator.free(pw);
    try std.testing.expectEqualStrings("123", pw);
}

test "BruteForce_CrackHashDigitsAndLowCaseDictAsTemplate_RestoredStringAsSpecified" {
    const pw = (try crack(std.testing.allocator, "md5", "0-9a-z", 1, 3, 1)) orelse return error.NoPassword;
    defer std.testing.allocator.free(pw);
    try std.testing.expectEqualStrings("123", pw);
}

test "BruteForce_CrackHashAllDictClassesAsTemplate_RestoredStringAsSpecified" {
    const pw = (try crack(std.testing.allocator, "md5", "0-9a-zA-Z", 1, 3, 1)) orelse return error.NoPassword;
    defer std.testing.allocator.free(pw);
    try std.testing.expectEqualStrings("123", pw);
}

test "BruteForce_CrackHashAsciiDictAsTemplate_RestoredStringAsSpecified" {
    const pw = (try crack(std.testing.allocator, "md5", "ASCII", 1, 3, 1)) orelse return error.NoPassword;
    defer std.testing.allocator.free(pw);
    try std.testing.expectEqualStrings("123", pw);
}

// --- threading -------------------------------------------------------------

test "BruteForce_CrackHashManyThreads_RestoredStringAsSpecified" {
    const pw = (try crack(std.testing.allocator, "md5", "12345", 1, 4, 2)) orelse return error.NoPassword;
    defer std.testing.allocator.free(pw);
    try std.testing.expectEqualStrings("123", pw);
}

// --- miss scenarios (must return null) -------------------------------------

test "BruteForce_CrackHashTooSmallMaxLength_RestoredStringNull" {
    const pw = try crack(std.testing.allocator, "md5", "12345", 1, 2, 1);
    try std.testing.expect(pw == null);
}

test "BruteForce_CrackHashDictionaryWithoutNecessaryChars_RestoredStringNull" {
    const pw = try crack(std.testing.allocator, "md5", "345", 1, 3, 1);
    try std.testing.expect(pw == null);
}

// --- algorithm family coverage for the core find path ----------------------

const find_algos = [_][]const u8{
    "crc32", "sha256", "tiger", "sha512", "blake3", "md4", "gost",
};

test "BruteForce_CrackHash multiple algorithm families find 123" {
    for (find_algos) |algo| {
        const pw = (try crack(std.testing.allocator, algo, "12345", 1, 4, 1)) orelse {
            std.debug.print("BruteForce: {s} did not restore 123\n", .{algo});
            return error.NoPassword;
        };
        const ok = std.mem.eql(u8, "123", pw);
        std.testing.allocator.free(pw);
        if (!ok) return error.UnexpectedPassword;
    }
}
