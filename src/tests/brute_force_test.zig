//! GoogleTest BruteForceTest parity: 10 scenarios × 50 algorithms (500 cases).
//!
//! Mirrors `src/_tst/BruteForceTest.cpp` INSTANTIATE_TEST_SUITE_P(All, ...):
//! crack the digest of "123" (UTF-16LE when `use_wide_string`) via
//! `bf.crackHash(..., no_probe = true)` — same path as C++ `bf_brute_force`.

const std = @import("std");
const bf = @import("bf");
const hashes = @import("hashes");

/// Same order as BruteForceTest.cpp / HashTest.h INSTANTIATE lists.
const algos = [_][]const u8{
    "crc32",       "crc32c",      "edonr256",    "edonr512",    "gost",
    "haval-128-3", "haval-128-4", "haval-128-5", "haval-160-3", "haval-160-4",
    "haval-160-5", "haval-192-3", "haval-192-4", "haval-192-5", "haval-224-3",
    "haval-224-4", "haval-224-5", "haval-256-3", "haval-256-4", "haval-256-5",
    "md2",         "md4",         "md5",         "ntlm",        "ripemd128",
    "ripemd160",   "ripemd256",   "ripemd320",   "sha-3-224",   "sha-3-256",
    "sha-3-384",   "sha-3-512",   "sha-3k-224",  "sha-3k-256",  "sha-3k-384",
    "sha-3k-512",  "sha1",        "sha224",      "sha256",      "sha384",
    "sha512",      "snefru128",   "snefru256",   "tiger",       "tiger2",
    "tth",         "whirlpool",   "blake2b",     "blake2s",     "blake3",
};

const Scenario = struct {
    dict: []const u8,
    passmin: u32,
    passmax: u32,
    threads: u32,
    expect_found: bool,
};

fn digestOf123(h: *const hashes.HashDefinition, out: []u8, allocator: std.mem.Allocator) !void {
    if (h.use_wide_string) {
        const wide = try bf.ansiToWide(allocator, "123");
        defer allocator.free(wide);
        hashes.compute(h, std.mem.sliceAsBytes(wide), out);
    } else {
        hashes.compute(h, "123", out);
    }
}

fn hexOfDigest(digest: []const u8, n: usize, hex_out: []u8) []const u8 {
    for (digest[0..n], 0..) |b, i| {
        _ = std.fmt.bufPrint(hex_out[i * 2 ..][0..2], "{X:0>2}", .{b}) catch unreachable;
    }
    return hex_out[0 .. n * 2];
}

fn crackWithHex(
    allocator: std.mem.Allocator,
    h: *const hashes.HashDefinition,
    hex: []const u8,
    dict: []const u8,
    passmin: u32,
    passmax: u32,
    num_threads: u32,
) !?[]u8 {
    var discard_buf: [256]u8 = undefined;
    var discarding: std.Io.Writer.Discarding = .init(&discard_buf);
    const result = try bf.crackHash(
        allocator,
        &discarding.writer,
        dict,
        hex,
        passmin,
        passmax,
        h,
        true, // no_probe — same as C++ bf_brute_force direct call
        num_threads,
        h.use_wide_string,
        false, // has_gpu — C++ passes gpu flag but tests run CPU-only here
    );
    return result.password;
}

fn crack(
    allocator: std.mem.Allocator,
    algo: []const u8,
    dict: []const u8,
    passmin: u32,
    passmax: u32,
    num_threads: u32,
) !?[]u8 {
    const h = hashes.getHash(algo) orelse return error.UnknownHash;
    var digest: [64]u8 align(8) = std.mem.zeroes([64]u8);
    try digestOf123(h, &digest, allocator);
    var hex_buf: [128]u8 = undefined;
    const hex = hexOfDigest(&digest, h.hash_length, &hex_buf);
    return crackWithHex(allocator, h, hex, dict, passmin, passmax, num_threads);
}

fn expectFound(algo: []const u8, s: Scenario) !void {
    const pw = (try crack(std.testing.allocator, algo, s.dict, s.passmin, s.passmax, s.threads)) orelse {
        std.debug.print("BruteForce miss: algo={s} dict={s} max={d}\n", .{ algo, s.dict, s.passmax });
        return error.NoPassword;
    };
    defer std.testing.allocator.free(pw);
    if (!std.mem.eql(u8, pw, "123")) {
        std.debug.print("BruteForce wrong pw: algo={s} got={s}\n", .{ algo, pw });
        return error.UnexpectedPassword;
    }
}

fn expectMiss(algo: []const u8, s: Scenario) !void {
    const pw = try crack(std.testing.allocator, algo, s.dict, s.passmin, s.passmax, s.threads);
    if (pw) |p| {
        defer std.testing.allocator.free(p);
        std.debug.print("BruteForce expected miss: algo={s} dict={s} got={s}\n", .{ algo, s.dict, p });
        return error.UnexpectedPassword;
    }
}

fn runAllAlgos(s: Scenario) !void {
    for (algos) |algo| {
        if (hashes.getHash(algo) == null) continue; // e.g. crc32c on aarch64
        if (s.expect_found) {
            try expectFound(algo, s);
        } else {
            try expectMiss(algo, s);
        }
    }
}

// --- TEST_P(BruteForceTest, ...) — one Zig test per scenario, all 50 algos ---

test "BruteForce_CrackHash_RestoredStringAsSpecified" {
    try runAllAlgos(.{ .dict = "12345", .passmin = 1, .passmax = 4, .threads = 1, .expect_found = true });
}

test "BruteForce_CrackHashWithBase64TransformStep_RestoredStringAsSpecified" {
    for (algos) |algo| {
        const h = hashes.getHash(algo) orelse continue; // e.g. crc32c on aarch64
        var digest: [64]u8 align(8) = std.mem.zeroes([64]u8);
        try digestOf123(h, &digest, std.testing.allocator);
        const n = h.hash_length;

        var b64: [128]u8 = undefined;
        const enc = std.base64.standard.Encoder;
        const b64_len = enc.calcSize(n);
        const b64_str = enc.encode(b64[0..b64_len], digest[0..n]);

        const dec = std.base64.standard.Decoder;
        var decoded: [64]u8 = undefined;
        const sz = dec.calcSizeForSlice(b64_str) catch return error.BadBase64;
        dec.decode(decoded[0..sz], b64_str) catch return error.BadBase64;

        var hex: [128]u8 = undefined;
        const hex_str = hexOfDigest(decoded[0..sz], sz, &hex);

        const pw = (try crackWithHex(std.testing.allocator, h, hex_str, "12345", 1, 4, 1)) orelse {
            std.debug.print("BruteForce base64 miss: algo={s}\n", .{algo});
            return error.NoPassword;
        };
        defer std.testing.allocator.free(pw);
        try std.testing.expectEqualStrings("123", pw);
    }
}

test "BruteForce_CrackHashDigitsDictAsTemplate_RestoredStringAsSpecified" {
    try runAllAlgos(.{ .dict = "0-9", .passmin = 1, .passmax = 3, .threads = 1, .expect_found = true });
}

test "BruteForce_CrackHashDigitsDictAsTemplateAndCustomChars_RestoredStringAsSpecified" {
    try runAllAlgos(.{ .dict = "0-9+-.#~&*", .passmin = 1, .passmax = 3, .threads = 1, .expect_found = true });
}

test "BruteForce_CrackHashDigitsAndLowCaseDictAsTemplate_RestoredStringAsSpecified" {
    try runAllAlgos(.{ .dict = "0-9a-z", .passmin = 1, .passmax = 3, .threads = 1, .expect_found = true });
}

test "BruteForce_CrackHashAllDictClassesAsTemplate_RestoredStringAsSpecified" {
    try runAllAlgos(.{ .dict = "0-9a-zA-Z", .passmin = 1, .passmax = 3, .threads = 1, .expect_found = true });
}

test "BruteForce_CrackHashAsciiDictAsTemplate_RestoredStringAsSpecified" {
    try runAllAlgos(.{ .dict = "ASCII", .passmin = 1, .passmax = 3, .threads = 1, .expect_found = true });
}

test "BruteForce_CrackHashManyThreads_RestoredStringAsSpecified" {
    try runAllAlgos(.{ .dict = "12345", .passmin = 1, .passmax = 4, .threads = 2, .expect_found = true });
}

test "BruteForce_CrackHashTooSmallMaxLength_RestoredStringNull" {
    try runAllAlgos(.{ .dict = "12345", .passmin = 1, .passmax = 2, .threads = 1, .expect_found = false });
}

test "BruteForce_CrackHashDictionaryWithoutNecessaryChars_RestoredStringNull" {
    try runAllAlgos(.{ .dict = "345", .passmin = 1, .passmax = 3, .threads = 1, .expect_found = false });
}
