//! Minimal `hashes` stand-in for the l2h fuzz test binary only.
//! Wired from `build.zig` — not used by the production `l2h` executable.
//!
//! `defs` mirrors every name/`hash_length` in `src/hc/hashes.zig` (including
//! `crc32c`, which production may omit on CPUs without SSE4.2). Keep in sync
//! when adding algorithms there.

const std = @import("std");

pub const InitFn = *const fn (context: *anyopaque) callconv(.c) void;
pub const UpdateFn = *const fn (context: *anyopaque, input: [*]const u8, len: usize) callconv(.c) void;
pub const FinalFn = *const fn (context: *anyopaque, digest: [*]u8) callconv(.c) void;
pub const DigestFn = *const fn (digest: [*]u8, input: [*]const u8, len: usize) callconv(.c) void;

pub const HashDefinition = struct {
    name: []const u8,
    description: []const u8 = "",
    hash_length: usize,
    use_wide_string: bool = false,
    init: InitFn = noopInit,
    update: UpdateFn = noopUpdate,
    final: FinalFn = noopFinal,
    digest: DigestFn = noopDigest,
};

fn noopInit(_: *anyopaque) callconv(.c) void {}
fn noopUpdate(_: *anyopaque, _: [*]const u8, _: usize) callconv(.c) void {}
fn noopFinal(_: *anyopaque, _: [*]u8) callconv(.c) void {}
fn noopDigest(digest: [*]u8, _: [*]const u8, _: usize) callconv(.c) void {
    @memset(digest[0..64], 0);
}

/// Same catalog as production `hashes.hashes` (+ always-on `crc32c`).
const defs = [_]HashDefinition{
    .{ .name = "blake2b", .hash_length = 64 },
    .{ .name = "blake2b-128", .hash_length = 16 },
    .{ .name = "blake2b-160", .hash_length = 20 },
    .{ .name = "blake2b-224", .hash_length = 28 },
    .{ .name = "blake2b-256", .hash_length = 32 },
    .{ .name = "blake2b-384", .hash_length = 48 },
    .{ .name = "blake2s", .hash_length = 32 },
    .{ .name = "blake2s-128", .hash_length = 16 },
    .{ .name = "blake2s-160", .hash_length = 20 },
    .{ .name = "blake2s-224", .hash_length = 28 },
    .{ .name = "blake3", .hash_length = 32 },
    .{ .name = "adler32", .hash_length = 4 },
    .{ .name = "crc32", .hash_length = 4 },
    .{ .name = "crc32c", .hash_length = 4 },
    .{ .name = "crc64-xz", .hash_length = 8 },
    .{ .name = "crc64-ecma", .hash_length = 8 },
    .{ .name = "crc64-iso", .hash_length = 8 },
    .{ .name = "crc64-ms", .hash_length = 8 },
    .{ .name = "edonr256", .hash_length = 32 },
    .{ .name = "edonr512", .hash_length = 64 },
    .{ .name = "gost", .hash_length = 32 },
    .{ .name = "streebog256", .hash_length = 32 },
    .{ .name = "streebog512", .hash_length = 64 },
    .{ .name = "haval-128-3", .hash_length = 16 },
    .{ .name = "haval-128-4", .hash_length = 16 },
    .{ .name = "haval-128-5", .hash_length = 16 },
    .{ .name = "haval-160-3", .hash_length = 20 },
    .{ .name = "haval-160-4", .hash_length = 20 },
    .{ .name = "haval-160-5", .hash_length = 20 },
    .{ .name = "haval-192-3", .hash_length = 24 },
    .{ .name = "haval-192-4", .hash_length = 24 },
    .{ .name = "haval-192-5", .hash_length = 24 },
    .{ .name = "haval-224-3", .hash_length = 28 },
    .{ .name = "haval-224-4", .hash_length = 28 },
    .{ .name = "haval-224-5", .hash_length = 28 },
    .{ .name = "haval-256-3", .hash_length = 32 },
    .{ .name = "haval-256-4", .hash_length = 32 },
    .{ .name = "haval-256-5", .hash_length = 32 },
    .{ .name = "md2", .hash_length = 16 },
    .{ .name = "md4", .hash_length = 16 },
    .{ .name = "md5", .hash_length = 16 },
    .{ .name = "murmur3-128", .hash_length = 16 },
    .{ .name = "murmur3-32", .hash_length = 4 },
    .{ .name = "ntlm", .hash_length = 16, .use_wide_string = true },
    .{ .name = "ripemd128", .hash_length = 16 },
    .{ .name = "ripemd160", .hash_length = 20 },
    .{ .name = "ripemd256", .hash_length = 32 },
    .{ .name = "ripemd320", .hash_length = 40 },
    .{ .name = "sha-3-224", .hash_length = 28 },
    .{ .name = "sha-3-256", .hash_length = 32 },
    .{ .name = "sha-3-384", .hash_length = 48 },
    .{ .name = "sha-3-512", .hash_length = 64 },
    .{ .name = "sha-3k-224", .hash_length = 28 },
    .{ .name = "sha-3k-256", .hash_length = 32 },
    .{ .name = "sha-3k-384", .hash_length = 48 },
    .{ .name = "sha-3k-512", .hash_length = 64 },
    .{ .name = "sha1", .hash_length = 20 },
    .{ .name = "sha224", .hash_length = 28 },
    .{ .name = "sha256", .hash_length = 32 },
    .{ .name = "sha384", .hash_length = 48 },
    .{ .name = "sha512", .hash_length = 64 },
    .{ .name = "sha512-224", .hash_length = 28 },
    .{ .name = "sha512-256", .hash_length = 32 },
    .{ .name = "shake128", .hash_length = 32 },
    .{ .name = "shake256", .hash_length = 64 },
    .{ .name = "sm3", .hash_length = 32 },
    .{ .name = "snefru128", .hash_length = 16 },
    .{ .name = "snefru256", .hash_length = 32 },
    .{ .name = "tiger", .hash_length = 24 },
    .{ .name = "tiger2", .hash_length = 24 },
    .{ .name = "tth", .hash_length = 24 },
    .{ .name = "whirlpool", .hash_length = 64 },
    .{ .name = "xxhash3", .hash_length = 8 },
    .{ .name = "xxhash32", .hash_length = 4 },
    .{ .name = "xxhash64", .hash_length = 8 },
};

pub fn getHash(name: []const u8) ?*const HashDefinition {
    for (&defs) |*h| {
        if (std.ascii.eqlIgnoreCase(h.name, name)) return h;
    }
    return null;
}

pub fn ensureOpenSslReady() void {}

pub fn compute(h: *const HashDefinition, _: []const u8, out: []u8) void {
    @memset(out[0..h.hash_length], 0);
}

pub fn createStringDigest(h: *const HashDefinition, input: []const u8, out: []u8, gpa: std.mem.Allocator) !void {
    _ = input;
    _ = gpa;
    compute(h, &.{}, out);
}
