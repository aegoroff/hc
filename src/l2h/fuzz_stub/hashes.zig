//! Minimal `hashes` stand-in for the l2h fuzz test binary only.
//! Wired from `build.zig` — not used by the production `l2h` executable.

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

/// Names that appear as hash properties / hash-check methods in queries.
const defs = [_]HashDefinition{
    .{ .name = "md5", .hash_length = 16 },
    .{ .name = "md4", .hash_length = 16 },
    .{ .name = "md2", .hash_length = 16 },
    .{ .name = "sha1", .hash_length = 20 },
    .{ .name = "sha224", .hash_length = 28 },
    .{ .name = "sha256", .hash_length = 32 },
    .{ .name = "sha384", .hash_length = 48 },
    .{ .name = "sha512", .hash_length = 64 },
    .{ .name = "sha512-224", .hash_length = 28 },
    .{ .name = "sha512-256", .hash_length = 32 },
    .{ .name = "ripemd160", .hash_length = 20 },
    .{ .name = "whirlpool", .hash_length = 64 },
    .{ .name = "blake2b", .hash_length = 64 },
    .{ .name = "blake2s", .hash_length = 32 },
    .{ .name = "sha-3-256", .hash_length = 32 },
    .{ .name = "sha-3-512", .hash_length = 64 },
    .{ .name = "sm3", .hash_length = 32 },
    .{ .name = "crc32", .hash_length = 4 },
    .{ .name = "ntlm", .hash_length = 16 },
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
