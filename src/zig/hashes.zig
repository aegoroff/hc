const std = @import("std");
const lib = @import("lib");

const c = @cImport({
    @cInclude("sph_tiger.h");
    @cInclude("sph_md2.h");
    @cInclude("sph_ripemd.h");
    @cInclude("blake3.h");
});

pub const InitFn = *const fn (context: *anyopaque) void;
pub const UpdateFn = *const fn (context: *anyopaque, input: [*]const u8, len: usize) void;
pub const FinalFn = *const fn (context: *anyopaque, digest: [*]u8) void;
pub const DigestFn = *const fn (digest: [*]u8, input: [*]const u8, len: usize) void;

pub const HashDefinition = struct {
    name: []const u8,
    hash_length: usize,
    weight: i32 = 0,
    use_wide_string: bool = false,
    has_gpu_implementation: bool = false,
    context_size: usize,
    init: InitFn,
    update: UpdateFn,
    final: FinalFn,
    digest: DigestFn,
};

fn streamingDigest(
    comptime Ctx: type,
    comptime initFn: anytype,
    comptime updateFn: anytype,
    comptime closeFn: anytype,
) DigestFn {
    return struct {
        fn call(digest: [*]u8, input: [*]const u8, input_len: usize) void {
            var ctx: Ctx = std.mem.zeroes(Ctx);
            initFn(&ctx);
            if (input_len != 0) updateFn(&ctx, input, input_len);
            closeFn(&ctx, digest);
        }
    }.call;
}

fn blake3Digest(digest: [*]u8, input: [*]const u8, input_len: usize) void {
    var hasher: c.blake3_hasher = std.mem.zeroes(c.blake3_hasher);
    c.blake3_hasher_init(&hasher);
    if (input_len != 0) c.blake3_hasher_update(&hasher, input, input_len);
    c.blake3_hasher_finalize(&hasher, digest, 32);
}

pub const hashes = [_]HashDefinition{
    .{
        .name = "tiger",
        .hash_length = 24,
        .context_size = @sizeOf(c.sph_tiger_context),
        .init = @ptrCast(&c.sph_tiger_init),
        .update = @ptrCast(&c.sph_tiger),
        .final = @ptrCast(&c.sph_tiger_close),
        .digest = streamingDigest(c.sph_tiger_context, c.sph_tiger_init, c.sph_tiger, c.sph_tiger_close),
    },
    .{
        .name = "tiger2",
        .hash_length = 24,
        .context_size = @sizeOf(c.sph_tiger_context),
        .init = @ptrCast(&c.sph_tiger2_init),
        .update = @ptrCast(&c.sph_tiger2),
        .final = @ptrCast(&c.sph_tiger2_close),
        .digest = streamingDigest(c.sph_tiger_context, c.sph_tiger2_init, c.sph_tiger2, c.sph_tiger2_close),
    },
    .{
        .name = "md2",
        .hash_length = 16,
        .context_size = @sizeOf(c.sph_md2_context),
        .init = @ptrCast(&c.sph_md2_init),
        .update = @ptrCast(&c.sph_md2),
        .final = @ptrCast(&c.sph_md2_close),
        .digest = streamingDigest(c.sph_md2_context, c.sph_md2_init, c.sph_md2, c.sph_md2_close),
    },
    .{
        .name = "ripemd160",
        .hash_length = 20,
        .context_size = @sizeOf(c.sph_ripemd160_context),
        .init = @ptrCast(&c.sph_ripemd160_init),
        .update = @ptrCast(&c.sph_ripemd160),
        .final = @ptrCast(&c.sph_ripemd160_close),
        .digest = streamingDigest(c.sph_ripemd160_context, c.sph_ripemd160_init, c.sph_ripemd160, c.sph_ripemd160_close),
    },
    .{
        .name = "ripemd128",
        .hash_length = 16,
        .context_size = @sizeOf(c.sph_ripemd128_context),
        .init = @ptrCast(&c.sph_ripemd128_init),
        .update = @ptrCast(&c.sph_ripemd128),
        .final = @ptrCast(&c.sph_ripemd128_close),
        .digest = streamingDigest(c.sph_ripemd128_context, c.sph_ripemd128_init, c.sph_ripemd128, c.sph_ripemd128_close),
    },
    .{
        .name = "blake3",
        .hash_length = 32,
        .context_size = @sizeOf(c.blake3_hasher),
        .init = @ptrCast(&c.blake3_hasher_init),
        .update = @ptrCast(&c.blake3_hasher_update),
        .final = @ptrCast(&c.blake3_hasher_finalize),
        .digest = &blake3Digest,
    },
};

pub fn getHash(name: []const u8) ?*const HashDefinition {
    for (&hashes) |*h| {
        if (std.ascii.eqlIgnoreCase(h.name, name)) return h;
    }
    return null;
}

pub fn compute(h: *const HashDefinition, input: []const u8, out: []u8) void {
    h.digest(out.ptr, input.ptr, input.len);
}

pub fn count() usize {
    return hashes.len;
}

fn expectHash(h: *const HashDefinition, input: []const u8, expected_hex: []const u8) !void {
    var digest: [64]u8 align(8) = std.mem.zeroes([64]u8);
    compute(h, input, &digest);
    var expected: [64]u8 = std.mem.zeroes([64]u8);
    lib.hexToBytes(expected_hex, &expected);
    try std.testing.expectEqualSlices(u8, expected[0..h.hash_length], digest[0..h.hash_length]);
}

test "tiger empty via dispatch table" {
    try expectHash(getHash("tiger").?, "", "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3");
}

test "blake3 empty via dispatch table" {
    try expectHash(getHash("blake3").?, "", "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");
}

test "md2 empty via dispatch table" {
    try expectHash(getHash("md2").?, "", "8350e5a3e24c153df2275c9f80692773");
}

test "ripemd160 empty via dispatch table" {
    try expectHash(getHash("ripemd160").?, "", "9c1185a5c5e9fc54612808977ee8f548b2258d31");
}

test "ripemd128 empty via dispatch table" {
    try expectHash(getHash("ripemd128").?, "", "cdf26213a150dc3ecb610f18f6b38b46");
}

test "getHash case-insensitive" {
    try std.testing.expect(getHash("TIGER") != null);
    try std.testing.expect(getHash("Blake3") != null);
    try std.testing.expect(getHash("nope") == null);
}

test "hash count" {
    try std.testing.expectEqual(@as(usize, 6), count());
}
