const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const bf = @import("bf");
const t = @import("types.zig");

pub const StringCtx = t.StringCtx;
pub const RunEnv = t.RunEnv;
pub const RunError = t.RunError;

pub fn hashFromString(
    string: []const u8,
    hash_def: *const hashes.HashDefinition,
    digest: []u8,
    allocator: std.mem.Allocator,
) RunError!void {
    if (hash_def.use_wide_string) {
        const wide = bf.ansiToWide(allocator, string) catch |err| switch (err) {
            error.InvalidUtf8 => return error.InvalidArgument,
            error.OutOfMemory => return error.OutOfMemory,
        };
        defer allocator.free(wide);
        hashes.compute(hash_def, std.mem.sliceAsBytes(wide), digest);
    } else {
        hashes.compute(hash_def, string, digest);
    }
}

pub fn strRun(
    ctx: *StringCtx,
    env: RunEnv,
    hash_def: *const hashes.HashDefinition,
) RunError!void {
    var digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    try hashFromString(ctx.string, hash_def, digest[0..hash_def.hash_length], env.allocator);

    var repr_buf: [t.MAX_DIGEST_SIZE * 2 + 8]u8 = undefined;
    const repr = t.formatHash(
        digest[0..hash_def.hash_length],
        ctx.builtin.is_print_low_case,
        ctx.is_base64,
        &repr_buf,
    );
    try env.out.writeAll(repr);
    try lib.newLine(env.out);
}

test "strRun computes tiger hex of string" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var sctx: StringCtx = .{ .builtin = &bctx, .string = "abc" };

    try strRun(&sctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(got.len > 0);
    try std.testing.expectEqual(@as(usize, 24 * 2 + 1), got.len);
}

test "strRun low case flag" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger", .is_print_low_case = true };
    var sctx: StringCtx = .{ .builtin = &bctx, .string = "" };

    try strRun(&sctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expectEqualStrings(
        "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3\n",
        got,
    );
}

test "hashFromString base64 string mode" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "md2" };
    var sctx: StringCtx = .{ .builtin = &bctx, .string = "", .is_base64 = true };

    try strRun(&sctx, env, hashes.getHash("md2").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(got.len > 0);
}
