const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types.zig");

pub fn hashFromString(
    string: []const u8,
    hash_def: *const hashes.HashDefinition,
    digest: []u8,
    allocator: std.mem.Allocator,
) t.RunError!void {
    hashes.createStringDigest(hash_def, string, digest, allocator) catch |err| return switch (err) {
        error.InvalidUtf8 => error.InvalidArgument,
        error.OutOfMemory => error.OutOfMemory,
    };
}

pub fn strRun(
    ctx: *t.StringCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
) t.RunError!void {
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
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var sctx: t.StringCtx = .{ .builtin = &bctx, .string = "abc" };

    // Act
    try strRun(&sctx, env, hashes.getHash("tiger").?);

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "abc", expected_digest[0..24]);
    var exp_buf: [t.MAX_DIGEST_SIZE * 2]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..24], false, &exp_buf);

    const got = std.Io.Writer.buffered(&writer);
    var want_buf: [t.MAX_DIGEST_SIZE * 2 + 2]u8 = undefined;
    const want = try std.fmt.bufPrint(&want_buf, "{s}\n", .{exp_hex});

    // Assert
    try std.testing.expectEqualStrings(want, got);
}

test "strRun low case flag" {
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger", .is_print_low_case = true };
    var sctx: t.StringCtx = .{ .builtin = &bctx, .string = "" };

    // Act
    try strRun(&sctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expectEqualStrings(
        "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3\n",
        got,
    );
}

test "hashFromString base64 string mode" {
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "md2" };
    var sctx: t.StringCtx = .{ .builtin = &bctx, .string = "", .is_base64 = true };

    // Act
    try strRun(&sctx, env, hashes.getHash("md2").?);

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("md2").?, "", expected_digest[0..16]);
    var exp_buf: [t.MAX_DIGEST_SIZE * 2]u8 = undefined;
    const exp_b64 = t.hashToBase64(expected_digest[0..16], &exp_buf);

    const got = std.Io.Writer.buffered(&writer);
    var want_buf: [t.MAX_DIGEST_SIZE * 2 + 2]u8 = undefined;
    const want = try std.fmt.bufPrint(&want_buf, "{s}\n", .{exp_b64});

    // Assert
    try std.testing.expectEqualStrings(want, got);
}
