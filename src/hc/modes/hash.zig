const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const bf = @import("bf");
const t = @import("types.zig");

const str = @import("str.zig");

pub const MIN_DEFAULT: i32 = 1;

pub const CrackParams = struct {
    dictionary: []const u8,
    passmin: i32,
    passmax: i32,
};

pub fn resolveCrackParams(ctx: *const t.HashCtx) CrackParams {
    return .{
        .dictionary = ctx.dictionary orelse bf.DEFAULT_ALPHABET,
        .passmin = if (ctx.min > 0) ctx.min else MIN_DEFAULT,
        .passmax = if (ctx.max > 0) ctx.max else @intCast(bf.MAX_DEFAULT),
    };
}

pub const TargetHash = struct {
    bytes: [t.MAX_DIGEST_SIZE]u8 align(8),
    has_value: bool,
};

pub fn resolveTargetHash(
    ctx: *const t.HashCtx,
    hash_def: *const hashes.HashDefinition,
    allocator: std.mem.Allocator,
) t.RunError!TargetHash {
    var result: TargetHash = .{ .bytes = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8), .has_value = false };

    if (ctx.performance) {
        const source: []const u8 = if (ctx.hash != null and ctx.hash.?.len > 0) ctx.hash.? else "12345";
        try str.hashFromString(source, hash_def, result.bytes[0..hash_def.hash_length], allocator);
        result.has_value = true;
        return result;
    }

    if (ctx.hash == null or ctx.hash.?.len == 0) {
        return result;
    }

    if (ctx.is_base64) {
        t.parseSearchHash(ctx.hash.?, true, hash_def, &result.bytes) catch return error.InvalidArgument;
    } else {
        t.parseSearchHash(ctx.hash.?, false, hash_def, &result.bytes) catch return error.InvalidArgument;
    }
    result.has_value = true;
    return result;
}

pub fn bfCrackHash(
    params: CrackParams,
    target_hex: []const u8,
    hash_def: *const hashes.HashDefinition,
    ctx: *const t.HashCtx,
    env: t.RunEnv,
) !void {
    const threads: u32 = if (ctx.threads > 0) @intCast(ctx.threads) else 0;
    const result = try bf.crackHash(
        env.allocator,
        env.io,
        env.out,
        params.dictionary,
        target_hex,
        @intCast(@max(params.passmin, 0)),
        @intCast(@max(params.passmax, 0)),
        hash_def,
        ctx.no_probe,
        threads,
        hash_def.use_wide_string,
    );
    if (result) |password| {
        env.allocator.free(password);
    }
}

pub fn hashRun(
    ctx: *t.HashCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
) !void {
    const params = resolveCrackParams(ctx);
    const target = try resolveTargetHash(ctx, hash_def, env.allocator);
    if (!target.has_value) {
        return;
    }
    var hexbuf: [t.MAX_DIGEST_SIZE * 2]u8 = undefined;
    const hex = t.hashToHex(target.bytes[0..hash_def.hash_length], false, &hexbuf);
    try bfCrackHash(params, hex, hash_def, ctx, env);
}

test "resolveCrackParams applies defaults" {
    var ctx: t.HashCtx = .{ .builtin = &.{ .hash_algorithm = "tiger" } };
    var p = resolveCrackParams(&ctx);
    try std.testing.expectEqualStrings(bf.DEFAULT_ALPHABET, p.dictionary);
    try std.testing.expectEqual(@as(i32, 1), p.passmin);
    try std.testing.expectEqual(@as(i32, 10), p.passmax);

    ctx.min = 3;
    ctx.max = 7;
    ctx.dictionary = "abc";
    p = resolveCrackParams(&ctx);
    try std.testing.expectEqualStrings("abc", p.dictionary);
    try std.testing.expectEqual(@as(i32, 3), p.passmin);
    try std.testing.expectEqual(@as(i32, 7), p.passmax);
}

test "resolveTargetHash performance computes target" {
    var ctx: t.HashCtx = .{
        .builtin = &.{ .hash_algorithm = "tiger" },
        .performance = true,
        .hash = "12345",
    };
    const target = try resolveTargetHash(&ctx, hashes.getHash("tiger").?, std.testing.allocator);

    var expected: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    try str.hashFromString("12345", hashes.getHash("tiger").?, expected[0..24], std.testing.allocator);
    try std.testing.expect(target.has_value);
    try std.testing.expectEqualSlices(u8, expected[0..24], target.bytes[0..24]);
}

test "resolveTargetHash empty hash returns no value" {
    var ctx: t.HashCtx = .{ .builtin = &.{ .hash_algorithm = "tiger" } };
    const target = try resolveTargetHash(&ctx, hashes.getHash("tiger").?, std.testing.allocator);
    try std.testing.expect(!target.has_value);
}

test "hashRun recovers short tiger password" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = arena.allocator(),
        .out = &writer,
    };

    const tiger = hashes.getHash("tiger").?;
    var digest: [24]u8 align(8) = undefined;
    hashes.compute(tiger, "ab", &digest);
    var hexbuf: [48]u8 = undefined;
    const hex = t.hashToHex(&digest, false, &hexbuf);

    var ctx: t.HashCtx = .{
        .builtin = &.{ .hash_algorithm = "tiger" },
        .hash = hex,
        .dictionary = "ab",
        .min = 1,
        .max = 2,
        .no_probe = true,
        .threads = 1,
    };
    try hashRun(&ctx, env, tiger);
    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, out, "Initial string is: ab") != null);
}

test "bfCrackHash propagates writer failure not as OutOfMemory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // Tiny fixed buffer: crackHash fails on the first print with WriteFailed.
    var buf: [1]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = arena.allocator(),
        .out = &writer,
    };

    const tiger = hashes.getHash("tiger").?;
    var digest: [24]u8 align(8) = undefined;
    hashes.compute(tiger, "a", &digest);
    var hexbuf: [48]u8 = undefined;
    const hex = t.hashToHex(&digest, false, &hexbuf);

    const params: CrackParams = .{
        .dictionary = "a",
        .passmin = 1,
        .passmax = 1,
    };
    var ctx: t.HashCtx = .{
        .builtin = &.{ .hash_algorithm = "tiger" },
        .no_probe = true,
        .threads = 1,
    };
    const err = bfCrackHash(params, hex, tiger, &ctx, env);
    try std.testing.expectError(error.WriteFailed, err);
}
