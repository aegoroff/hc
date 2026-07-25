const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types.zig");

const str = @import("str.zig");

pub const HashCtx = t.HashCtx;
pub const RunEnv = t.RunEnv;
pub const RunError = t.RunError;

pub const MIN_DEFAULT: i32 = 1;
pub const MAX_DEFAULT: i32 = 10;

pub const DIGITS = "0123456789";
pub const LOW_CASE = "abcdefghijklmnopqrstuvwxyz";
pub const UPPER_CASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

pub const default_alphabet = DIGITS ++ LOW_CASE ++ UPPER_CASE;

pub const CrackParams = struct {
    dictionary: []const u8,
    passmin: i32,
    passmax: i32,
};

pub fn resolveCrackParams(ctx: *const HashCtx) CrackParams {
    return .{
        .dictionary = ctx.dictionary orelse default_alphabet,
        .passmin = if (ctx.min > 0) ctx.min else MIN_DEFAULT,
        .passmax = if (ctx.max > 0) ctx.max else MAX_DEFAULT,
    };
}

pub const TargetHash = struct {
    bytes: [t.MAX_DIGEST_SIZE]u8 align(8),
    has_value: bool,
};

pub fn resolveTargetHash(
    ctx: *const HashCtx,
    hash_def: *const hashes.HashDefinition,
    allocator: std.mem.Allocator,
) RunError!TargetHash {
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
    target: []const u8,
    hash_def: *const hashes.HashDefinition,
    env: RunEnv,
) RunError!void {
    _ = params;
    _ = target;
    _ = hash_def;
    _ = env;
    // TODO: brute-force restore engine (port of src/srclib/bf.c) is not yet
    // available on the Zig side. Until bf.c is ported, hash restore stays a
    // placeholder.
    return error.NotImplemented;
}

pub fn hashRun(
    ctx: *HashCtx,
    env: RunEnv,
    hash_def: *const hashes.HashDefinition,
) RunError!void {
    const params = resolveCrackParams(ctx);
    const target = try resolveTargetHash(ctx, hash_def, env.allocator);
    if (!target.has_value) {
        return;
    }
    try bfCrackHash(params, target.bytes[0..hash_def.hash_length], hash_def, env);
}

test "resolveCrackParams applies defaults" {
    var ctx: HashCtx = .{ .builtin = &.{ .hash_algorithm = "tiger" } };
    var p = resolveCrackParams(&ctx);
    try std.testing.expectEqualStrings(default_alphabet, p.dictionary);
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
    var ctx: HashCtx = .{
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
    var ctx: HashCtx = .{ .builtin = &.{ .hash_algorithm = "tiger" } };
    const target = try resolveTargetHash(&ctx, hashes.getHash("tiger").?, std.testing.allocator);
    try std.testing.expect(!target.has_value);
}

test "hashRun returns NotImplemented until bf is ported" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var ctx: HashCtx = .{
        .builtin = &.{ .hash_algorithm = "tiger" },
        .hash = "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3",
    };
    try std.testing.expectError(
        error.NotImplemented,
        hashRun(&ctx, env, hashes.getHash("tiger").?),
    );
}
