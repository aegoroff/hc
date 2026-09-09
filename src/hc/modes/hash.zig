const std = @import("std");
const hashes = @import("hashes");
const bf = @import("bf");
const t = @import("types.zig");

const MIN_DEFAULT: i32 = 1;

/// Crack knobs for `restore`. `min`/`max` of 0 mean `hc hash` defaults (1 and 10).
pub const RestoreOpts = struct {
    dictionary: ?[]const u8 = null,
    min: i32 = 0,
    max: i32 = 0,
    no_probe: bool = false,
    threads: u32 = 0,
};

/// Crack `digest` (raw hash bytes, length = `hash_def.hash_length`).
/// Caller owns a non-null result. Probe, timings, and the result line go to `out`.
pub fn restore(
    hash_def: *const hashes.HashDefinition,
    digest: []const u8,
    opts: RestoreOpts,
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
) !?[]u8 {
    const dictionary = opts.dictionary orelse bf.DEFAULT_ALPHABET;
    const passmin: i32 = if (opts.min > 0) opts.min else MIN_DEFAULT;
    const passmax: i32 = if (opts.max > 0) opts.max else @intCast(bf.MAX_DEFAULT);
    if (passmin > passmax) return error.InvalidRange;

    return bf.crackHash(
        gpa,
        io,
        out,
        dictionary,
        digest,
        @intCast(@max(passmin, 0)),
        @intCast(@max(passmax, 0)),
        hash_def,
        opts.no_probe,
        opts.threads,
        hash_def.use_wide_string,
    );
}

pub fn hashRun(
    ctx: *t.HashCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
) !void {
    const passmin: i32 = if (ctx.min > 0) ctx.min else MIN_DEFAULT;
    const passmax: i32 = if (ctx.max > 0) ctx.max else @intCast(bf.MAX_DEFAULT);
    if (passmin > passmax) {
        try env.out.print("Minimum password length {d} is greater than maximum {d}\n", .{ passmin, passmax });
        return error.InvalidArgument;
    }

    var target: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    var has_target = false;
    if (ctx.performance) {
        const source: []const u8 = if (ctx.hash != null and ctx.hash.?.len > 0) ctx.hash.? else "12345";
        hashes.createStringDigest(hash_def, source, target[0..hash_def.hash_length], env.allocator) catch |err| return switch (err) {
            error.InvalidUtf8 => error.InvalidArgument,
            error.OutOfMemory => error.OutOfMemory,
        };
        has_target = true;
    } else if (ctx.hash != null and ctx.hash.?.len > 0) {
        t.parseSearchHash(ctx.hash.?, ctx.is_base64, hash_def, &target) catch {
            // main maps InvalidArgument to a silent exit 1: print the reason.
            try env.out.print("invalid search hash: {s}\n", .{ctx.hash.?});
            return error.InvalidArgument;
        };
        has_target = true;
    }
    if (!has_target) return;

    const threads: u32 = if (ctx.threads > 0) @intCast(ctx.threads) else 0;
    const result = try restore(hash_def, target[0..hash_def.hash_length], .{
        .dictionary = ctx.dictionary,
        .min = ctx.min,
        .max = ctx.max,
        .no_probe = ctx.no_probe,
        .threads = threads,
    }, env.allocator, env.io, env.out);
    if (result) |password| {
        env.allocator.free(password);
    }
}

test "hashRun recovers short tiger password" {
    // Arrange
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
        .hash = hex,
        .dictionary = "ab",
        .min = 1,
        .max = 2,
        .no_probe = true,
        .threads = 1,
    };

    // Act
    try hashRun(&ctx, env, tiger);

    // Assert
    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, out, "Initial string is: ab") != null);
}

test "hashRun applies default dictionary and bounds" {
    // Arrange
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
    hashes.compute(tiger, "z", &digest);
    var hexbuf: [48]u8 = undefined;
    const hex = t.hashToHex(&digest, false, &hexbuf);

    // No dictionary/min/max: defaults must kick in (DEFAULT_ALPHABET, 1..10).
    var ctx: t.HashCtx = .{
        .hash = hex,
        .no_probe = true,
        .threads = 1,
    };

    // Act
    try hashRun(&ctx, env, tiger);

    // Assert
    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, out, "Initial string is: z") != null);
}

test "hashRun performance mode cracks the performance source" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = arena.allocator(),
        .out = &writer,
    };

    var ctx: t.HashCtx = .{
        .performance = true,
        .hash = "12345",
        .dictionary = "12345",
        .min = 5,
        .max = 5,
        .no_probe = true,
        .threads = 1,
    };

    // Act
    try hashRun(&ctx, env, hashes.getHash("tiger").?);

    // Assert
    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, out, "Initial string is: 12345") != null);
}

test "hashRun without hash writes nothing" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = arena.allocator(),
        .out = &writer,
    };

    var ctx: t.HashCtx = .{};

    // Act
    try hashRun(&ctx, env, hashes.getHash("tiger").?);

    // Assert
    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expectEqual(@as(usize, 0), out.len);
}

test "hashRun min greater than max reports and aborts" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = arena.allocator(),
        .out = &writer,
    };

    var ctx: t.HashCtx = .{
        .min = 5,
        .max = 2,
        .no_probe = true,
        .threads = 1,
    };

    // Act
    const err = hashRun(&ctx, env, hashes.getHash("tiger").?);

    // Assert
    try std.testing.expectError(error.InvalidArgument, err);
    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, out, "Minimum password length 5 is greater than maximum 2") != null);
}

test "hashRun invalid search hash reports and aborts" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = arena.allocator(),
        .out = &writer,
    };

    var ctx: t.HashCtx = .{
        .hash = "ZZZZ",
        .no_probe = true,
        .threads = 1,
    };

    // Act
    const err = hashRun(&ctx, env, hashes.getHash("tiger").?);

    // Assert
    try std.testing.expectError(error.InvalidArgument, err);
    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, out, "invalid search hash: ZZZZ") != null);
}

test "hashRun oversized passmax reports and aborts" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var buf: [256]u8 = undefined;
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

    var ctx: t.HashCtx = .{
        .hash = hex,
        .dictionary = "a",
        .min = 1,
        .max = 600000000,
        .no_probe = true,
        .threads = 1,
    };

    // Act
    const err = hashRun(&ctx, env, tiger);

    // Assert
    try std.testing.expectError(error.PassmaxTooBig, err);
    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, out, "Max string length is too big: 600000000") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "Nothing found") == null);
}

test "hashRun propagates writer failure not as OutOfMemory" {
    // Arrange
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

    var ctx: t.HashCtx = .{
        .hash = hex,
        .dictionary = "a",
        .min = 1,
        .max = 1,
        .no_probe = true,
        .threads = 1,
    };

    // Act
    const err = hashRun(&ctx, env, tiger);

    // Assert
    try std.testing.expectError(error.WriteFailed, err);
}

test "restore returns the cracked password" {
    // Arrange
    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiger = hashes.getHash("tiger").?;
    var digest: [24]u8 align(8) = undefined;
    hashes.compute(tiger, "ab", &digest);

    // Act
    const got = try restore(tiger, &digest, .{
        .dictionary = "ab",
        .min = 1,
        .max = 2,
        .no_probe = true,
        .threads = 1,
    }, std.testing.allocator, io, &writer);
    defer if (got) |p| std.testing.allocator.free(p);

    // Assert
    try std.testing.expectEqualStrings("ab", got.?);
}

test "restore miss returns null" {
    // Arrange
    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiger = hashes.getHash("tiger").?;
    var digest: [24]u8 align(8) = undefined;
    hashes.compute(tiger, "z", &digest);

    // Act
    const got = try restore(tiger, &digest, .{
        .dictionary = "ab",
        .min = 1,
        .max = 2,
        .no_probe = true,
        .threads = 1,
    }, std.testing.allocator, io, &writer);

    // Assert
    try std.testing.expect(got == null);
}

test "restore min greater than max is InvalidRange" {
    // Arrange
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiger = hashes.getHash("tiger").?;
    var digest: [24]u8 align(8) = undefined;
    hashes.compute(tiger, "a", &digest);

    // Act
    try std.testing.expectError(error.InvalidRange, restore(tiger, &digest, .{
        .min = 5,
        .max = 2,
        .no_probe = true,
        .threads = 1,
    }, std.testing.allocator, io, &writer));

    // Assert
    try std.testing.expectEqual(@as(usize, 0), std.Io.Writer.buffered(&writer).len);
}
