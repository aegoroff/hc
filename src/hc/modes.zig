const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");

pub const types = @import("modes/types.zig");
pub const str = @import("modes/str.zig");
const hash = @import("modes/hash.zig");
pub const file = @import("modes/file.zig");
pub const dir = @import("modes/dir.zig");
const save = @import("modes/save.zig");

pub const StringCtx = types.StringCtx;
pub const HashCtx = types.HashCtx;
pub const FileOptions = types.FileOptions;
pub const FileCtx = types.FileCtx;
pub const DirCtx = types.DirCtx;
pub const RunEnv = types.RunEnv;

/// Resolve algorithm name via `hashes.getHash`; prints and returns UnknownHash if missing.
pub fn resolveHash(name: []const u8, env: RunEnv) types.RunError!*const hashes.HashDefinition {
    return hashes.getHash(name) orelse {
        try env.out.print("Unknown hash: {s}", .{name});
        try lib.newLine(env.out);
        return error.UnknownHash;
    };
}

pub const strRun = str.strRun;
pub const hashRun = hash.hashRun;
pub const fileRun = file.fileRun;
pub const defaultAlphabet = @import("bf").DEFAULT_ALPHABET;
pub const dirRun = dir.dirRun;

comptime {
    _ = types;
    _ = str;
    _ = hash;
    _ = file;
    _ = dir;
    _ = save;
}

test "resolveHash resolves known hash" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const h = try resolveHash("tiger", env);
    try std.testing.expectEqualStrings("tiger", h.name);
}

test "resolveHash rejects unknown hash" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    try std.testing.expectError(error.UnknownHash, resolveHash("nope", env));
    try std.testing.expectEqualStrings("Unknown hash: nope\n", std.Io.Writer.buffered(&writer));
}

test "resolveHash then strRun prints digest" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var sctx: StringCtx = .{ .string = "", .low_case = true };

    const h = try resolveHash("tiger", env);
    try strRun(&sctx, env, h);

    try std.testing.expectEqualStrings(
        "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3\n",
        std.Io.Writer.buffered(&writer),
    );
}
