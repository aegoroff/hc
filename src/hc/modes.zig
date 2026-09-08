const std = @import("std");
const hashes = @import("hashes");

pub const types = @import("modes/types.zig");
pub const str = @import("modes/str.zig");
pub const hash = @import("modes/hash.zig");
pub const file = @import("modes/file.zig");
pub const dir = @import("modes/dir.zig");

pub const StringCtx = types.StringCtx;
pub const HashCtx = types.HashCtx;
pub const FileOptions = types.FileOptions;
pub const FileCtx = types.FileCtx;
pub const DirCtx = types.DirCtx;
pub const RunEnv = types.RunEnv;

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
}

test "strRun prints digest" {
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var sctx: StringCtx = .{ .string = "", .low_case = true };
    const h = hashes.getHash("tiger").?;

    // Act
    try strRun(&sctx, env, h);

    // Assert
    try std.testing.expectEqualStrings(
        "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3\n",
        std.Io.Writer.buffered(&writer),
    );
}
