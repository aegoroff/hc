const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types.zig");

pub const RunEnv = t.RunEnv;
pub const RunError = t.RunError;
pub const BuiltinCtx = t.BuiltinCtx;

pub fn builtinInit(bctx: *const BuiltinCtx, env: RunEnv) RunError!*const hashes.HashDefinition {
    const h = hashes.getHash(bctx.hash_algorithm) orelse {
        try env.out.print("Unknown hash: {s}", .{bctx.hash_algorithm});
        try lib.newLine(env.out);
        return error.UnknownHash;
    };
    return h;
}

pub fn allowSfvOption(
    result_in_sfv: bool,
    hash_def: *const hashes.HashDefinition,
    out: *std.Io.Writer,
) RunError!bool {
    if (result_in_sfv) {
        if (!std.ascii.eqlIgnoreCase(hash_def.name, "crc32") and
            !std.ascii.eqlIgnoreCase(hash_def.name, "crc32c"))
        {
            try out.print(
                "\n --sfv option doesn't support {s} algorithm. Only crc32 or crc32c supported",
                .{hash_def.name},
            );
            try lib.newLine(out);
            return false;
        }
    }
    return true;
}

test "builtinInit resolves known hash" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: BuiltinCtx = .{ .hash_algorithm = "tiger" };
    const h = try builtinInit(&bctx, env);
    try std.testing.expectEqualStrings("tiger", h.name);
}

test "builtinInit rejects unknown hash" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: BuiltinCtx = .{ .hash_algorithm = "nope" };
    try std.testing.expectError(error.UnknownHash, builtinInit(&bctx, env));
    try std.testing.expectEqualStrings("Unknown hash: nope\n", std.Io.Writer.buffered(&writer));
}

const str = @import("str.zig");

test "builtinInit then strRun prints digest" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: BuiltinCtx = .{ .hash_algorithm = "tiger", .is_print_low_case = true };
    var sctx: str.StringCtx = .{ .builtin = &bctx, .string = "" };

    const h = try builtinInit(&bctx, env);
    try str.strRun(&sctx, env, h);

    try std.testing.expectEqualStrings(
        "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3\n",
        std.Io.Writer.buffered(&writer),
    );
}

test "allowSfvOption rejects non-crc with trailing newline" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const h = hashes.getHash("md5").?;

    try std.testing.expect(!try allowSfvOption(true, h, &writer));
    try std.testing.expectEqualStrings(
        "\n --sfv option doesn't support md5 algorithm. Only crc32 or crc32c supported\n",
        std.Io.Writer.buffered(&writer),
    );
}

test "allowSfvOption allows crc32" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const h = hashes.getHash("crc32").?;

    try std.testing.expect(try allowSfvOption(true, h, &writer));
    try std.testing.expectEqual(@as(usize, 0), std.Io.Writer.buffered(&writer).len);
}
