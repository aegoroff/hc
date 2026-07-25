const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types.zig");

const builtin = @import("builtin.zig");
const file = @import("file.zig");

pub const DirCtx = t.DirCtx;
pub const RunEnv = t.RunEnv;
pub const RunError = t.RunError;

pub fn trimQuotes(str: []const u8) []const u8 {
    var s = str;
    if (s.len > 0 and (s[0] == '\'' or s[0] == '"')) s = s[1..];
    if (s.len > 0 and (s[s.len - 1] == '\'' or s[s.len - 1] == '"')) s = s[0 .. s.len - 1];
    return s;
}

pub fn nameMatches(
    name: []const u8,
    include: ?[]const u8,
    exclude: ?[]const u8,
) bool {
    if (exclude) |ex| {
        if (ex.len > 0 and std.mem.indexOf(u8, name, ex) != null) return false;
    }
    if (include) |inc| {
        if (inc.len > 0 and std.mem.indexOf(u8, name, inc) == null) return false;
    }
    return true;
}

fn buildFileCtx(template: *const DirCtx, builtin_ctx: *const t.BuiltinCtx, path: []const u8) t.FileCtx {
    return .{
        .builtin = builtin_ctx,
        .file_path = path,
        .limit = template.limit,
        .offset = template.offset,
        .hash = template.hash,
        .show_time = template.show_time,
        .result_in_sfv = template.result_in_sfv,
        .is_verify = template.is_verify,
        .is_base64 = template.is_base64,
    };
}

fn joinPath(allocator: std.mem.Allocator, dir: []const u8, name: []const u8) ![]const u8 {
    if (dir.len == 0) return allocator.dupe(u8, name);
    const need_sep = dir[dir.len - 1] != '/';
    if (need_sep) {
        return std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, name });
    }
    return std.fmt.allocPrint(allocator, "{s}{s}", .{ dir, name });
}

fn processFile(
    full_path: []const u8,
    template: *const DirCtx,
    builtin_ctx: *const t.BuiltinCtx,
    env: RunEnv,
    hash_def: *const hashes.HashDefinition,
    search_mode: bool,
) RunError!void {
    if (search_mode) {
        var fctx = buildFileCtx(template, builtin_ctx, full_path);
        fctx.hash = template.search_hash;
        const res = file.calculateFile(full_path, &fctx, env, hash_def) catch return;
        if (!(res.matches orelse false)) return;
        var size_buf: [64]u8 = undefined;
        var sw: std.Io.Writer = .fixed(&size_buf);
        lib.formatSize(res.file_size, &sw) catch return;
        const size_str = std.Io.Writer.buffered(&sw);
        try env.out.print("{s}{s}{s}\n", .{ full_path, t.FILE_INFO_COLUMN_SEPARATOR, size_str });
        return;
    }

    var fctx = buildFileCtx(template, builtin_ctx, full_path);
    try file.hashAndWriteFile(full_path, &fctx, env, hash_def);
}

pub fn dirRun(
    ctx: *DirCtx,
    env: RunEnv,
    hash_def: *const hashes.HashDefinition,
) RunError!void {
    if (!try builtin.allowSfvOption(ctx.result_in_sfv, hash_def, env.out)) {
        return;
    }

    const search_mode = ctx.search_hash != null;
    const path = trimQuotes(ctx.dir_path);
    const io = env.io;
    const allocator = env.allocator;

    var root = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch {
        try env.out.print("{s}: cannot open directory\n", .{path});
        return;
    };
    defer root.close(io);

    if (ctx.recursively) {
        var walker = root.walk(allocator) catch return error.OutOfMemory;
        defer walker.deinit();
        while (true) {
            const entry = walker.next(io) catch null orelse break;
            if (entry.kind != .file) continue;
            const full = joinPath(allocator, path, entry.path) catch return error.OutOfMemory;
            defer allocator.free(full);
            if (!nameMatches(entry.basename, ctx.include_pattern, ctx.exclude_pattern)) continue;
            processFile(full, ctx, ctx.builtin, env, hash_def, search_mode) catch |e| {
                if (e == error.OutOfMemory) return e;
            };
        }
    } else {
        var it = root.iterate();
        while (true) {
            const entry = it.next(io) catch null orelse break;
            if (entry.kind != .file) continue;
            const full = joinPath(allocator, path, entry.name) catch return error.OutOfMemory;
            defer allocator.free(full);
            if (!nameMatches(entry.name, ctx.include_pattern, ctx.exclude_pattern)) continue;
            processFile(full, ctx, ctx.builtin, env, hash_def, search_mode) catch |e| {
                if (e == error.OutOfMemory) return e;
            };
        }
    }
}

test "trimQuotes strips surrounding quotes" {
    try std.testing.expectEqualStrings("foo", trimQuotes("\"foo\""));
    try std.testing.expectEqualStrings("foo", trimQuotes("'foo'"));
    try std.testing.expectEqualStrings("foo", trimQuotes("foo"));
    try std.testing.expectEqualStrings("", trimQuotes("\"\""));
}

test "nameMatches substring include/exclude" {
    try std.testing.expect(nameMatches("readme.txt", "readme", null));
    try std.testing.expect(!nameMatches("data.bin", "readme", null));
    try std.testing.expect(!nameMatches("readme.txt", null, ".txt"));
    try std.testing.expect(nameMatches("data.bin", null, ".txt"));
    try std.testing.expect(nameMatches("a.txt", ".txt", ".bak"));
    try std.testing.expect(!nameMatches("a.bak", ".txt", ".bak"));
}

test "dirRun hashes files recursively" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};

    var d = std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true }) catch return;
    var f1 = try d.createFile(io, "x.txt", .{});
    try f1.writeStreamingAll(io, "xxx");
    f1.close(io);
    var f2 = try d.createFile(io, "y.txt", .{});
    try f2.writeStreamingAll(io, "yyyy");
    f2.close(io);
    d.close(io);

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var dctx: DirCtx = .{
        .builtin = &bctx,
        .dir_path = base,
        .recursively = true,
    };

    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(got.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, got, "x.txt") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "y.txt") != null);
}

test "dirRun include filter" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_filter_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};

    var d = std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true }) catch return;
    var f1 = try d.createFile(io, "keep.txt", .{});
    try f1.writeStreamingAll(io, "k");
    f1.close(io);
    var f2 = try d.createFile(io, "skip.log", .{});
    try f2.writeStreamingAll(io, "s");
    f2.close(io);
    d.close(io);

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var dctx: DirCtx = .{
        .builtin = &bctx,
        .dir_path = base,
        .recursively = true,
        .include_pattern = ".txt",
    };

    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, "keep.txt") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "skip.log") == null);
}

test "dirRun search hash lists only matching files" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_search_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};

    // match.txt holds "abc", nomatch.txt holds "xyz"
    var d = std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true }) catch return;
    var f1 = try d.createFile(io, "match.txt", .{});
    try f1.writeStreamingAll(io, "abc");
    f1.close(io);
    var f2 = try d.createFile(io, "nomatch.txt", .{});
    try f2.writeStreamingAll(io, "xyz");
    f2.close(io);
    d.close(io);

    var expected: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "abc", expected[0..24]);
    var exp_hex_buf: [64]u8 = undefined;
    const search_hex = t.hashToHex(expected[0..24], false, &exp_hex_buf);

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var dctx: DirCtx = .{
        .builtin = &bctx,
        .dir_path = base,
        .recursively = true,
        .search_hash = search_hex,
    };

    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, "match.txt") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "nomatch.txt") == null);
}
