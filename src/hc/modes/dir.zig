const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types.zig");

const builtin = @import("builtin.zig");
const file = @import("file.zig");
const save = @import("save.zig");

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
    if (include) |inc| {
        if (inc.len > 0 and !anySubPatternMatches(name, inc)) return false;
    }
    if (exclude) |ex| {
        if (ex.len > 0 and anySubPatternMatches(name, ex)) return false;
    }
    return true;
}

/// Case-blind equality for a single byte (mirrors APR_FNM_CASE_BLIND).
fn charEqIgnoreCase(a: u8, b: u8) bool {
    return std.ascii.toLower(a) == std.ascii.toLower(b);
}

/// Glob full-string match mirroring `apr_fnmatch` with `APR_FNM_CASE_BLIND`.
/// Supports `*` (any run), `?` (any single byte) and case-blind literals.
/// Character classes (`[...]`) are not implemented (no test exercises them).
pub fn globMatch(pattern: []const u8, name: []const u8) bool {
    var pi: usize = 0;
    var ni: usize = 0;
    var has_star = false;
    var star_pi: usize = 0;
    var star_ni: usize = 0;

    while (ni < name.len) {
        if (pi < pattern.len) {
            switch (pattern[pi]) {
                '*' => {
                    has_star = true;
                    star_pi = pi;
                    star_ni = ni;
                    pi += 1;
                    continue;
                },
                '?' => {
                    pi += 1;
                    ni += 1;
                    continue;
                },
                else => |pc| {
                    if (charEqIgnoreCase(pc, name[ni])) {
                        pi += 1;
                        ni += 1;
                        continue;
                    }
                },
            }
        }
        if (has_star) {
            pi = star_pi + 1;
            star_ni += 1;
            ni = star_ni;
            continue;
        }
        return false;
    }
    while (pi < pattern.len and pattern[pi] == '*') pi += 1;
    return pi == pattern.len;
}

/// True if any `;`-separated sub-pattern of `pattern` glob-matches `name`
/// (mirrors traverse_match_to_composite_pattern).
fn anySubPatternMatches(name: []const u8, pattern: []const u8) bool {
    var it = std.mem.splitScalar(u8, pattern, ';');
    while (it.next()) |sub| {
        if (globMatch(sub, name)) return true;
    }
    return false;
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
    // Use the platform-native separator so Windows dir output uses '\' (matching
    // the CMake/msbuild binary and the C# test expectations) and POSIX uses '/'.
    // Treat both '/' and '\' as an existing trailing separator on either OS.
    const last = dir[dir.len - 1];
    const need_sep = last != '/' and last != '\\';
    if (need_sep) {
        return std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ dir, std.fs.path.sep_str, name });
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
        // Effective search target: an explicit --search hash, otherwise the -m
        // digest (C's dir.c defaults hash_to_search_ to ctx->hash_).
        fctx.hash = template.search_hash orelse template.hash;
        const res = file.calculateFile(full_path, &fctx, env, hash_def) catch return;
        if (!(res.matches orelse false)) return;
        var size_buf: [64]u8 = undefined;
        var sw: std.Io.Writer = .fixed(&size_buf);
        lib.formatSize(res.file_size, &sw) catch return;
        const size_str = std.Io.Writer.buffered(&sw);
        try env.out.print("{s}{s}{s}\n", .{ full_path, t.FILE_INFO_COLUMN_SEPARATOR, size_str });
        try env.out.flush();
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

    // Search mode when an explicit --search hash OR a -m digest is present and
    // we are not in checksum-verify (-c) mode. Mirrors C dir.c, where -m without
    // -c runs in search mode (only the matching file is emitted with its size).
    const search_mode = (ctx.search_hash != null or ctx.hash != null) and !ctx.is_verify;
    const path = trimQuotes(ctx.dir_path);
    const io = env.io;
    const allocator = env.allocator;

    // When -o <save> is given, C dir.c tees every result line to BOTH the
    // console and the save file (shared SaveTee helper with file mode).
    var tee = save.SaveTee.init(allocator, ctx.save_result_path);
    defer tee.deinit();
    const sink_env = tee.sinkEnv(env);

    var root = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch {
        try sink_env.out.print("{s}: cannot open directory\n", .{path});
        try tee.flush(env.out);
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
            processFile(full, ctx, ctx.builtin, sink_env, hash_def, search_mode) catch |e| {
                if (e == error.OutOfMemory) return e;
            };
            try tee.flush(env.out);
        }
    } else {
        var it = root.iterate();
        while (true) {
            const entry = it.next(io) catch null orelse break;
            if (entry.kind != .file) continue;
            const full = joinPath(allocator, path, entry.name) catch return error.OutOfMemory;
            defer allocator.free(full);
            if (!nameMatches(entry.name, ctx.include_pattern, ctx.exclude_pattern)) continue;
            processFile(full, ctx, ctx.builtin, sink_env, hash_def, search_mode) catch |e| {
                if (e == error.OutOfMemory) return e;
            };
            try tee.flush(env.out);
        }
    }

    tee.finish(env);
}

test "trimQuotes strips surrounding quotes" {
    try std.testing.expectEqualStrings("foo", trimQuotes("\"foo\""));
    try std.testing.expectEqualStrings("foo", trimQuotes("'foo'"));
    try std.testing.expectEqualStrings("foo", trimQuotes("foo"));
    try std.testing.expectEqualStrings("", trimQuotes("\"\""));
}

test "nameMatches glob include/exclude" {
    try std.testing.expect(nameMatches("readme.txt", "readme*", null));
    try std.testing.expect(!nameMatches("data.bin", "readme*", null));
    try std.testing.expect(!nameMatches("readme.txt", null, "*.txt"));
    try std.testing.expect(nameMatches("data.bin", null, "*.txt"));
    try std.testing.expect(nameMatches("a.txt", "*.txt", "*.bak"));
    try std.testing.expect(!nameMatches("a.bak", "*.txt", "*.bak"));
}

test "nameMatches literal full match (not substring)" {
    // "empty" must match "empty" but not "notempty" (apr_fnmatch semantics).
    try std.testing.expect(nameMatches("empty", "empty", null));
    try std.testing.expect(!nameMatches("notempty", "empty", null));
    try std.testing.expect(nameMatches("notempty", null, "empty"));
}

test "nameMatches composite pattern separated by ;" {
    try std.testing.expect(nameMatches("notempty", "empty;notempty", null));
    try std.testing.expect(nameMatches("empty", "empty;notempty", null));
    try std.testing.expect(!nameMatches("other", "empty;notempty", null));
    try std.testing.expect(!nameMatches("notempty", null, "empty;notempty"));
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
        .include_pattern = "*.txt",
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
