const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types.zig");

const builtin = @import("builtin.zig");
const file = @import("file.zig");
const save = @import("save.zig");

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

/// Case-blind equality for a single byte.
fn charEqIgnoreCase(a: u8, b: u8) bool {
    return std.ascii.toLower(a) == std.ascii.toLower(b);
}

/// Glob full-string match: `*` (any run), `?` (any single byte), case-blind
/// literals. Character classes (`[...]`) are not implemented (no test exercises them).
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

/// True if any `;`-separated sub-pattern of `pattern` glob-matches `name`.
fn anySubPatternMatches(name: []const u8, pattern: []const u8) bool {
    var it = std.mem.splitScalar(u8, pattern, ';');
    while (it.next()) |sub| {
        if (globMatch(sub, name)) return true;
    }
    return false;
}

fn buildFileCtx(template: *const t.DirCtx, path: []const u8) t.FileCtx {
    return .{
        .opts = template.opts,
        .file_path = path,
    };
}

/// Effective entry kind for walk decisions. `DT_UNKNOWN` filesystems (XFS
/// ftype=0, some FUSE) give no d_type, which would both hide regular files
/// and block recursion into subdirectories; stat those entries instead.
/// No-follow keeps symlinks skipped exactly as on filesystems that do fill
/// d_type. Entries that cannot be stat'ed (vanished mid-walk) stay unknown.
/// Windows is comptime-excluded: its enumeration maps attributes totally
/// (reparse/directory/file — `.unknown` is unreachable), and `statFile`
/// there cannot open directories anyway.
pub fn effectiveEntryKind(
    dir: std.Io.Dir,
    io: std.Io,
    name: []const u8,
    kind: std.Io.File.Kind,
) std.Io.File.Kind {
    if (comptime @import("builtin").os.tag == .windows) return kind;
    if (kind != .unknown) return kind;
    const st = dir.statFile(io, name, .{ .follow_symlinks = false }) catch return .unknown;
    return st.kind;
}

/// Walk/iterate errors skip the bad entry. OOM still aborts;
/// `--noerroronfind` suppresses the diagnostic line.
fn reportFindError(ctx: *const t.DirCtx, env: t.RunEnv, path_hint: []const u8, err: anyerror) t.RunError!void {
    if (err == error.OutOfMemory) return error.OutOfMemory;
    if (ctx.no_error_on_find) return;
    try env.out.print("{s}: {s}\n", .{ path_hint, @errorName(err) });
}

fn processFile(
    full_path: []const u8,
    template: *const t.DirCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
    search_mode: bool,
) t.RunError!void {
    if (search_mode) {
        var fctx = buildFileCtx(template, full_path);
        // Effective search target: an explicit --search hash, otherwise the -m
        // digest.
        fctx.opts.hash = template.search_hash orelse template.opts.hash;
        // Search-mode: abort on OOM, skip the entry for other calculate failures.
        const res = file.calculateFile(full_path, &fctx, env, hash_def) catch |e| {
            if (e == error.OutOfMemory) return e;
            return;
        };
        if (!(res.matches orelse false)) return;
        var size_buf: [64]u8 = undefined;
        var sw: std.Io.Writer = .fixed(&size_buf);
        lib.formatSize(res.file_size, &sw) catch return;
        const size_str = std.Io.Writer.buffered(&sw);
        try env.out.print("{s}{s}{s}\n", .{ full_path, t.FILE_INFO_COLUMN_SEPARATOR, size_str });
        try env.out.flush();
        return;
    }

    var fctx = buildFileCtx(template, full_path);
    try file.hashAndWriteFile(full_path, &fctx, env, hash_def);
}

pub fn dirRun(
    ctx: *t.DirCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
) t.RunError!void {
    if (!try builtin.allowSfvOption(ctx.opts.result_in_sfv, hash_def, env.out)) {
        return;
    }

    // Search mode when an explicit --search hash OR a -m digest is present and
    // we are not in checksum-verify (-c) mode: only the matching file is
    // emitted with its size. An empty target is ignored (calculateFile compares
    // only non-empty hashes; an empty -m must not suppress all output).
    const search_target = ctx.search_hash orelse ctx.opts.hash;
    const search_mode = search_target != null and search_target.?.len > 0 and !ctx.opts.is_verify;
    const path = lib.trimQuotes(ctx.dir_path);
    const io = env.io;
    const allocator = env.allocator;

    // When -o <save> is given, tee every result line to both the console and
    // the save file (shared SaveTee helper with file mode).
    // defer finish before deinit so early returns (e.g. openDir failure) still
    // persist the capture.
    var tee = save.SaveTee.init(allocator, ctx.opts.save_result_path);
    defer tee.deinit();
    defer tee.finish(env);
    const sink_env = tee.sinkEnv(env);

    // An invalid -m/--search hash fails for every file alike; validate once up
    // front instead of hashing the tree just to emit nothing (file mode reports
    // the same condition). Base64 stays off: file/dir -b is output-only.
    if (search_mode) {
        var probe: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
        t.parseSearchHash(search_target.?, false, hash_def, &probe) catch {
            try sink_env.out.print("invalid search hash: {s}\n", .{search_target.?});
            try tee.flush(env.out);
            return;
        };
    }

    var root = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch {
        // `--noerroronfind` suppresses "cannot open directory".
        if (!ctx.no_error_on_find) {
            try sink_env.out.print("{s}: cannot open directory\n", .{path});
            try tee.flush(env.out);
        }
        return;
    };
    defer root.close(io);

    if (ctx.recursively) {
        // Selective walker: enter() failures keep the entry path for diagnostics
        // and leave siblings reachable (unlike catch-null break on Walker.next).
        var walker = root.walkSelectively(allocator) catch return error.OutOfMemory;
        defer walker.deinit();
        while (true) {
            const maybe_entry = walker.next(io) catch |err| {
                try reportFindError(ctx, sink_env, path, err);
                try tee.flush(env.out);
                continue;
            };
            const entry = maybe_entry orelse break;
            const kind = effectiveEntryKind(entry.dir, io, entry.basename, entry.kind);
            if (kind == .directory) {
                walker.enter(io, entry) catch |err| {
                    const full = try std.fs.path.join(allocator, &.{ path, entry.path });
                    defer allocator.free(full);
                    try reportFindError(ctx, sink_env, full, err);
                    try tee.flush(env.out);
                    continue;
                };
                continue;
            }
            if (kind != .file) continue;
            const full = try std.fs.path.join(allocator, &.{ path, entry.path });
            defer allocator.free(full);
            if (!nameMatches(entry.basename, ctx.include_pattern, ctx.exclude_pattern)) continue;
            processFile(full, ctx, sink_env, hash_def, search_mode) catch |e| {
                if (e == error.OutOfMemory) return e;
            };
            try tee.flush(env.out);
        }
    } else {
        var it = root.iterate();
        while (true) {
            const maybe_entry = it.next(io) catch |err| {
                try reportFindError(ctx, sink_env, path, err);
                try tee.flush(env.out);
                continue;
            };
            const entry = maybe_entry orelse break;
            if (effectiveEntryKind(root, io, entry.name, entry.kind) != .file) continue;
            const full = try std.fs.path.join(allocator, &.{ path, entry.name });
            defer allocator.free(full);
            if (!nameMatches(entry.name, ctx.include_pattern, ctx.exclude_pattern)) continue;
            processFile(full, ctx, sink_env, hash_def, search_mode) catch |e| {
                if (e == error.OutOfMemory) return e;
            };
            try tee.flush(env.out);
        }
    }
}

test "effectiveEntryKind resolves unknown entries via no-follow stat" {
    // POSIX only: the stat fallback is comptime-disabled on Windows (its
    // enumeration never yields .unknown, and statFile there cannot open
    // directories).
    if (comptime @import("builtin").os.tag == .windows) return;

    // Arrange — walk decisions get a d_type kind; on DT_UNKNOWN filesystems
    // (XFS ftype=0, some FUSE) it is .unknown and must resolve by stat without
    // following symlinks, keeping the skip-symlinks rule uniform everywhere.
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_kind_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};

    var d = std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true }) catch return;
    defer d.close(io);
    var f1 = try d.createFile(io, "a.txt", .{});
    try f1.writeStreamingAll(io, "aaa");
    f1.close(io);
    try d.createDir(io, "sub", perms);
    d.symLink(io, "a.txt", "link.txt", .{}) catch return error.SkipZigTest;

    // Act
    const file_kind = effectiveEntryKind(d, io, "a.txt", .unknown);
    const dir_kind = effectiveEntryKind(d, io, "sub", .unknown);
    const sym_kind = effectiveEntryKind(d, io, "link.txt", .unknown);
    const passthrough = effectiveEntryKind(d, io, "a.txt", .file);
    const missing = effectiveEntryKind(d, io, "nope", .unknown);

    // Assert
    try std.testing.expectEqual(std.Io.File.Kind.file, file_kind);
    try std.testing.expectEqual(std.Io.File.Kind.directory, dir_kind);
    try std.testing.expectEqual(std.Io.File.Kind.sym_link, sym_kind);
    try std.testing.expectEqual(std.Io.File.Kind.file, passthrough);
    try std.testing.expectEqual(std.Io.File.Kind.unknown, missing);
}

test "nameMatches glob include/exclude" {
    // Act + Assert — include and exclude globs against known names
    try std.testing.expect(nameMatches("readme.txt", "readme*", null));
    try std.testing.expect(!nameMatches("data.bin", "readme*", null));
    try std.testing.expect(!nameMatches("readme.txt", null, "*.txt"));
    try std.testing.expect(nameMatches("data.bin", null, "*.txt"));
    try std.testing.expect(nameMatches("a.txt", "*.txt", "*.bak"));
    try std.testing.expect(!nameMatches("a.bak", "*.txt", "*.bak"));
}

test "nameMatches literal full match (not substring)" {
    // Act + Assert — "empty" must match "empty" but not "notempty"
    try std.testing.expect(nameMatches("empty", "empty", null));
    try std.testing.expect(!nameMatches("notempty", "empty", null));
    try std.testing.expect(nameMatches("notempty", null, "empty"));
}

test "nameMatches composite pattern separated by ;" {
    // Act + Assert — each `;`-separated sub-pattern is a full match candidate
    try std.testing.expect(nameMatches("notempty", "empty;notempty", null));
    try std.testing.expect(nameMatches("empty", "empty;notempty", null));
    try std.testing.expect(!nameMatches("other", "empty;notempty", null));
    try std.testing.expect(!nameMatches("notempty", null, "empty;notempty"));
}

test "dirRun hashes files recursively" {
    // Arrange
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

    var x_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "xxx", x_digest[0..24]);
    var x_buf: [64]u8 = undefined;
    const x_hex = t.hashToHex(x_digest[0..24], false, &x_buf);
    var y_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "yyyy", y_digest[0..24]);
    var y_buf: [64]u8 = undefined;
    const y_hex = t.hashToHex(y_digest[0..24], false, &y_buf);

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = base,
        .recursively = true,
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    // Walk order is readdir-dependent: assert each full line exactly instead
    // of one combined string. `join` uses the native separator.
    const got = std.Io.Writer.buffered(&writer);
    var want_buf: [2][256]u8 = undefined;
    const lines = [_][]const u8{
        try std.fmt.bufPrint(&want_buf[0], "{s}{s}x.txt{s}3 bytes{s}{s}\n", .{ base, std.fs.path.sep_str, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, x_hex }),
        try std.fmt.bufPrint(&want_buf[1], "{s}{s}y.txt{s}4 bytes{s}{s}\n", .{ base, std.fs.path.sep_str, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, y_hex }),
    };

    // Assert
    for (lines) |want| try std.testing.expect(std.mem.indexOf(u8, got, want) != null);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, got, "\n"));
}

test "dirRun include filter" {
    // Arrange
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

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "k", expected_digest[0..24]);
    var exp_buf: [64]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..24], false, &exp_buf);

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = base,
        .recursively = true,
        .include_pattern = "*.txt",
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}keep.txt{s}1 bytes{s}{s}\n", .{ base, std.fs.path.sep_str, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, exp_hex }),
        got,
    );
}

test "dirRun empty directory emits nothing" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_empty_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = base,
        .recursively = true,
    };

    // Act

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    // Assert
    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expectEqualStrings("", got);
}

test "dirRun search hash lists only matching files" {
    // Arrange
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
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = base,
        .recursively = true,
        .search_hash = search_hex,
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(std.mem.indexOf(u8, got, "match.txt") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "nomatch.txt") == null);
}

test "dirRun invalid -m search hash reports once and skips the walk" {
    // Arrange — a bad -m digest is a query-level error: report it up front
    // (file mode prints "invalid search hash" too) instead of hashing the
    // tree only to suppress every line.
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_bad_hash_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};

    var d = std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true }) catch return;
    var f1 = try d.createFile(io, "a.txt", .{});
    try f1.writeStreamingAll(io, "aaa");
    f1.close(io);
    d.close(io);

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{ .hash = "ZZZZ" },
        .dir_path = base,
        .recursively = true,
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    // Assert
    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, "invalid search hash: ZZZZ") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "a.txt") == null);
}

test "dirRun invalid --search hash reports once and skips the walk" {
    // Arrange — same validation for the explicit --search target.
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_bad_search_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = base,
        .recursively = true,
        .search_hash = "NOTHEX",
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    // Assert
    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, "invalid search hash: NOTHEX") != null);
}

test "dirRun empty search hash falls back to normal hashing" {
    // Arrange — calculateFile compares only non-empty hashes; an empty
    // --search must not switch into match-only mode and blank the output.
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_empty_search_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};

    var d = std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true }) catch return;
    var f1 = try d.createFile(io, "a.txt", .{});
    try f1.writeStreamingAll(io, "aaa");
    f1.close(io);
    d.close(io);

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = base,
        .recursively = true,
        .search_hash = "",
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    // Assert
    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, "a.txt") != null);
}

test "dirRun continues after unreadable subdirectory" {
    // Arrange — POSIX only: mode 0 directories reproduce AccessDenied on enter.
    if (comptime @import("builtin").os.tag == .windows) return;

    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_denied_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer restoreModeAndDeleteTree(io, base, "denied");

    var d = try std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true });
    var f1 = try d.createFile(io, "a.txt", .{});
    try f1.writeStreamingAll(io, "aaa");
    f1.close(io);
    try d.createDir(io, "denied", .fromMode(0));
    var f2 = try d.createFile(io, "b.txt", .{});
    try f2.writeStreamingAll(io, "bbb");
    f2.close(io);
    d.close(io);

    var buf: [1024]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = base,
        .recursively = true,
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(std.mem.indexOf(u8, got, "a.txt") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "b.txt") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "denied") != null);
}

test "dirRun noerroronfind suppresses walk diagnostics" {
    if (comptime @import("builtin").os.tag == .windows) return;

    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "modes_dir_noerr_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer restoreModeAndDeleteTree(io, base, "denied");

    var d = try std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true });
    var f1 = try d.createFile(io, "ok.txt", .{});
    try f1.writeStreamingAll(io, "ok");
    f1.close(io);
    try d.createDir(io, "denied", .fromMode(0));
    d.close(io);

    var buf: [1024]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = base,
        .recursively = true,
        .no_error_on_find = true,
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(std.mem.indexOf(u8, got, "ok.txt") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "AccessDenied") == null);
    try std.testing.expect(std.mem.indexOf(u8, got, "PermissionDenied") == null);
}

test "dirRun -o saves cannot-open-directory error" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const missing = "modes_dir_missing_probe_nope";
    const save_path = "modes_dir_missing_save_out.txt";
    defer std.Io.Dir.cwd().deleteFile(io, save_path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{
            .save_result_path = save_path,
        },
        .dir_path = missing,
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const console = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(std.mem.indexOf(u8, console, "cannot open directory") != null);

    const saved = try std.Io.Dir.cwd().readFileAlloc(io, save_path, std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(saved);
    const saved_lf = try std.mem.replaceOwned(u8, std.testing.allocator, saved, "\r\n", "\n");
    defer std.testing.allocator.free(saved_lf);
    try std.testing.expectEqualStrings(console, saved_lf);
}

test "dirRun noerroronfind suppresses cannot-open-directory" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const missing = "modes_dir_missing_noerr_probe_nope";

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var dctx: t.DirCtx = .{
        .opts = .{},
        .dir_path = missing,
        .no_error_on_find = true,
    };

    // Act
    try dirRun(&dctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expectEqual(@as(usize, 0), got.len);
}

fn restoreModeAndDeleteTree(io: std.Io, base: []const u8, denied_name: []const u8) void {
    // deleteTree must open `denied`; restore mode first.
    var bd = std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true }) catch {
        std.Io.Dir.cwd().deleteTree(io, base) catch {};
        return;
    };
    bd.setFilePermissions(io, denied_name, .fromMode(0o700), .{}) catch {};
    bd.close(io);
    std.Io.Dir.cwd().deleteTree(io, base) catch {};
}
