const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types.zig");

const builtin = @import("builtin.zig");
const save = @import("save.zig");

pub const FileCtx = t.FileCtx;
pub const RunEnv = t.RunEnv;
pub const RunError = t.RunError;

pub const FileResult = struct {
    digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8),
    digest_len: usize = 0,
    file_size: u64 = 0,
    time: lib.Time = .{},
    hash_computed: bool = false,
    matches: ?bool = null,
    open_error: ?[]const u8 = null,
    offset_error: ?[]const u8 = null,
    info_error: ?[]const u8 = null,
    hash_error: ?[]const u8 = null,

    pub fn hasStructuralError(self: *const FileResult) bool {
        return self.open_error != null or
            self.offset_error != null or self.info_error != null or
            self.hash_error != null;
    }
};

pub const OFFSET_TOO_BIG = "Offset is greater then file size";

fn calcHashStream(
    file: std.Io.File,
    io: std.Io,
    hash_def: *const hashes.HashDefinition,
    file_size: u64,
    limit: u64,
    offset: u64,
    digest: []u8,
) RunError!?[]const u8 {
    const file_part_size = @min(limit, file_size);

    // Stack context (MAX_CONTEXT_SIZE >= every algo) avoids a per-file heap
    // allocation; the read buffer below uses the page allocator directly so it
    // is returned to the OS even when the caller passes the process-wide arena
    // (whose .free is a no-op). Together this prevents a per-file leak of up to
    // FILE_BIG_BUFFER_SIZE (1 MiB) during directory walks.
    var ctx_storage: [t.MAX_CONTEXT_SIZE]u8 align(16) = std.mem.zeroes([t.MAX_CONTEXT_SIZE]u8);
    const ctx_ptr: *anyopaque = @ptrCast(&ctx_storage);
    hash_def.init(ctx_ptr);

    if (file_part_size == 0) {
        hashes.compute(hash_def, "", digest);
        return null;
    }
    if (offset >= file_size) {
        return null;
    }

    const page_size = if (file_part_size > t.FILE_BIG_BUFFER_SIZE) t.FILE_BIG_BUFFER_SIZE else file_part_size;
    const read_buf = std.heap.page_allocator.alloc(u8, page_size) catch return error.OutOfMemory;
    defer std.heap.page_allocator.free(read_buf);

    var total_read: u64 = 0;
    while (total_read < limit) {
        const want = @min(page_size, limit - total_read);
        const got = file.readPositional(io, &.{read_buf[0..want]}, offset + total_read) catch return "read error";
        if (got == 0) break;
        hash_def.update(ctx_ptr, read_buf.ptr, got);
        total_read += got;
    }

    hash_def.final(ctx_ptr, digest.ptr);
    return null;
}

pub fn calculateFile(
    path: []const u8,
    ctx: *FileCtx,
    env: RunEnv,
    hash_def: *const hashes.HashDefinition,
) RunError!FileResult {
    var result: FileResult = .{};
    result.digest_len = hash_def.hash_length;
    const io = env.io;

    const dir = std.Io.Dir.cwd();
    var file = dir.openFile(io, path, .{}) catch {
        result.open_error = "open error";
        return result;
    };
    defer file.close(io);

    const stat = file.stat(io) catch {
        result.info_error = "stat error";
        return result;
    };
    result.file_size = stat.size;

    const offset_u: u64 = @intCast(@max(ctx.opts.offset, 0));
    const limit_u: u64 = blk: {
        // A non-positive limit means "no limit" (whole file), mirroring the C
        // baseline which maps a zero limit to MAXLONG64. The CLI default already
        // passes maxInt(i64); this guards any caller that forwards 0.
        if (ctx.opts.limit <= 0) break :blk std.math.maxInt(u64);
        break :blk @intCast(ctx.opts.limit);
    };

    var digest_to_compare: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    var is_zero_search_hash = false;
    const has_search = ctx.opts.hash != null and ctx.opts.hash.?.len > 0;
    if (has_search) {
        // File/dir `-b` is output-only (C fhash_to_digest always took hex). Hash
        // mode uses `-b` for input Base64; do not reuse that here.
        t.parseSearchHash(ctx.opts.hash.?, false, hash_def, &digest_to_compare) catch {
            result.hash_error = "invalid search hash";
            return result;
        };
        var empty_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
        hashes.compute(hash_def, "", empty_digest[0..hash_def.hash_length]);
        if (std.mem.eql(u8, empty_digest[0..hash_def.hash_length], digest_to_compare[0..hash_def.hash_length])) {
            is_zero_search_hash = true;
        }
    }

    const hash_started = std.Io.Clock.awake.now(io);

    if (offset_u >= stat.size and stat.size > 0) {
        result.offset_error = OFFSET_TOO_BIG;
    } else {
        const err_msg = calcHashStream(file, io, hash_def, stat.size, limit_u, offset_u, result.digest[0..hash_def.hash_length]) catch |e| {
            return e;
        };
        if (err_msg) |m| {
            result.hash_error = m;
        } else {
            result.hash_computed = true;
        }
    }
    result.time = lib.elapsedSince(io, hash_started);

    if (has_search) {
        const matches = if (!result.hash_computed)
            false
        else blk: {
            const eq = std.mem.eql(
                u8,
                result.digest[0..hash_def.hash_length],
                digest_to_compare[0..hash_def.hash_length],
            );
            break :blk (!is_zero_search_hash and eq) or (is_zero_search_hash and stat.size == 0);
        };
        result.matches = matches;
    }

    return result;
}

fn writeResult(
    path: []const u8,
    ctx: *FileCtx,
    hash_def: *const hashes.HashDefinition,
    res: *const FileResult,
    env: RunEnv,
) RunError!void {
    const out = env.out;
    const is_print_sfv = ctx.opts.result_in_sfv;
    const is_print_verify = ctx.opts.is_verify;

    var hash_repr_buf: [t.MAX_DIGEST_SIZE * 2 + 8]u8 = undefined;
    const hash_repr: ?[]const u8 = if (res.hash_computed)
        t.formatHash(res.digest[0..hash_def.hash_length], ctx.opts.builtin.is_print_low_case, ctx.opts.is_base64, &hash_repr_buf)
    else
        null;

    const has_search = ctx.opts.hash != null and ctx.opts.hash.?.len > 0;
    // C contract (file.c): `is_validate_file_by_hash_ = ctx->hash_ != NULL`, so
    // a file given with -m is ALWAYS in validate mode — emit "File is valid" /
    // "File is invalid" regardless of -c. Search mode (path | size, non-match
    // suppressed) is the *dir* path (filehash.c's hash_to_search &&
    // !is_validate_file_by_hash), not file. -c / is_verify only selects the SFV
    // output format (hash | path) below — it does not toggle VALID/INVALID.
    // The classic do_not_output suppression is therefore unreachable here
    // (matches is only ever set when has_search), matching C; black-box tests
    // CmdFileTests.CalcFile_ValidateFile_{Success,Failure} lock this behavior.
    const validation: ?[]const u8 = if (has_search)
        (if (res.matches orelse false) t.VALID else t.INVALID)
    else
        null;

    var size_buf: [64]u8 = undefined;
    var size_writer: std.Io.Writer = .fixed(&size_buf);
    try lib.formatSize(res.file_size, &size_writer);
    const size_str = std.Io.Writer.buffered(&size_writer);

    var time_buf: [96]u8 = undefined;
    var time_writer: std.Io.Writer = .fixed(&time_buf);
    try lib.formatTime(res.time, &time_writer);
    const time_str = std.Io.Writer.buffered(&time_writer);

    if (is_print_sfv) {
        if (hash_repr) |h| {
            try out.print("{s}{s}{s}\n", .{ std.fs.path.basenameWindows(path), t.SFV_SEPARATOR, h });
        }
    } else if (is_print_verify) {
        if (hash_repr) |h| {
            try out.print("{s}{s}{s}\n", .{ h, t.SFV_SEPARATOR, path });
        }
    } else if (res.hasStructuralError()) {
        const msg = res.open_error orelse res.offset_error orelse
            res.info_error orelse res.hash_error orelse "";
        try out.print("{s}{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, msg });
    } else if (ctx.opts.show_time) {
        const tail = validation orelse hash_repr orelse "";
        try out.print("{s}{s}{s}{s}{s}{s}{s}\n", .{
            path,                t.FILE_INFO_COLUMN_SEPARATOR,
            size_str,            t.FILE_INFO_COLUMN_SEPARATOR,
            time_str,            t.FILE_INFO_COLUMN_SEPARATOR,
            tail,
        });
    } else {
        const tail = validation orelse hash_repr orelse "";
        try out.print("{s}{s}{s}{s}{s}\n", .{
            path,
            t.FILE_INFO_COLUMN_SEPARATOR,
            size_str,
            t.FILE_INFO_COLUMN_SEPARATOR,
            tail,
        });
    }
    // Dir walks hash many files into the process stdout buffer (16 KiB in
    // main); flush so each file's line appears as soon as it is ready.
    try out.flush();
}

pub fn hashAndWriteFile(
    path: []const u8,
    ctx: *FileCtx,
    env: RunEnv,
    hash_def: *const hashes.HashDefinition,
) RunError!void {
    const res = try calculateFile(path, ctx, env, hash_def);
    try writeResult(path, ctx, hash_def, &res, env);
}

pub fn fileRun(
    ctx: *FileCtx,
    env: RunEnv,
    hash_def: *const hashes.HashDefinition,
) RunError!void {
    if (!try builtin.allowSfvOption(ctx.opts.result_in_sfv, hash_def, env.out)) {
        return;
    }
    // Mirror C file.c: -o tees the result line to console and a save file.
    // defer finish before deinit so error returns still persist the capture —
    // matching dir mode (and C file.c which wrote the result/error line to -o).
    var tee = save.SaveTee.init(env.allocator, ctx.opts.save_result_path);
    defer tee.deinit();
    defer tee.finish(env);
    const sink_env = tee.sinkEnv(env);
    try hashAndWriteFile(ctx.file_path, ctx, sink_env, hash_def);
    try tee.flush(env.out);
}

fn writeTempFile(io: std.Io, path: []const u8, content: []const u8) !void {
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, content);
}

test "fileRun hashes a temp file (tiger)" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var fctx: FileCtx = .{ .opts = .{ .builtin = &bctx }, .file_path = path };

    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "hello", expected_digest[0..24]);
    var exp_buf: [64]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..24], false, &exp_buf);

    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&buf, "{s}{s}5 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, exp_hex }),
        got,
    );
}

test "fileRun partial hash with offset and limit" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_partial_probe.txt";
    try writeTempFile(io, path, "0123456789");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var fctx: FileCtx = .{
        .opts = .{
            .builtin = &bctx,
            .offset = 2,
            .limit = 4,
        },
        .file_path = path,
    };

    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "2345", expected_digest[0..24]);
    var exp_buf: [64]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..24], false, &exp_buf);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(got.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, got, exp_hex) != null);
}

test "fileRun validates matching hash" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_validate_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "hello", expected_digest[0..24]);
    var exp_hex_buf: [64]u8 = undefined;
    const expected_hex = t.hashToHex(expected_digest[0..24], false, &exp_hex_buf);

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var fctx: FileCtx = .{
        .opts = .{
            .builtin = &bctx,
            .hash = expected_hex,
        },
        .file_path = path,
    };

    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, t.VALID) != null);
    try std.testing.expect(std.mem.indexOf(u8, got, t.INVALID) == null);
}

test "fileRun -b does not reinterpret -m hex as Base64" {
    // Regression: file/dir -b is output-only; -m stays hex (classic fhash_to_digest).
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_validate_b64_flag_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "hello", expected_digest[0..24]);
    var exp_hex_buf: [64]u8 = undefined;
    const expected_hex = t.hashToHex(expected_digest[0..24], false, &exp_hex_buf);

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var fctx: FileCtx = .{
        .opts = .{
            .builtin = &bctx,
            .hash = expected_hex,
            .is_base64 = true,
        },
        .file_path = path,
    };

    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, t.VALID) != null);
    try std.testing.expect(std.mem.indexOf(u8, got, t.INVALID) == null);
}

test "fileRun rejects non-matching hash" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_invalidate_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    // Valid tiger hex length (48), but wrong digest.
    var fctx: FileCtx = .{
        .opts = .{
            .builtin = &bctx,
            .hash = "000000000000000000000000000000000000000000000000",
        },
        .file_path = path,
    };

    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, t.INVALID) != null);
}

test "fileRun prints hash_error for invalid -m" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_bad_search_hash_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var fctx: FileCtx = .{
        .opts = .{
            .builtin = &bctx,
            .hash = "not-a-hex-digest",
        },
        .file_path = path,
    };

    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, got, "invalid search hash") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, t.INVALID) == null);
}

test "hasStructuralError includes hash_error" {
    var res: FileResult = .{ .hash_error = "read error" };
    try std.testing.expect(res.hasStructuralError());
    res = .{};
    try std.testing.expect(!res.hasStructuralError());
}

test "fileRun -o tees console output into save file" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_save_probe.txt";
    const save_path = "modes_file_save_out.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    defer std.Io.Dir.cwd().deleteFile(io, save_path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    const bctx: t.BuiltinCtx = .{ .hash_algorithm = "tiger" };
    var fctx: FileCtx = .{
        .opts = .{
            .builtin = &bctx,
            .save_result_path = save_path,
        },
        .file_path = path,
    };

    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const console = std.Io.Writer.buffered(&writer);
    try std.testing.expect(console.len > 0);

    const saved = try std.Io.Dir.cwd().readFileAlloc(io, save_path, std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(saved);
    // Windows save path translates \n → \r\n (legacy CRT text mode); compare
    // logical lines so the tee contract holds on every OS.
    const saved_lf = try std.mem.replaceOwned(u8, std.testing.allocator, saved, "\r\n", "\n");
    defer std.testing.allocator.free(saved_lf);
    try std.testing.expectEqualStrings(console, saved_lf);
}
