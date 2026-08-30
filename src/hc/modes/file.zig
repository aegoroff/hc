const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types.zig");

const save = @import("save.zig");

pub const FileResult = struct {
    digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8),
    digest_len: usize = 0,
    file_size: u64 = 0,
    time: lib.Time = .{},
    hash_computed: bool = false,
    matches: ?bool = null,
    /// Structural failure message (open/stat/offset/hash); at most one is set.
    err: ?[]const u8 = null,

    pub fn isOffsetTooBig(self: *const FileResult) bool {
        return if (self.err) |e| std.mem.eql(u8, e, OFFSET_TOO_BIG) else false;
    }
};

pub const OFFSET_TOO_BIG = "Offset is greater than file size";

fn calcHashStream(
    file: std.Io.File,
    io: std.Io,
    hash_def: *const hashes.HashDefinition,
    file_size: u64,
    limit: u64,
    offset: u64,
    digest: []u8,
) t.RunError!?[]const u8 {
    const file_part_size = @min(limit, file_size);

    // Empty file/part: oneshot digest only (no streaming init).
    if (file_part_size == 0) {
        hashes.compute(hash_def, "", digest);
        return null;
    }

    // Stack context (MAX_CONTEXT_SIZE >= every algo) avoids a per-file heap
    // allocation; the read buffer below uses the page allocator directly so it
    // is returned to the OS even when the caller passes the process-wide arena
    // (whose .free is a no-op). Together this prevents a per-file leak of up to
    // FILE_BIG_BUFFER_SIZE (1 MiB) during directory walks.
    var ctx_storage: [t.MAX_CONTEXT_SIZE]u8 align(t.MAX_CONTEXT_ALIGN) = std.mem.zeroes([t.MAX_CONTEXT_SIZE]u8);
    const ctx_ptr: *anyopaque = @ptrCast(&ctx_storage);
    hash_def.init(ctx_ptr);

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
    ctx: *t.FileCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
) t.RunError!FileResult {
    var result: FileResult = .{};
    result.digest_len = hash_def.hash_length;
    const io = env.io;

    const dir = std.Io.Dir.cwd();
    var file = dir.openFile(io, path, .{}) catch {
        result.err = "open error";
        return result;
    };
    defer file.close(io);

    const stat = file.stat(io) catch {
        result.err = "stat error";
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
    const has_search = ctx.opts.hash != null and ctx.opts.hash.?.len > 0;
    if (has_search) {
        // File/dir `-b` is output-only (C fhash_to_digest always took hex). Hash
        // mode uses `-b` for input Base64; do not reuse that here.
        t.parseSearchHash(ctx.opts.hash.?, false, hash_def, &digest_to_compare) catch {
            result.err = "invalid search hash";
            return result;
        };
    }

    const hash_started = std.Io.Clock.awake.now(io);

    if (offset_u > 0 and offset_u >= stat.size) {
        result.err = OFFSET_TOO_BIG;
    } else {
        const err_msg = calcHashStream(file, io, hash_def, stat.size, limit_u, offset_u, result.digest[0..hash_def.hash_length]) catch |e| {
            return e;
        };
        if (err_msg) |m| {
            result.err = m;
        } else {
            result.hash_computed = true;
        }
    }
    result.time = lib.elapsedSince(io, hash_started);

    if (has_search) {
        result.matches = result.hash_computed and std.mem.eql(
            u8,
            result.digest[0..hash_def.hash_length],
            digest_to_compare[0..hash_def.hash_length],
        );
    }

    return result;
}

fn writeResult(
    path: []const u8,
    ctx: *t.FileCtx,
    hash_def: *const hashes.HashDefinition,
    res: *const FileResult,
    env: t.RunEnv,
) t.RunError!void {
    const out = env.out;
    const is_print_sfv = ctx.opts.result_in_sfv;
    const is_print_verify = ctx.opts.is_verify;

    var hash_repr_buf: [t.MAX_DIGEST_SIZE * 2 + 8]u8 = undefined;
    const hash_repr: ?[]const u8 = if (res.hash_computed)
        t.formatHash(res.digest[0..hash_def.hash_length], ctx.opts.low_case, ctx.opts.is_base64, &hash_repr_buf)
    else
        null;

    const has_search = ctx.opts.hash != null and ctx.opts.hash.?.len > 0;
    // A file given with -m is always in validate mode — emit "File is valid" /
    // "File is invalid" regardless of -c. Search mode (path | size, non-match
    // suppressed) is the *dir* path, not file. -c / is_verify only selects the
    // SFV output format (hash | path) below — it does not toggle VALID/INVALID.
    // The do_not_output suppression is therefore unreachable here (matches is
    // only ever set when has_search).
    const validation: ?[]const u8 = if (has_search)
        (if (res.matches orelse false) t.VALID else t.INVALID)
    else
        null;

    var size_buf: [64]u8 = undefined;
    var size_writer: std.Io.Writer = .fixed(&size_buf);
    try lib.formatSize(res.file_size, &size_writer);
    const size_str = std.Io.Writer.buffered(&size_writer);

    if (is_print_sfv) {
        if (hash_repr) |h| {
            try out.print("{s}{s}{s}\n", .{ std.fs.path.basenameWindows(path), t.SFV_SEPARATOR, h });
        }
    } else if (is_print_verify) {
        if (hash_repr) |h| {
            try out.print("{s}{s}{s}\n", .{ h, t.CHECKSUM_SEPARATOR, path });
        }
    } else if (res.err) |msg| {
        try out.print("{s}{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, msg });
    } else {
        const sep = t.FILE_INFO_COLUMN_SEPARATOR;
        const tail = validation orelse hash_repr orelse "";
        try out.print("{s}{s}{s}", .{ path, sep, size_str });
        if (ctx.opts.show_time) {
            var time_buf: [96]u8 = undefined;
            var time_writer: std.Io.Writer = .fixed(&time_buf);
            try lib.formatTime(res.time, &time_writer);
            try out.print("{s}{s}", .{ sep, std.Io.Writer.buffered(&time_writer) });
        }
        try out.print("{s}{s}\n", .{ sep, tail });
    }
    // Dir walks hash many files into the process stdout buffer (16 KiB in
    // main); flush so each file's line appears as soon as it is ready.
    try out.flush();
}

pub fn hashAndWriteFile(
    path: []const u8,
    ctx: *t.FileCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
) t.RunError!void {
    const res = try calculateFile(path, ctx, env, hash_def);
    try writeResult(path, ctx, hash_def, &res, env);
}

pub fn fileRun(
    ctx: *t.FileCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
) t.RunError!void {
    // -o tees the result line to console and a save file.
    // defer finish before deinit so error returns still persist the capture —
    // matching dir mode.
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
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{ .opts = .{}, .file_path = path };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "hello", expected_digest[0..24]);
    var exp_buf: [64]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..24], false, &exp_buf);

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}5 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, exp_hex }),
        got,
    );
}

test "fileRun partial hash with offset and limit" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_partial_probe.txt";
    try writeTempFile(io, path, "0123456789");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{
            .offset = 2,
            .limit = 4,
        },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "2345", expected_digest[0..24]);
    var exp_buf: [64]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..24], false, &exp_buf);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}10 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, exp_hex }),
        got,
    );
}

test "fileRun validates matching hash" {
    // Arrange
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
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{
            .hash = expected_hex,
        },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}5 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, t.VALID }),
        got,
    );
}

test "fileRun -b does not reinterpret -m hex as Base64" {
    // Arrange
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
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{
            .hash = expected_hex,
            .is_base64 = true,
        },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(std.mem.indexOf(u8, got, t.VALID) != null);
    try std.testing.expect(std.mem.indexOf(u8, got, t.INVALID) == null);
}

test "fileRun crc32 00000000 matches nonempty collision" {
    // Arrange
    const payload = "\x9d\x0a\xd9\x6d";
    const crc32 = hashes.getHash("crc32").?;
    var collision_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(crc32, payload, collision_digest[0..4]);
    var hex_buf: [8]u8 = undefined;
    const hex = t.hashToHex(collision_digest[0..4], false, &hex_buf);
    try std.testing.expectEqualStrings("00000000", hex);

    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_crc32_zero_collision_probe.bin";
    try writeTempFile(io, path, payload);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{
            .hash = "00000000",
        },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, crc32);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}4 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, t.VALID }),
        got,
    );
}

test "fileRun rejects non-matching hash" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_invalidate_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    // Valid tiger hex length (48), but wrong digest.
    var fctx: t.FileCtx = .{
        .opts = .{
            .hash = "000000000000000000000000000000000000000000000000",
        },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}5 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, t.INVALID }),
        got,
    );
}

test "fileRun nonexistent file reports open error" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_missing_probe.txt";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{ .opts = .{}, .file_path = path };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}open error\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR }),
        got,
    );
}

test "fileRun -c checksum format is digest then path" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_checksum_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "hello", expected_digest[0..24]);
    var exp_buf: [64]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..24], false, &exp_buf);

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{ .is_verify = true },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}{s}\n", .{ exp_hex, t.CHECKSUM_SEPARATOR, path }),
        got,
    );
}

test "fileRun --sfv prints basename and crc32" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_sfv_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("crc32").?, "hello", expected_digest[0..4]);
    var exp_buf: [64]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..4], false, &exp_buf);

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{ .result_in_sfv = true },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("crc32").?);

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.fmt.bufPrint(&want, "{s}{s}{s}\n", .{ path, t.SFV_SEPARATOR, exp_hex }),
        got,
    );
}

test "fileRun -t keeps the digest tail after the time column" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_time_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var expected_digest: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(hashes.getHash("tiger").?, "hello", expected_digest[0..24]);
    var exp_buf: [64]u8 = undefined;
    const exp_hex = t.hashToHex(expected_digest[0..24], false, &exp_buf);

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{ .show_time = true },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    // Elapsed time is nondeterministic: pin the head and the digest tail of
    // the 4-field line instead of the full string.
    var head_buf: [128]u8 = undefined;
    var tail_buf: [128]u8 = undefined;
    const head = try std.fmt.bufPrint(&head_buf, "{s}{s}5 bytes{s}", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR });
    const tail = try std.fmt.bufPrint(&tail_buf, "{s}{s}\n", .{ t.FILE_INFO_COLUMN_SEPARATOR, exp_hex });
    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(std.mem.startsWith(u8, got, head));
    try std.testing.expect(std.mem.endsWith(u8, got, tail));
}

test "fileRun prints err for invalid -m" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_bad_search_hash_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{
            .hash = "not-a-hex-digest",
        },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(std.mem.indexOf(u8, got, "invalid search hash") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, t.INVALID) == null);
}

test "isOffsetTooBig matches OFFSET_TOO_BIG only" {
    // Arrange / Act / Assert
    var res: FileResult = .{ .err = "read error" };
    try std.testing.expect(!res.isOffsetTooBig());
    try std.testing.expect(res.err != null);
    res = .{ .err = OFFSET_TOO_BIG };
    try std.testing.expect(res.isOffsetTooBig());
    res = .{};
    try std.testing.expect(res.err == null);
    try std.testing.expect(!res.isOffsetTooBig());
}

test "fileRun -o tees console output into save file" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_save_probe.txt";
    const save_path = "modes_file_save_out.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    defer std.Io.Dir.cwd().deleteFile(io, save_path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{
            .save_result_path = save_path,
        },
        .file_path = path,
    };

    // Act
    try fileRun(&fctx, env, hashes.getHash("tiger").?);

    const console = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(console.len > 0);

    const saved = try std.Io.Dir.cwd().readFileAlloc(io, save_path, std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(saved);
    // Windows save path translates \n → \r\n; compare logical lines so the
    // tee contract holds on every OS.
    const saved_lf = try std.mem.replaceOwned(u8, std.testing.allocator, saved, "\r\n", "\n");
    defer std.testing.allocator.free(saved_lf);
    try std.testing.expectEqualStrings(console, saved_lf);
}
