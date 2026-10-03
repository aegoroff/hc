const std = @import("std");
const builtin = @import("builtin");
const lib = @import("lib");
const hashes = @import("hashes");
const t = @import("types");
const Report = @import("../report.zig").Report;

/// Why a file produced no digest.
pub const FileFailure = enum {
    invalid_search_hash,
    offset_past_eof,
    open,
    stat,
    read,

    /// Reason printed after `path | ` in every output format.
    pub fn message(self: FileFailure) []const u8 {
        return switch (self) {
            .invalid_search_hash => "invalid search hash",
            .offset_past_eof => t.OFFSET_TOO_BIG,
            .open => "open error",
            .stat => "stat error",
            .read => "read error",
        };
    }
};

/// Digest of one file with its timing and search verdict.
pub const FileHashed = struct {
    digest: FileDigest,
    time: lib.Time = .{},
    /// Set only when a search hash (`-m` / `--search`) was given.
    matches: ?bool = null,
};

/// Result of hashing one file: a digest or the reason there is none.
pub const FileOutcome = union(enum) {
    hashed: FileHashed,
    failed: FileFailure,
};

/// How a file outcome is printed.
pub const OutputFormat = enum {
    /// `path | size [| time] | digest`, or the `-m` verdict instead of the digest.
    listing,
    /// `name    digest` (`--sfv`).
    sfv,
    /// `digest path` (`-c`).
    checksum,
    /// `path | size`, matching files only (dir search).
    search,

    pub fn fromOptions(opts: *const t.FileOptions) OutputFormat {
        if (opts.result_in_sfv) return .sfv;
        if (opts.is_verify) return .checksum;
        return .listing;
    }
};

pub const FileWindow = t.FileWindow;
pub const FileDigest = t.FileDigest;
pub const FileDigestError = t.FileDigestError;

/// Kind of the entry at `path` (following symlinks).
pub fn pathKind(io: std.Io, path: []const u8) t.PathError!std.Io.File.Kind {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.OpenFailed;
    defer file.close(io);
    const st = file.stat(io) catch return error.StatFailed;
    return st.kind;
}

/// Size of the file at `path`. When stat reports 0 (procfs, pipes) the file
/// is read to EOF and the bytes counted, like `createFileDigest` does.
pub fn fileSize(io: std.Io, path: []const u8) t.PathError!u64 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.OpenFailed;
    defer file.close(io);
    const st = file.stat(io) catch return error.StatFailed;
    if (st.size != 0) return st.size;
    var buf: [16 * 1024]u8 = undefined;
    var reader: WindowReader = .{ .file = file, .io = io, .pos = 0 };
    var total: u64 = 0;
    while (true) {
        const got = reader.read(&buf) catch return error.ReadFailed;
        if (got == 0) return total;
        total += got;
    }
}

/// Digest of a file window. `path` is a filesystem path, not text to hash.
/// Presentation (SFV, timing, `-m`) stays in `hashFile` / `writeOutcome`.
/// Reads until EOF or `limit` rather than trusting the stat size, which is
/// 0 for pipes and procfs files.
pub fn createFileDigest(
    h: *const hashes.HashDefinition,
    path: []const u8,
    window: FileWindow,
    io: std.Io,
) FileDigestError!FileDigest {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.OpenFailed;
    defer file.close(io);

    const st = file.stat(io) catch return error.StatFailed;
    const offset_u: u64 = @intCast(@max(window.offset, 0));
    const limit_u: u64 = if (window.limit <= 0)
        std.math.maxInt(u64)
    else
        @intCast(window.limit);

    var result: FileDigest = .{
        .len = h.hash_length,
        .file_size = st.size,
    };
    const window_read = try hashFileWindow(file, io, h, st.size, limit_u, offset_u, result.bytes[0..h.hash_length]);
    if (st.size == 0 and window_read.reached_eof) result.file_size = offset_u + window_read.bytes;
    return result;
}

/// Sequential reader over a file window. Positional reads keep regular
/// files independent of the descriptor position; pipes and terminals cannot
/// seek, so the first `Unseekable` switches to streaming reads and skips the
/// window offset by reading it.
const WindowReader = struct {
    file: std.Io.File,
    io: std.Io,
    pos: u64,
    streaming: bool = false,

    /// Fills at most `buf.len` bytes; 0 means end of file.
    fn read(self: *WindowReader, buf: []u8) FileDigestError!usize {
        if (!self.streaming) {
            if (self.file.readPositional(self.io, &.{buf}, self.pos)) |got| {
                self.pos += got;
                return got;
            } else |err| switch (err) {
                error.Unseekable => {
                    self.streaming = true;
                    try self.skip(self.pos, buf);
                },
                else => return error.ReadFailed,
            }
        }
        return self.readStream(buf);
    }

    fn readStream(self: *WindowReader, buf: []u8) FileDigestError!usize {
        return self.file.readStreaming(self.io, &.{buf}) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => error.ReadFailed,
        };
    }

    /// Discards `count` leading stream bytes through `scratch`. Stops early
    /// at EOF; the next `read` then returns 0.
    fn skip(self: *WindowReader, count: u64, scratch: []u8) FileDigestError!void {
        var left = count;
        while (left > 0) {
            const got = try self.readStream(scratch[0..@intCast(@min(scratch.len, left))]);
            if (got == 0) return;
            left -= got;
        }
    }
};

const WindowRead = struct {
    /// Bytes hashed.
    bytes: u64,
    /// The read stopped at EOF, not at `limit`.
    reached_eof: bool,
};

fn hashFileWindow(
    file: std.Io.File,
    io: std.Io,
    hash_def: *const hashes.HashDefinition,
    file_size: u64,
    limit: u64,
    offset: u64,
    digest: []u8,
) FileDigestError!WindowRead {
    // Stack context avoids a per-file heap allocation; the read buffer uses
    // the page allocator so it is returned to the OS even when the caller
    // passes a process-wide arena (whose .free is a no-op).
    var ctx_storage: [t.MAX_CONTEXT_SIZE]u8 align(t.MAX_CONTEXT_ALIGN) = std.mem.zeroes([t.MAX_CONTEXT_SIZE]u8);
    const ctx_ptr: *anyopaque = @ptrCast(&ctx_storage);
    hash_def.init(ctx_ptr);

    // A zero stat size is unknown, not empty (pipes, procfs): use a full buffer.
    const known_size = if (file_size == 0) t.FILE_BIG_BUFFER_SIZE else file_size;
    const page_size: usize = @intCast(@max(@min(limit, known_size, t.FILE_BIG_BUFFER_SIZE), 1));
    const read_buf = std.heap.page_allocator.alloc(u8, page_size) catch return error.OutOfMemory;
    defer std.heap.page_allocator.free(read_buf);

    var reader: WindowReader = .{ .file = file, .io = io, .pos = offset };
    var total_read: u64 = 0;
    var reached_eof = false;
    while (total_read < limit) {
        const remaining = limit - total_read;
        const want: usize = @intCast(@min(page_size, remaining));
        const got = try reader.read(read_buf[0..want]);
        if (got == 0) {
            reached_eof = true;
            break;
        }
        hash_def.update(ctx_ptr, read_buf.ptr, got);
        total_read += got;
    }
    // limit is at least 1, so an empty first read at a non-zero offset means
    // the offset is at or past EOF.
    if (offset > 0 and total_read == 0) return error.OffsetPastEof;

    hash_def.final(ctx_ptr, digest.ptr);
    return .{ .bytes = total_read, .reached_eof = reached_eof };
}

/// Hashes the `opts.offset` / `opts.limit` window of `path` and, when
/// `opts.hash` is non-empty, compares the digest against it.
pub fn hashFile(
    hash_def: *const hashes.HashDefinition,
    path: []const u8,
    opts: *const t.FileOptions,
    io: std.Io,
) error{OutOfMemory}!FileOutcome {
    var expected: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    const search = if (opts.hash) |s| (if (s.len > 0) s else null) else null;
    if (search) |s| {
        // File/dir `-b` is output-only (C fhash_to_digest always took hex). Hash
        // mode uses `-b` for input Base64; do not reuse that here.
        t.parseSearchHash(s, false, hash_def, &expected) catch return .{ .failed = .invalid_search_hash };
    }

    const started = std.Io.Clock.awake.now(io);
    const digest = createFileDigest(hash_def, path, .{
        .offset = opts.offset,
        .limit = opts.limit,
    }, io) catch |err| return switch (err) {
        error.OffsetPastEof => .{ .failed = .offset_past_eof },
        error.OpenFailed => .{ .failed = .open },
        error.StatFailed => .{ .failed = .stat },
        error.ReadFailed => .{ .failed = .read },
        error.OutOfMemory => error.OutOfMemory,
    };
    return .{ .hashed = .{
        .digest = digest,
        .time = lib.elapsedSince(io, started),
        .matches = if (search != null) std.mem.eql(u8, digest.slice(), expected[0..digest.len]) else null,
    } };
}

/// Prints the line for `outcome` in `format`. A failure prints
/// `path | reason` in every format; a search non-match prints nothing.
pub fn writeOutcome(
    out: *std.Io.Writer,
    path: []const u8,
    outcome: *const FileOutcome,
    format: OutputFormat,
    opts: *const t.FileOptions,
) t.RunError!void {
    const sep = t.FILE_INFO_COLUMN_SEPARATOR;
    const res = switch (outcome.*) {
        .failed => |failure| return out.print("{s}{s}{s}\n", .{ path, sep, failure.message() }),
        .hashed => |*h| h,
    };

    var hash_buf: [t.MAX_DIGEST_SIZE * 2 + 8]u8 = undefined;
    const hash_repr = t.formatHash(res.digest.slice(), opts.low_case, opts.is_base64, &hash_buf);

    var size_buf: [64]u8 = undefined;
    var size_writer: std.Io.Writer = .fixed(&size_buf);
    try lib.formatSize(res.digest.file_size, &size_writer);
    const size_str = std.Io.Writer.buffered(&size_writer);

    switch (format) {
        .sfv => try out.print("{s}{s}{s}\n", .{ std.Io.Dir.path.basename(path), t.SFV_SEPARATOR, hash_repr }),
        .checksum => try out.print("{s}{s}{s}\n", .{ hash_repr, t.CHECKSUM_SEPARATOR, path }),
        .search => if (res.matches orelse false) try out.print("{s}{s}{s}\n", .{ path, sep, size_str }),
        .listing => {
            // With -m the verdict replaces the digest at the end of the line.
            const tail = if (res.matches) |m| (if (m) t.VALID else t.INVALID) else hash_repr;
            try out.print("{s}{s}{s}", .{ path, sep, size_str });
            if (opts.show_time) {
                var time_buf: [96]u8 = undefined;
                var time_writer: std.Io.Writer = .fixed(&time_buf);
                try lib.formatTime(res.time, &time_writer);
                try out.print("{s}{s}", .{ sep, std.Io.Writer.buffered(&time_writer) });
            }
            try out.print("{s}{s}\n", .{ sep, tail });
        },
    }
}

/// Hashes `path` and commits its line to `report`. A failed file counts
/// against the run.
pub fn hashIntoReport(
    report: *Report,
    path: []const u8,
    opts: *const t.FileOptions,
    format: OutputFormat,
    hash_def: *const hashes.HashDefinition,
) t.RunError!void {
    const outcome = try hashFile(hash_def, path, opts, report.env.io);
    try writeOutcome(report.writer(), path, &outcome, format, opts);
    try report.commit(outcome == .hashed);
}

const FileJob = struct {
    ctx: *const t.FileCtx,
    hash_def: *const hashes.HashDefinition,

    fn run(job: FileJob, report: *Report) t.RunError!void {
        const opts = &job.ctx.opts;
        try hashIntoReport(report, job.ctx.file_path, opts, .fromOptions(opts), job.hash_def);
    }
};

/// Hashes one file. Returns `error.ProcessingFailed` when the file or the
/// `-o` save failed; the reason is already printed.
pub fn fileRun(
    ctx: *t.FileCtx,
    env: t.RunEnv,
    hash_def: *const hashes.HashDefinition,
) t.RunError!void {
    const job: FileJob = .{ .ctx = ctx, .hash_def = hash_def };
    return Report.run(env, ctx.opts.save_result_path, job, FileJob.run);
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
        try std.mem.print(&want, "{s}{s}5 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, exp_hex }),
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
        try std.mem.print(&want, "{s}{s}10 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, exp_hex }),
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
        try std.mem.print(&want, "{s}{s}5 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, t.VALID }),
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
    try std.testing.expect(std.mem.find(u8, got, t.VALID) != null);
    try std.testing.expect(std.mem.find(u8, got, t.INVALID) == null);
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
        try std.mem.print(&want, "{s}{s}4 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, t.VALID }),
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
        try std.mem.print(&want, "{s}{s}5 bytes{s}{s}\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR, t.INVALID }),
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
    try std.testing.expectError(error.ProcessingFailed, fileRun(&fctx, env, hashes.getHash("tiger").?));

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.mem.print(&want, "{s}{s}open error\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR }),
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
        try std.mem.print(&want, "{s}{s}{s}\n", .{ exp_hex, t.CHECKSUM_SEPARATOR, path }),
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
        try std.mem.print(&want, "{s}{s}{s}\n", .{ path, t.SFV_SEPARATOR, exp_hex }),
        got,
    );
}

test "fileRun --sfv keeps backslash in POSIX file name" {
    // A backslash is an ordinary file name byte on POSIX, not a separator.
    if (comptime builtin.target.os.tag == .windows) return error.SkipZigTest;

    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_sfv_back\\slash.txt";
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
        try std.mem.print(&want, "{s}{s}{s}\n", .{ path, t.SFV_SEPARATOR, exp_hex }),
        got,
    );
}

test "fileRun -c reports open error for missing file" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_verify_missing_probe.txt";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{ .opts = .{ .is_verify = true }, .file_path = path };

    // Act
    try std.testing.expectError(error.ProcessingFailed, fileRun(&fctx, env, hashes.getHash("md5").?));

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.mem.print(&want, "{s}{s}open error\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR }),
        got,
    );
}

test "fileRun --sfv reports open error for missing file" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_sfv_missing_probe.txt";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{ .opts = .{ .result_in_sfv = true }, .file_path = path };

    // Act
    try std.testing.expectError(error.ProcessingFailed, fileRun(&fctx, env, hashes.getHash("crc32").?));

    const got = std.Io.Writer.buffered(&writer);
    var want: [256]u8 = undefined;

    // Assert
    try std.testing.expectEqualStrings(
        try std.mem.print(&want, "{s}{s}open error\n", .{ path, t.FILE_INFO_COLUMN_SEPARATOR }),
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
    const head = try std.mem.print(&head_buf, "{s}{s}5 bytes{s}", .{ path, t.FILE_INFO_COLUMN_SEPARATOR, t.FILE_INFO_COLUMN_SEPARATOR });
    const tail = try std.mem.print(&tail_buf, "{s}{s}\n", .{ t.FILE_INFO_COLUMN_SEPARATOR, exp_hex });
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
    try std.testing.expectError(error.ProcessingFailed, fileRun(&fctx, env, hashes.getHash("tiger").?));

    const got = std.Io.Writer.buffered(&writer);

    // Assert
    try std.testing.expect(std.mem.find(u8, got, "invalid search hash") != null);
    try std.testing.expect(std.mem.find(u8, got, t.INVALID) == null);
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

test "createFileDigest hashes a temp file" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_digest_probe.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    const tiger = hashes.getHash("tiger").?;
    var want: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(tiger, "hello", want[0..tiger.hash_length]);

    // Act
    const got = try createFileDigest(tiger, path, .{}, io);

    // Assert
    try std.testing.expectEqual(@as(u64, 5), got.file_size);
    try std.testing.expectEqual(tiger.hash_length, got.len);
    try std.testing.expectEqualSlices(u8, want[0..tiger.hash_length], got.slice());
}

test "createFileDigest hashes a file window" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_window_probe.txt";
    try writeTempFile(io, path, "0123456789");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    const tiger = hashes.getHash("tiger").?;
    var want: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(tiger, "2345", want[0..tiger.hash_length]);

    // Act
    const got = try createFileDigest(tiger, path, .{ .offset = 2, .limit = 4 }, io);

    // Assert
    try std.testing.expectEqual(@as(u64, 10), got.file_size);
    try std.testing.expectEqualSlices(u8, want[0..tiger.hash_length], got.slice());
}

test "createFileDigest empty file at offset 0" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_empty_digest_probe.txt";
    try writeTempFile(io, path, "");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    const tiger = hashes.getHash("tiger").?;
    var want: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(tiger, "", want[0..tiger.hash_length]);

    // Act
    const got = try createFileDigest(tiger, path, .{}, io);

    // Assert
    try std.testing.expectEqual(@as(u64, 0), got.file_size);
    try std.testing.expectEqualSlices(u8, want[0..tiger.hash_length], got.slice());
}

test "createFileDigest offset past EOF" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_eof_digest_probe.txt";
    try writeTempFile(io, path, "ab");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // Act

    // Assert
    try std.testing.expectError(
        error.OffsetPastEof,
        createFileDigest(hashes.getHash("tiger").?, path, .{ .offset = 2 }, io),
    );
}

test "createFileDigest missing file" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_missing_digest_probe.txt";
    std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // Act

    // Assert
    try std.testing.expectError(
        error.OpenFailed,
        createFileDigest(hashes.getHash("tiger").?, path, .{}, io),
    );
}

fn renderOutcome(
    buf: []u8,
    outcome: FileOutcome,
    format: OutputFormat,
    opts: t.FileOptions,
) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    try writeOutcome(&w, "dir/a.txt", &outcome, format, &opts);
    return std.Io.Writer.buffered(&w);
}

fn testHashed(matches: ?bool) FileOutcome {
    return .{ .hashed = .{
        .digest = .{ .bytes = [_]u8{ 0xde, 0xad, 0xbe, 0xef } ++ @as([t.MAX_DIGEST_SIZE - 4]u8, @splat(0)), .len = 4, .file_size = 3 },
        .matches = matches,
    } };
}

test "writeOutcome prints a failure in every format" {
    // Arrange
    const outcome: FileOutcome = .{ .failed = .open };
    var buf: [128]u8 = undefined;

    for (std.enums.values(OutputFormat)) |format| {
        // Act
        const got = try renderOutcome(&buf, outcome, format, .{});

        // Assert
        try std.testing.expectEqualStrings("dir/a.txt | open error\n", got);
    }
}

test "writeOutcome search prints matching files only" {
    // Arrange
    var buf: [128]u8 = undefined;

    // Act
    const hit = try std.testing.allocator.dupe(u8, try renderOutcome(&buf, testHashed(true), .search, .{}));
    defer std.testing.allocator.free(hit);
    const miss = try renderOutcome(&buf, testHashed(false), .search, .{});

    // Assert
    try std.testing.expectEqualStrings("dir/a.txt | 3 bytes\n", hit);
    try std.testing.expectEqualStrings("", miss);
}

test "writeOutcome listing ends with the -m verdict or the digest" {
    // Arrange
    var buf: [128]u8 = undefined;
    const opts: t.FileOptions = .{ .low_case = true };

    // Act
    const valid = try std.testing.allocator.dupe(u8, try renderOutcome(&buf, testHashed(true), .listing, opts));
    defer std.testing.allocator.free(valid);
    const invalid = try std.testing.allocator.dupe(u8, try renderOutcome(&buf, testHashed(false), .listing, opts));
    defer std.testing.allocator.free(invalid);
    const plain = try renderOutcome(&buf, testHashed(null), .listing, opts);

    // Assert
    try std.testing.expectEqualStrings("dir/a.txt | 3 bytes | File is valid\n", valid);
    try std.testing.expectEqualStrings("dir/a.txt | 3 bytes | File is invalid\n", invalid);
    try std.testing.expectEqualStrings("dir/a.txt | 3 bytes | deadbeef\n", plain);
}

test "writeOutcome sfv prints the base name and checksum puts the digest first" {
    // Arrange
    var buf: [128]u8 = undefined;
    const opts: t.FileOptions = .{ .low_case = true };

    // Act
    const sfv = try std.testing.allocator.dupe(u8, try renderOutcome(&buf, testHashed(null), .sfv, opts));
    defer std.testing.allocator.free(sfv);
    const checksum = try renderOutcome(&buf, testHashed(null), .checksum, opts);

    // Assert
    try std.testing.expectEqualStrings("a.txt    deadbeef\n", sfv);
    try std.testing.expectEqualStrings("deadbeef dir/a.txt\n", checksum);
}

test "OutputFormat.fromOptions prefers sfv over checksum" {
    // Arrange
    const both: t.FileOptions = .{ .result_in_sfv = true, .is_verify = true };
    const verify: t.FileOptions = .{ .is_verify = true };

    // Act
    const from_both = OutputFormat.fromOptions(&both);
    const from_verify = OutputFormat.fromOptions(&verify);

    // Assert
    try std.testing.expectEqual(OutputFormat.sfv, from_both);
    try std.testing.expectEqual(OutputFormat.checksum, from_verify);
}

test "fileRun -o into a missing directory reports the reason and fails" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "modes_file_save_fail_probe.txt";
    const save_path = "modes_file_save_fail_missing_dir/out.txt";
    try writeTempFile(io, path, "hello");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const env: t.RunEnv = .{
        .io = io,
        .allocator = std.testing.allocator,
        .out = &writer,
    };
    var fctx: t.FileCtx = .{
        .opts = .{ .save_result_path = save_path },
        .file_path = path,
    };

    // Act
    const result = fileRun(&fctx, env, hashes.getHash("tiger").?);

    // Assert
    try std.testing.expectError(error.ProcessingFailed, result);
    const got = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.find(u8, got, "5 bytes") != null);
    try std.testing.expect(std.mem.endsWith(
        u8,
        got,
        "Error opening file: " ++ save_path ++ " Error message: FileNotFound\n",
    ));
}

/// Linux FIFO fed by a writer thread, so `createFileDigest` sees a pipe:
/// stat size 0 and no positional reads.
const TestFifo = struct {
    path: [:0]const u8,
    payload: []const u8,
    thread: std.Thread = undefined,

    fn start(self: *TestFifo) !void {
        const linux = std.os.linux;
        std.Io.Dir.cwd().deleteFile(std.Io.Threaded.global_single_threaded.io(), self.path) catch {};
        if (linux.errno(linux.mknod(self.path, linux.S.IFIFO | 0o600, 0)) != .SUCCESS) return error.SkipZigTest;
        self.thread = try std.Thread.spawn(.{}, feed, .{self});
    }

    /// Opening for write blocks until the reader opens the FIFO.
    fn feed(self: *TestFifo) void {
        const linux = std.os.linux;
        const rc = linux.open(self.path, .{ .ACCMODE = .WRONLY }, 0);
        if (linux.errno(rc) != .SUCCESS) return;
        const fd: i32 = @intCast(rc);
        defer _ = linux.close(fd);
        // Short or failed writes surface as a digest mismatch in the test.
        _ = linux.write(fd, self.payload.ptr, self.payload.len);
    }

    fn finish(self: *TestFifo) void {
        self.thread.join();
        std.Io.Dir.cwd().deleteFile(std.Io.Threaded.global_single_threaded.io(), self.path) catch {};
    }
};

test "createFileDigest hashes a pipe until EOF" {
    if (builtin.target.os.tag != .linux) return error.SkipZigTest;

    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiger = hashes.getHash("tiger").?;
    var fifo: TestFifo = .{ .path = "modes_file_fifo_probe", .payload = "hello from a pipe" };
    try fifo.start();
    defer fifo.finish();

    // Act
    const got = try createFileDigest(tiger, fifo.path, .{}, io);

    // Assert
    var want: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(tiger, fifo.payload, want[0..tiger.hash_length]);
    try std.testing.expectEqualSlices(u8, want[0..tiger.hash_length], got.slice());
    try std.testing.expectEqual(@as(u64, fifo.payload.len), got.file_size);
}

test "createFileDigest hashes a pipe window" {
    if (builtin.target.os.tag != .linux) return error.SkipZigTest;

    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiger = hashes.getHash("tiger").?;
    var fifo: TestFifo = .{ .path = "modes_file_fifo_window_probe", .payload = "0123456789" };
    try fifo.start();
    defer fifo.finish();

    // Act
    const got = try createFileDigest(tiger, fifo.path, .{ .offset = 2, .limit = 4 }, io);

    // Assert
    var want: [t.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([t.MAX_DIGEST_SIZE]u8);
    hashes.compute(tiger, "2345", want[0..tiger.hash_length]);
    try std.testing.expectEqualSlices(u8, want[0..tiger.hash_length], got.slice());
}

test "createFileDigest pipe offset past EOF" {
    if (builtin.target.os.tag != .linux) return error.SkipZigTest;

    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    var fifo: TestFifo = .{ .path = "modes_file_fifo_eof_probe", .payload = "ab" };
    try fifo.start();
    defer fifo.finish();

    // Act
    const result = createFileDigest(hashes.getHash("tiger").?, fifo.path, .{ .offset = 2 }, io);

    // Assert
    try std.testing.expectError(error.OffsetPastEof, result);
}

test "fileSize counts procfs bytes that stat reports as 0" {
    if (builtin.target.os.tag != .linux) return error.SkipZigTest;

    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();

    // Act
    const size = try fileSize(io, "/proc/self/status");
    const kind = try pathKind(io, "/proc/self/status");

    // Assert
    try std.testing.expect(size > 0);
    try std.testing.expectEqual(std.Io.File.Kind.file, kind);
}

test "pathKind reports a directory and a missing path" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();

    // Act
    const dir_kind = try pathKind(io, ".");
    const missing = pathKind(io, "modes_file_missing_probe.txt");

    // Assert
    try std.testing.expectEqual(std.Io.File.Kind.directory, dir_kind);
    try std.testing.expectError(error.OpenFailed, missing);
}
