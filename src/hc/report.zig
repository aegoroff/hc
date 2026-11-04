const std = @import("std");
const builtin = @import("builtin");
const t = @import("types");

/// Output of one file or dir run. Records go to the console as soon as they
/// are committed, are copied into the `-o` save file when one is given, and
/// a failed record makes the whole run fail.
pub const Report = struct {
    env: t.RunEnv,
    capture: ?std.Io.Writer.Allocating,
    save_path: ?[]const u8,
    save_file: ?std.Io.File.Atomic = null,
    save_destination: ?[:0]u8 = null,
    failed: bool = false,

    /// Runs `body(context, &report)`, then publishes the streamed save file, even when
    /// `body` returned an error. Returns `error.ProcessingFailed` when a
    /// record failed or the save failed; the reason is already printed.
    pub fn run(
        env: t.RunEnv,
        save_path: ?[]const u8,
        context: anytype,
        comptime body: fn (@TypeOf(context), *Report) t.RunError!void,
    ) t.RunError!void {
        var report: Report = .{
            .env = env,
            .capture = if (save_path != null) .init(env.allocator) else null,
            .save_path = save_path,
        };
        defer if (report.capture) |*aw| aw.deinit();
        defer if (report.save_destination) |path| env.allocator.free(path);
        defer if (report.save_file) |*file| file.deinit(env.io);
        body(context, &report) catch |err| {
            report.save() catch {};
            return err;
        };
        try report.save();
        if (report.failed) return error.ProcessingFailed;
    }

    /// Refuses to save over an input file (including links). Returns false
    /// for the temporary report file if a directory walk encounters it.
    pub fn protectInput(self: *Report, path: []const u8) t.RunError!bool {
        const save_path = self.save_path orelse return true;
        const cwd = std.Io.Dir.cwd();
        if (self.save_file) |file| {
            const input_stat = cwd.statFile(self.env.io, path, .{}) catch |err| switch (err) {
                error.FileNotFound => return true,
                else => return self.rejectSave(path, @errorName(err)),
            };
            const temp_stat = file.file.stat(self.env.io) catch return self.rejectSave(path, "cannot identify temporary report");
            if (input_stat.inode == temp_stat.inode) {
                var input = cwd.openFile(self.env.io, path, .{}) catch return self.rejectSave(path, "cannot open input");
                defer input.close(self.env.io);
                const input_id = fileIdentity(input) catch return self.rejectSave(path, "cannot identify input");
                const temp_id = fileIdentity(file.file) catch return self.rejectSave(path, "cannot identify temporary report");
                if (std.meta.eql(input_id, temp_id)) return false;
            }
        }
        const output_stat = cwd.statFile(self.env.io, save_path, .{}) catch |err| switch (err) {
            error.FileNotFound => return true,
            else => return self.rejectSave(path, @errorName(err)),
        };
        const input_stat = cwd.statFile(self.env.io, path, .{}) catch |err| switch (err) {
            error.FileNotFound => return true,
            else => return self.rejectSave(path, @errorName(err)),
        };
        if (input_stat.kind != .file or output_stat.kind != .file or input_stat.inode != output_stat.inode) return true;
        var output = cwd.openFile(self.env.io, save_path, .{}) catch |err| switch (err) {
            error.FileNotFound => return true,
            else => return self.rejectSave(path, @errorName(err)),
        };
        defer output.close(self.env.io);
        var input = cwd.openFile(self.env.io, path, .{}) catch |err| switch (err) {
            error.FileNotFound => return true,
            else => return self.rejectSave(path, @errorName(err)),
        };
        defer input.close(self.env.io);
        const input_id = fileIdentity(input) catch return self.rejectSave(path, "cannot identify input");
        const output_id = fileIdentity(output) catch return self.rejectSave(path, "cannot identify output");
        if (std.meta.eql(input_id, output_id)) return self.rejectSave(path, "output is an input file");
        return true;
    }

    fn rejectSave(self: *Report, path: []const u8, reason: []const u8) t.RunError {
        self.save_path = null;
        self.env.out.print("Refusing to save report for {s}: {s}\n", .{ path, reason }) catch return error.WriteFailed;
        return error.ProcessingFailed;
    }

    /// Writer for the record being built; `commit` ends the record.
    pub fn writer(self: *Report) *std.Io.Writer {
        return if (self.capture) |*aw| &aw.writer else self.env.out;
    }

    /// Shows the record on the console now and, when `ok` is false, counts it
    /// as a failure of the run.
    pub fn commit(self: *Report, ok: bool) t.RunError!void {
        if (!ok) self.failed = true;
        const aw = if (self.capture) |*a| a else return self.env.out.flush();
        const bytes = aw.written();
        const shown = self.env.out.writeAll(bytes);
        const flushed = self.env.out.flush();
        const saved = self.writeSaved(bytes);
        aw.writer.end = 0;
        try shown;
        try flushed;
        try saved;
    }

    fn writeSaved(self: *Report, bytes: []const u8) t.RunError!void {
        const path = self.save_path orelse return;
        if (self.save_file == null) {
            const stat = std.Io.Dir.cwd().statFile(self.env.io, path, .{}) catch |err| switch (err) {
                error.FileNotFound => null,
                else => {
                    self.save_path = null;
                    try self.env.out.print("\nError opening file: {s} Error message: {s}\n", .{ path, @errorName(err) });
                    return error.ProcessingFailed;
                },
            };
            if (stat) |st| {
                if (st.kind != .file) {
                    self.save_path = null;
                    try self.env.out.print("\nError opening file: {s} Error message: NotRegularFile\n", .{path});
                    return error.ProcessingFailed;
                }
            }
            if (stat != null) {
                self.save_destination = std.Io.Dir.cwd().realPathFileAlloc(self.env.io, path, self.env.allocator) catch |err| {
                    self.save_path = null;
                    if (err == error.OutOfMemory) return error.OutOfMemory;
                    try self.env.out.print("\nError opening file: {s} Error message: {s}\n", .{ path, @errorName(err) });
                    return error.ProcessingFailed;
                };
            }
            const destination = self.save_destination orelse path;
            self.save_file = std.Io.Dir.cwd().createFileAtomic(self.env.io, destination, .{
                .replace = true,
                .permissions = if (stat) |st| st.permissions else .default_file,
            }) catch |err| {
                self.save_path = null;
                try self.env.out.print("\nError opening file: {s} Error message: {s}\n", .{ path, @errorName(err) });
                return error.ProcessingFailed;
            };
        }
        const file = &self.save_file.?.file;
        const written = if (builtin.target.os.tag == .windows)
            writeWithCrlf(self.env.io, file, bytes)
        else
            file.writeStreamingAll(self.env.io, bytes);
        written catch |err| {
            self.save_path = null;
            try self.env.out.print("\nError writing file: {s} Error message: {s}\n", .{ path, @errorName(err) });
            return error.ProcessingFailed;
        };
    }

    fn save(self: *Report) t.RunError!void {
        const aw = if (self.capture) |*a| a else return;
        try self.writeSaved(aw.written());
        const path = self.save_path orelse return;
        self.save_file.?.replace(self.env.io) catch |err| {
            try self.env.out.print("\nError saving file: {s} Error message: {s}\n", .{ path, @errorName(err) });
            return error.ProcessingFailed;
        };
    }
};

const FileIdentity = struct { device: u64, inode: u128 };

fn fileIdentity(file: std.Io.File) error{StatFailed}!FileIdentity {
    if (builtin.target.os.tag == .windows) {
        const windows = std.os.windows;
        const FileIdInfo = extern struct { volume: u64, id: [16]u8 };
        const api = struct {
            extern "kernel32" fn GetFileInformationByHandleEx(windows.HANDLE, c_int, *FileIdInfo, windows.DWORD) callconv(.winapi) windows.BOOL;
        };
        var info: FileIdInfo = undefined;
        if (api.GetFileInformationByHandleEx(file.handle, 18, &info, @sizeOf(FileIdInfo)) == .FALSE) return error.StatFailed;
        return .{ .device = info.volume, .inode = std.mem.readInt(u128, &info.id, .little) };
    }
    if (builtin.target.os.tag == .linux) {
        const linux = std.os.linux;
        var stat: linux.Statx = undefined;
        while (true) switch (linux.errno(linux.statx(file.handle, "", linux.AT.EMPTY_PATH, .{ .INO = true }, &stat))) {
            .SUCCESS => {
                if (!stat.mask.INO) return error.StatFailed;
                return .{ .device = (@as(u64, stat.dev_major) << 32) | stat.dev_minor, .inode = stat.ino };
            },
            .INTR => continue,
            else => return error.StatFailed,
        };
    }
    var stat: std.c.Stat = undefined;
    const fstat = if (std.posix.lfs64_abi) std.c.fstat64 else std.c.fstat;
    while (true) switch (std.posix.errno(fstat(file.handle, &stat))) {
        .SUCCESS => return .{ .device = @bitCast(@as(i64, @intCast(stat.dev))), .inode = @intCast(stat.ino) },
        .INTR => continue,
        else => return error.StatFailed,
    };
}

fn writeWithCrlf(io: std.Io, f: *std.Io.File, bytes: []const u8) !void {
    var start: usize = 0;
    var i: usize = 0;
    while (i < bytes.len) : (i += 1) {
        if (bytes[i] == '\n' and (i == 0 or bytes[i - 1] != '\r')) {
            if (i > start) try f.writeStreamingAll(io, bytes[start..i]);
            try f.writeStreamingAll(io, "\r\n");
            start = i + 1;
        }
    }
    if (start < bytes.len) try f.writeStreamingAll(io, bytes[start..]);
}

fn testEnv(out: *std.Io.Writer) t.RunEnv {
    return .{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = out,
    };
}

const TestRecords = struct {
    oks: []const bool,
    fail_with: ?t.RunError = null,

    fn run(records: TestRecords, report: *Report) t.RunError!void {
        for (records.oks, 0..) |ok, i| {
            try report.writer().print("record {d}\n", .{i});
            try report.commit(ok);
        }
        if (records.fail_with) |err| return err;
    }
};

test "Report without save path writes records straight to the console" {
    // Arrange
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);

    // Act
    try Report.run(testEnv(&w), null, TestRecords{ .oks = &.{ true, true } }, TestRecords.run);

    // Assert
    try std.testing.expectEqualStrings("record 0\nrecord 1\n", std.Io.Writer.buffered(&w));
}

test "Report keeps going after a failed record and fails the run at the end" {
    // Arrange
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);

    // Act
    const result = Report.run(testEnv(&w), null, TestRecords{ .oks = &.{ false, true } }, TestRecords.run);

    // Assert
    try std.testing.expectError(error.ProcessingFailed, result);
    try std.testing.expectEqualStrings("record 0\nrecord 1\n", std.Io.Writer.buffered(&w));
}

test "Report saves committed records when the body returns an error" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const save_path = "modes_report_save_on_error_probe.txt";
    defer std.Io.Dir.cwd().deleteFile(io, save_path) catch {};
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    const records: TestRecords = .{ .oks = &.{true}, .fail_with = error.WriteFailed };

    // Act
    const result = Report.run(testEnv(&w), save_path, records, TestRecords.run);

    // Assert
    try std.testing.expectError(error.WriteFailed, result);
    try std.testing.expectEqualStrings("record 0\n", std.Io.Writer.buffered(&w));
    const saved = try std.Io.Dir.cwd().readFileAlloc(io, save_path, std.testing.allocator, .limited(64));
    defer std.testing.allocator.free(saved);
    const saved_lf = try std.mem.replaceOwned(u8, std.testing.allocator, saved, "\r\n", "\n");
    defer std.testing.allocator.free(saved_lf);
    try std.testing.expectEqualStrings("record 0\n", saved_lf);
}

test "Report returns a console write failure when saving" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const save_path = "modes_report_console_failure_probe.txt";
    defer std.Io.Dir.cwd().deleteFile(io, save_path) catch {};
    var buf: [1]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);

    // Act
    const result = Report.run(testEnv(&w), save_path, TestRecords{ .oks = &.{true} }, TestRecords.run);

    // Assert
    try std.testing.expectError(error.WriteFailed, result);
}

test "Report streams a large report with a bounded allocator" {
    // Arrange
    const io = std.testing.io;
    const save_path = "modes_report_bounded_memory.txt";
    defer std.Io.Dir.cwd().deleteFile(io, save_path) catch {};
    var storage: [4096]u8 = undefined;
    var allocator: std.heap.FixedBufferAllocator = .init(&storage);
    var out_buf: [256]u8 = undefined;
    var out: std.Io.Writer.Discarding = .init(&out_buf);
    const env: t.RunEnv = .{ .io = io, .allocator = allocator.allocator(), .out = &out.writer };
    const Records = struct {
        fn run(_: void, report: *Report) t.RunError!void {
            for (0..10000) |_| {
                try report.writer().writeAll("record\n");
                try report.commit(true);
            }
        }
    };

    // Act
    try Report.run(env, save_path, {}, Records.run);

    // Assert
    const saved = try std.Io.Dir.cwd().readFileAlloc(io, save_path, std.testing.allocator, .limited(100000));
    defer std.testing.allocator.free(saved);
    try std.testing.expectEqual(@as(usize, 10000), std.mem.count(u8, saved, "record"));
    try std.testing.expect(std.mem.endsWith(u8, saved, if (builtin.target.os.tag == .windows) "record\r\n" else "record\n"));
}

test "Report ignores its temporary file during a walk" {
    // Arrange
    const io = std.testing.io;
    const base = "modes_report_temp_walk";
    const cwd = std.Io.Dir.cwd();
    try cwd.createDir(io, base, .default_dir);
    defer cwd.deleteTree(io, base) catch {};
    var buf: [64]u8 = undefined;
    var out: std.Io.Writer = .fixed(&buf);
    const Records = struct {
        fn run(_: void, report: *Report) t.RunError!void {
            try report.writer().writeAll("record\n");
            try report.commit(true);
            var dir = std.Io.Dir.cwd().openDir(report.env.io, base, .{ .iterate = true }) catch return error.ProcessingFailed;
            defer dir.close(report.env.io);
            var it = dir.iterate();
            var count: usize = 0;
            while (it.next(report.env.io) catch return error.ProcessingFailed) |entry| {
                const path = std.Io.Dir.path.join(report.env.allocator, &.{ base, entry.name }) catch return error.OutOfMemory;
                defer report.env.allocator.free(path);
                if (try report.protectInput(path)) return error.ProcessingFailed;
                count += 1;
            }
            if (count != 1) return error.ProcessingFailed;
        }
    };

    // Act
    try Report.run(testEnv(&out), base ++ "/report.txt", {}, Records.run);

    // Assert
    const saved = try cwd.readFileAlloc(io, base ++ "/report.txt", std.testing.allocator, .limited(64));
    defer std.testing.allocator.free(saved);
    try std.testing.expectEqualStrings(if (builtin.target.os.tag == .windows) "record\r\n" else "record\n", saved);
}
