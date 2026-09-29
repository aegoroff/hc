const std = @import("std");
const builtin = @import("builtin");
const t = @import("modes_types");

/// Output of one file or dir run. Records go to the console as soon as they
/// are committed, are copied into the `-o` save file when one is given, and
/// a failed record makes the whole run fail.
pub const Report = struct {
    env: t.RunEnv,
    capture: ?std.Io.Writer.Allocating,
    save_path: ?[]const u8,
    teed: usize = 0,
    failed: bool = false,

    /// Runs `body(context, &report)`, then writes the save file, even when
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
        body(context, &report) catch |err| {
            report.save() catch {};
            return err;
        };
        try report.save();
        if (report.failed) return error.ProcessingFailed;
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
        const all = aw.writer.buffer[0..aw.writer.end];
        if (self.teed < all.len) {
            self.env.out.writeAll(all[self.teed..]) catch {};
            self.env.out.flush() catch {};
            self.teed = all.len;
        }
    }

    fn save(self: *Report) t.RunError!void {
        const path = self.save_path orelse return;
        const aw = if (self.capture) |*a| a else return;
        try writeSaveFile(self.env, path, aw.writer.buffer[0..aw.writer.end]);
    }
};

fn writeSaveFile(env: t.RunEnv, save_path: []const u8, bytes: []const u8) t.RunError!void {
    var f = std.Io.Dir.cwd().createFile(env.io, save_path, .{}) catch |err| {
        try env.out.print("\nError opening file: {s} Error message: {s}\n", .{ save_path, @errorName(err) });
        return error.ProcessingFailed;
    };
    defer f.close(env.io);
    // On Windows write "\n" as "\r\n" so save-file line endings match the
    // platform convention.
    const written = if (builtin.os.tag == .windows)
        writeWithCrlf(env.io, &f, bytes)
    else
        f.writeStreamingAll(env.io, bytes);
    written catch |err| {
        try env.out.print("\nError writing file: {s} Error message: {s}\n", .{ save_path, @errorName(err) });
        return error.ProcessingFailed;
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
