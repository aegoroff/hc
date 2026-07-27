//! Shared `-o` / `--save` tee: capture mode output, stream it to the console,
//! then write the same bytes to the save file (with Windows CRLF translation).

const std = @import("std");
const builtin = @import("builtin");
const t = @import("types.zig");

pub const RunEnv = t.RunEnv;
pub const RunError = t.RunError;

/// Captures writes when `save_path` is set so callers can tee to the real
/// console and persist the same bytes to disk (C file.c / dir.c behaviour).
pub const SaveTee = struct {
    capture: ?std.Io.Writer.Allocating = null,
    teed: usize = 0,
    save_path: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator, save_path: ?[]const u8) SaveTee {
        return .{
            .capture = if (save_path != null) std.Io.Writer.Allocating.init(allocator) else null,
            .save_path = save_path,
        };
    }

    pub fn deinit(self: *SaveTee) void {
        if (self.capture) |*aw| aw.deinit();
        self.capture = null;
    }

    /// Env whose `out` is the capture buffer when saving, otherwise unchanged.
    pub fn sinkEnv(self: *SaveTee, env: RunEnv) RunEnv {
        var sink = env;
        if (self.capture) |*aw| sink.out = &aw.writer;
        return sink;
    }

    /// Copy newly appended capture bytes to the real console and flush.
    pub fn flush(self: *SaveTee, console: *std.Io.Writer) RunError!void {
        const aw = if (self.capture) |*a| a else {
            // No save file: callers already wrote/flushed `env.out` directly.
            return;
        };
        const all = aw.writer.buffer[0..aw.writer.end];
        if (self.teed > all.len) self.teed = 0;
        if (self.teed < all.len) {
            console.writeAll(all[self.teed..]) catch {};
            console.flush() catch {};
            self.teed = all.len;
        }
    }

    /// Persist the full capture to `save_path` (no-op when not capturing).
    pub fn finish(self: *SaveTee, env: RunEnv) void {
        const path = self.save_path orelse return;
        const aw = if (self.capture) |*a| a else return;
        writeSaveFile(env, path, aw.writer.buffer[0..aw.writer.end]);
    }
};

fn writeSaveFile(env: RunEnv, save_path: []const u8, bytes: []const u8) void {
    var f = std.Io.Dir.cwd().createFile(env.io, save_path, .{}) catch {
        env.out.print("\nError opening file: {s} Error message: ", .{save_path}) catch {};
        return;
    };
    defer f.close(env.io);
    // Legacy CRT text mode on Windows translated "\n" -> "\r\n". Mirror that so
    // C# black-box tests (Environment.NewLine) match the save file byte-for-byte.
    if (builtin.os.tag == .windows) {
        writeWithCrlf(env.io, &f, bytes);
    } else {
        f.writeStreamingAll(env.io, bytes) catch {};
    }
}

fn writeWithCrlf(io: std.Io, f: *std.Io.File, bytes: []const u8) void {
    var start: usize = 0;
    var i: usize = 0;
    while (i < bytes.len) : (i += 1) {
        if (bytes[i] == '\n' and (i == 0 or bytes[i - 1] != '\r')) {
            if (i > start) f.writeStreamingAll(io, bytes[start..i]) catch return;
            f.writeStreamingAll(io, "\r\n") catch return;
            start = i + 1;
        }
    }
    if (start < bytes.len) f.writeStreamingAll(io, bytes[start..]) catch {};
}

test "SaveTee without path is a no-op passthrough" {
    var tee = SaveTee.init(std.testing.allocator, null);
    defer tee.deinit();
    try std.testing.expect(tee.capture == null);
    var buf: [16]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try tee.flush(&w);
    tee.finish(.{
        .io = std.Io.Threaded.global_single_threaded.io(),
        .allocator = std.testing.allocator,
        .out = &w,
    });
}
