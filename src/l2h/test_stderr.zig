//! Mute / capture stderr around intentional parse/diag noise in l2h tests.

const std = @import("std");
const builtin = @import("builtin");

/// Redirect stderr to /dev/null. Returns a dup'd fd to restore, or -1 on
/// failure / Windows (POSIX-only open flags). Cosmetics for tests that only
/// check return values — use `Capture` when the text itself is asserted.
pub fn mute() c_int {
    if (builtin.os.tag == .windows) return -1;
    const null_fd = std.c.open("/dev/null", .{ .ACCMODE = .WRONLY });
    if (null_fd < 0) return -1;
    const saved = std.c.dup(std.posix.STDERR_FILENO);
    if (saved < 0) {
        _ = std.c.close(null_fd);
        return -1;
    }
    if (std.c.dup2(null_fd, std.posix.STDERR_FILENO) < 0) {
        _ = std.c.close(saved);
        _ = std.c.close(null_fd);
        return -1;
    }
    _ = std.c.close(null_fd);
    return saved;
}

pub fn restore(saved: c_int) void {
    if (builtin.os.tag == .windows) return;
    _ = std.c.dup2(saved, std.posix.STDERR_FILENO);
    _ = std.c.close(saved);
}

/// Redirect stderr so tests can assert on fehler text (parse path goes through
/// C and cannot return `diag.Reported`). POSIX: pipe + dup2. Windows: temp
/// file + PEB `hStdError` (same approach as `YazapStdoutRedirect`).
pub const Capture = struct {
    posix_saved: c_int = -1,
    posix_read: c_int = -1,
    posix_write: c_int = -1,
    win_saved: if (builtin.os.tag == .windows) ?std.os.windows.HANDLE else void =
        if (builtin.os.tag == .windows) null else {},
    win_file: if (builtin.os.tag == .windows) ?std.Io.File else void =
        if (builtin.os.tag == .windows) null else {},
    win_path_buf: if (builtin.os.tag == .windows) [160]u8 else void =
        if (builtin.os.tag == .windows) undefined else {},
    win_path_len: usize = 0,

    pub fn begin() Capture {
        if (comptime builtin.os.tag == .windows) return beginWindows();
        return beginPosix();
    }

    fn beginPosix() Capture {
        var fds: [2]c_int = undefined;
        if (std.c.pipe(&fds) != 0) return .{};
        const saved = std.c.dup(std.posix.STDERR_FILENO);
        if (saved < 0) {
            _ = std.c.close(fds[0]);
            _ = std.c.close(fds[1]);
            return .{};
        }
        if (std.c.dup2(fds[1], std.posix.STDERR_FILENO) < 0) {
            _ = std.c.close(saved);
            _ = std.c.close(fds[0]);
            _ = std.c.close(fds[1]);
            return .{};
        }
        return .{ .posix_saved = saved, .posix_read = fds[0], .posix_write = fds[1] };
    }

    fn beginWindows() Capture {
        if (comptime builtin.os.tag != .windows) unreachable;
        const io = std.testing.io;
        var self: Capture = .{};
        const path = std.fmt.bufPrint(&self.win_path_buf, ".zig-cache/l2h-stderr-{d}", .{
            std.Thread.getCurrentId(),
        }) catch return .{};
        self.win_path_len = path.len;

        const file = std.Io.Dir.cwd().createFile(io, path, .{}) catch return .{};
        const params = std.os.windows.peb().ProcessParameters;
        self.win_saved = params.hStdError;
        params.hStdError = file.handle;
        self.win_file = file;
        return self;
    }

    /// Restore stderr and return captured bytes (caller frees). Empty on setup failure.
    pub fn end(self: *Capture, allocator: std.mem.Allocator) ![]u8 {
        if (comptime builtin.os.tag == .windows) return try self.endWindows(allocator);
        return try self.endPosix(allocator);
    }

    fn endPosix(self: *Capture, allocator: std.mem.Allocator) ![]u8 {
        if (self.posix_saved < 0) return try allocator.dupe(u8, "");
        _ = std.c.dup2(self.posix_saved, std.posix.STDERR_FILENO);
        _ = std.c.close(self.posix_saved);
        self.posix_saved = -1;
        _ = std.c.close(self.posix_write);
        self.posix_write = -1;

        var list: std.ArrayList(u8) = .empty;
        errdefer list.deinit(allocator);
        var buf: [512]u8 = undefined;
        while (true) {
            const n = std.c.read(self.posix_read, @ptrCast(&buf), buf.len);
            if (n <= 0) break;
            try list.appendSlice(allocator, buf[0..@intCast(n)]);
        }
        _ = std.c.close(self.posix_read);
        self.posix_read = -1;
        return try list.toOwnedSlice(allocator);
    }

    fn endWindows(self: *Capture, allocator: std.mem.Allocator) ![]u8 {
        if (comptime builtin.os.tag != .windows) unreachable;
        const file = self.win_file orelse return try allocator.dupe(u8, "");
        const saved = self.win_saved orelse return try allocator.dupe(u8, "");
        const io = std.testing.io;
        const path = self.win_path_buf[0..self.win_path_len];

        std.os.windows.peb().ProcessParameters.hStdError = saved;
        self.win_saved = null;
        self.win_file = null;
        file.close(io);

        const contents = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024)) catch {
            std.Io.Dir.cwd().deleteFile(io, path) catch {};
            return try allocator.dupe(u8, "");
        };
        std.Io.Dir.cwd().deleteFile(io, path) catch {};
        return contents;
    }
};
