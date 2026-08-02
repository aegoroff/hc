//! Mute stderr around intentional parse/diag noise in l2h tests.

const std = @import("std");
const builtin = @import("builtin");

/// Redirect stderr to /dev/null. Returns a dup'd fd to restore, or -1 on
/// failure / Windows (POSIX-only: std.c.open flag type is invalid under
/// x86_64_win). Cosmetics only — tests assert on return values, not stderr.
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
