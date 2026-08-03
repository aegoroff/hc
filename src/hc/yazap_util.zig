//! Shared yazap workarounds for `hc` and `l2h`.
//!
//! Yazap skips empty argv tokens and treats leading `-` values as options;
//! both CLIs rewrite those into `-opt=value` before parse. Help also goes to
//! stderr by default — redirect it to stdout for pipe-friendly `-h`.

const std = @import("std");
const builtin = @import("builtin");

/// True when `tok` is a negative decimal integer (e.g. "-10"). Tokens that
/// look like options ("--limit", "-l") are rejected.
pub fn isNegativeNumber(tok: []const u8) bool {
    if (tok.len < 2 or tok[0] != '-') return false;
    for (tok[1..]) |c| if (c < '0' or c > '9') return false;
    return true;
}

/// True when `tok` is a bare value-expecting option with no attached value
/// (e.g. `-s` / `--source`, but not `-s=x` or `-sx`).
pub fn isBareNamedOption(tok: []const u8, shorts: []const u8, longs: []const []const u8) bool {
    if (tok.len == 2 and tok[0] == '-') {
        for (shorts) |c| if (tok[1] == c) return true;
        return false;
    }
    if (std.mem.startsWith(u8, tok, "--") and std.mem.indexOfScalar(u8, tok, '=') == null) {
        const name = tok[2..];
        for (longs) |n| if (std.mem.eql(u8, name, n)) return true;
        return false;
    }
    return false;
}

fn attachValue(
    allocator: std.mem.Allocator,
    opt_tok: []const u8,
    val_tok: []const u8,
) ![:0]const u8 {
    const buf = try allocator.allocSentinel(u8, opt_tok.len + 1 + val_tok.len, 0);
    @memcpy(buf[0..opt_tok.len], opt_tok);
    buf[opt_tok.len] = '=';
    @memcpy(buf[opt_tok.len + 1 ..][0..val_tok.len], val_tok);
    return buf;
}

/// Rewrites argv when `should_attach(opt, next)` is true into `-opt=next`.
/// Returns the original slice unchanged when no rewrite is needed.
pub fn normalizeArgv(
    allocator: std.mem.Allocator,
    argv: []const [:0]const u8,
    should_attach: *const fn (opt_tok: []const u8, next_tok: []const u8) bool,
) ![]const [:0]const u8 {
    var merged: usize = 0;
    {
        var i: usize = 0;
        while (i < argv.len) {
            if (i + 1 < argv.len and should_attach(argv[i], argv[i + 1])) {
                merged += 1;
                i += 2;
            } else i += 1;
        }
    }
    if (merged == 0) return argv;

    const out = try allocator.alloc([:0]const u8, argv.len - merged);
    errdefer allocator.free(out);

    var oi: usize = 0;
    var i: usize = 0;
    while (i < argv.len) {
        if (i + 1 < argv.len and should_attach(argv[i], argv[i + 1])) {
            out[oi] = try attachValue(allocator, argv[i], argv[i + 1]);
            oi += 1;
            i += 2;
        } else {
            out[oi] = argv[i];
            oi += 1;
            i += 1;
        }
    }
    return out;
}

/// Point stderr at stdout for the duration of yazap help/diagnostics so
/// pipes (`hc -h | less`) see release-compatible output. On Windows Zig's
/// `File.stderr()` reads the PEB handle, so the redirect mutates that;
/// elsewhere `dup2` remaps fd 2.
pub const YazapStdoutRedirect = struct {
    saved: if (builtin.os.tag == .windows) std.os.windows.HANDLE else std.posix.fd_t,

    pub fn begin() !YazapStdoutRedirect {
        if (builtin.os.tag == .windows) {
            const params = std.os.windows.peb().ProcessParameters;
            const saved = params.hStdError;
            params.hStdError = params.hStdOutput;
            return .{ .saved = saved };
        } else {
            const saved = std.c.dup(std.posix.STDERR_FILENO);
            if (saved < 0) return error.Unexpected;
            if (std.c.dup2(std.posix.STDOUT_FILENO, std.posix.STDERR_FILENO) < 0) {
                _ = std.c.close(saved);
                return error.Unexpected;
            }
            return .{ .saved = saved };
        }
    }

    pub fn restore(self: YazapStdoutRedirect) void {
        if (builtin.os.tag == .windows) {
            std.os.windows.peb().ProcessParameters.hStdError = self.saved;
        } else {
            _ = std.c.dup2(self.saved, std.posix.STDERR_FILENO);
            _ = std.c.close(self.saved);
        }
    }
};

test "isBareNamedOption matches short and long forms" {
    const shorts = [_]u8{ 'q', 'f' };
    const longs = [_][]const u8{ "query", "file" };
    try std.testing.expect(isBareNamedOption("-q", &shorts, &longs));
    try std.testing.expect(isBareNamedOption("--query", &shorts, &longs));
    try std.testing.expect(!isBareNamedOption("-q=x", &shorts, &longs));
    try std.testing.expect(!isBareNamedOption("-qx", &shorts, &longs));
    try std.testing.expect(!isBareNamedOption("--query=x", &shorts, &longs));
    try std.testing.expect(!isBareNamedOption("-z", &shorts, &longs));
}
