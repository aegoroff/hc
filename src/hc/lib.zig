const std = @import("std");
const builtin = @import("builtin");

/// Set Windows console input/output code page to UTF-8 so digests and paths
/// print correctly. No-op on non-Windows.
pub fn setupConsoleUtf8() void {
    if (comptime builtin.os.tag != .windows) return;
    const kernel32 = struct {
        extern "kernel32" fn SetConsoleOutputCP(wCodePageID: u32) callconv(.winapi) i32;
        extern "kernel32" fn SetConsoleCP(wCodePageID: u32) callconv(.winapi) i32;
    };
    _ = kernel32.SetConsoleOutputCP(65001);
    _ = kernel32.SetConsoleCP(65001);
}

/// Architecture suffix for copyright / help banners (`hc` and `l2h`).
/// The C binary hardcoded "x64"; keep that on x86_64 and extend elsewhere.
pub fn archSuffix() []const u8 {
    return switch (builtin.cpu.arch) {
        .x86_64 => "x64",
        .aarch64 => "arm64",
        .x86 => "x86",
        else => "native",
    };
}

/// Application version from `-Dversion=` / build options (shared by `hc` and `l2h`).
pub fn productVersion() []const u8 {
    return @import("build_options").version;
}

pub const COPYRIGHT_NOTICE = "Copyright (C) 2009-2026 Alexander Egorov. All rights reserved.";

/// `"<name> <version> <arch>\nCopyright …"` — yazap app description for `hc` / `l2h`.
pub fn productBanner(allocator: std.mem.Allocator, app_name: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s} {s} {s}\n{s}", .{
        app_name,
        productVersion(),
        archSuffix(),
        COPYRIGHT_NOTICE,
    });
}

/// Prints `"\n" ++ banner ++ "\n\n"` (legacy `hc_print_copyright` layout).
pub fn printProductBanner(out: *std.Io.Writer, app_name: []const u8) !void {
    try out.print("\n{s} {s} {s}\n{s}\n\n", .{
        app_name,
        productVersion(),
        archSuffix(),
        COPYRIGHT_NOTICE,
    });
}

pub const BINARY_THOUSAND: u64 = 1024;

pub const SizeUnit = enum(u8) {
    bytes = 0,
    kbytes = 1,
    mbytes = 2,
    gbytes = 3,
    tbytes = 4,
    pbytes = 5,
    ebytes = 6,
};

pub const FileSize = struct {
    unit: SizeUnit = .bytes,
    size: f64 = 0.0,
    size_in_bytes: u64 = 0,
};

pub const Time = struct {
    years: u32 = 0,
    days: u32 = 0,
    hours: u32 = 0,
    minutes: u32 = 0,
    seconds: f64 = 0.0,
    total_seconds: f64 = 0.0,
};

pub const size_suffixes = [_][]const u8{
    "bytes", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb",
};

pub fn normalizeSize(size: u64) FileSize {
    var result: FileSize = .{};
    result.size_in_bytes = size;
    result.unit = if (size == 0)
        .bytes
    else
        @enumFromInt(@as(u8, @intCast(std.math.log2_int(u64, size) / std.math.log2_int(u64, BINARY_THOUSAND))));
    if (result.unit != .bytes) {
        const u: u8 = @intFromEnum(result.unit);
        result.size = @as(f64, @floatFromInt(size)) / std.math.pow(f64, @as(f64, BINARY_THOUSAND), @floatFromInt(u));
    }
    return result;
}

pub fn normalizeTime(seconds: f64) Time {
    var result: Time = .{};
    result.total_seconds = seconds;
    // Long-password estimates (pow(dictlen, passmax) for -x 12+) routinely
    // exceed maxInt(u64), and probe timing can also be negative/non-finite on a
    // clock failure. Guard @intFromFloat and the u32 years field so neither
    // traps in Debug/ReleaseSafe nor becomes UB in ReleaseFast; the displayed
    // value is clamped instead. Clamp at 2^63 (exactly representable in f64);
    // maxInt(u64) itself is not, so @floatFromInt(maxInt(u64)) rounds above it
    // and @intFromFloat would still trap. 2^63 seconds ≈ 292 million years —
    // far beyond any meaningful estimate.
    const CLAMP_SECS: u64 = @as(u64, 1) << 63;
    const clamp_f: f64 = @floatFromInt(CLAMP_SECS);
    const usable = std.math.isFinite(seconds) and seconds >= 0 and seconds < clamp_f;
    const total_u: u64 = if (usable)
        @intFromFloat(seconds)
    else if (std.math.isFinite(seconds) and seconds >= clamp_f)
        CLAMP_SECS
    else
        0;
    const SECS_PER_YEAR = 31536000;

    result.years = @intCast(@min(total_u / SECS_PER_YEAR, std.math.maxInt(u32)));
    result.days = @intCast((total_u % SECS_PER_YEAR) / 86400);
    result.hours = @intCast(((total_u % 31536000) % 86400) / 3600);
    result.minutes = @intCast((total_u % 3600) / 60);
    const whole_secs: f64 = @floatFromInt((total_u % 3600) % 60);
    const frac: f64 = if (usable) seconds - @floor(seconds) else 0;
    result.seconds = whole_secs + frac;
    return result;
}

pub fn formatSize(size: u64, w: *std.Io.Writer) !void {
    const n = normalizeSize(size);
    if (n.unit != .bytes) {
        try w.print("{d:.2} {s} ({d} {s})", .{ n.size, size_suffixes[@intFromEnum(n.unit)], n.size_in_bytes, size_suffixes[0] });
    } else {
        try w.print("{d} {s}", .{ n.size_in_bytes, size_suffixes[0] });
    }
}

pub fn formatTime(time: Time, w: *std.Io.Writer) !void {
    // Emit from the first non-zero leading field onward; seconds always last.
    const leading = [_]struct { v: u32, unit: []const u8 }{
        .{ .v = time.years, .unit = "years" },
        .{ .v = time.days, .unit = "days" },
        .{ .v = time.hours, .unit = "hr" },
        .{ .v = time.minutes, .unit = "min" },
    };
    var start: usize = 0;
    while (start < leading.len and leading[start].v == 0) start += 1;
    for (leading[start..]) |f| try w.print("{d} {s} ", .{ f.v, f.unit });
    try w.print("{d:.3} sec", .{time.seconds});
}

pub fn newLine(w: *std.Io.Writer) !void {
    try w.writeAll("\n");
}

// --- yazap argv / help workarounds (shared by hc and l2h) -------------------

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

/// Rewrites argv when `should_attach(opt, next)` is true into `-opt=next`.
/// Returns the original slice unchanged when no rewrite is needed.
/// Yazap skips empty argv tokens and treats leading `-` values as options.
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
            out[oi] = try std.fmt.allocPrintSentinel(allocator, "{s}={s}", .{ argv[i], argv[i + 1] }, 0);
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

/// Point stderr at stdout for yazap help/diagnostics so pipes (`hc -h | less`)
/// see release-compatible output. On Windows Zig's `File.stderr()` reads the
/// PEB handle, so the redirect mutates that; elsewhere `dup2` remaps fd 2.
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

/// Elapsed awake-clock time from `start` until now, as a display `Time`.
pub fn elapsedSince(io: std.Io, start: std.Io.Timestamp) Time {
    const finish = std.Io.Clock.awake.now(io);
    const ns = start.durationTo(finish).nanoseconds;
    const secs = @as(f64, @floatFromInt(@as(i128, ns))) / @as(f64, @floatFromInt(std.time.ns_per_s));
    return normalizeTime(secs);
}

/// Strip leading/trailing `'` or `"` (any number of layers).
pub fn trimQuotes(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, "\"'");
}

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

test "trimQuotes strips surrounding quotes" {
    try std.testing.expectEqualStrings("foo", trimQuotes("\"foo\""));
    try std.testing.expectEqualStrings("foo", trimQuotes("'foo'"));
    try std.testing.expectEqualStrings("foo", trimQuotes("foo"));
    try std.testing.expectEqualStrings("", trimQuotes("\"\""));
    try std.testing.expectEqualStrings("foo", trimQuotes("\"'foo'\""));
}

test "normalizeSize bytes" {
    const s = normalizeSize(512);
    try std.testing.expectEqual(SizeUnit.bytes, s.unit);
    try std.testing.expectEqual(@as(u64, 512), s.size_in_bytes);
}

test "normalizeSize Kb" {
    const s = normalizeSize(2048);
    try std.testing.expectEqual(SizeUnit.kbytes, s.unit);
    try std.testing.expectEqual(@as(f64, 2.0), s.size);
}

test "normalizeSize Mb" {
    const s = normalizeSize(5 * 1024 * 1024);
    try std.testing.expectEqual(SizeUnit.mbytes, s.unit);
    try std.testing.expectEqual(@as(f64, 5.0), s.size);
}

test "formatSize small" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatSize(512, &writer);
    try std.testing.expectEqualStrings("512 bytes", std.Io.Writer.buffered(&writer));
}

test "formatSize big" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatSize(1572864, &writer);
    try std.testing.expectEqualStrings("1.50 Mb (1572864 bytes)", std.Io.Writer.buffered(&writer));
}

test "SizeToString KBytesBoundary" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatSize(1024, &writer);
    try std.testing.expectEqualStrings("1.00 Kb (1024 bytes)", std.Io.Writer.buffered(&writer));
}

test "SizeToString KBytes" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatSize(BINARY_THOUSAND * 2 + 10, &writer);
    try std.testing.expectEqualStrings("2.01 Kb (2058 bytes)", std.Io.Writer.buffered(&writer));
}

test "SizeToString BytesZero" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatSize(0, &writer);
    try std.testing.expectEqualStrings("0 bytes", std.Io.Writer.buffered(&writer));
}

test "SizeToString Bytes" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatSize(20, &writer);
    try std.testing.expectEqualStrings("20 bytes", std.Io.Writer.buffered(&writer));
}

test "SizeToString MaxValue" {
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatSize(std.math.maxInt(u64), &writer);
    try std.testing.expectEqualStrings("16.00 Eb (18446744073709551615 bytes)", std.Io.Writer.buffered(&writer));
}

test "formatTime seconds only" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const t = normalizeTime(3.5);
    try formatTime(t, &writer);
    try std.testing.expectEqualStrings("3.500 sec", std.Io.Writer.buffered(&writer));
}

test "formatTime with minutes" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const t = normalizeTime(125.0);
    try formatTime(t, &writer);
    try std.testing.expectEqualStrings("2 min 5.000 sec", std.Io.Writer.buffered(&writer));
}

test "ToStringTime BigValueYears" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatTime(normalizeTime(50000001.0), &writer);
    try std.testing.expectEqualStrings("1 years 213 days 16 hr 53 min 21.000 sec", std.Io.Writer.buffered(&writer));
}

test "normalizeTime does not trap on overflow estimates" {
    // pow(dictlen, passmax) for -x 12+ exceeds maxInt(u64); previously this
    // trapped @intFromFloat in Debug/ReleaseSafe. It must clamp instead.
    const huge = @as(f64, 3.0e21);
    const t = normalizeTime(huge);
    try std.testing.expect(t.years >= 100_000_000); // clamped to ~292 million years
    try std.testing.expect(t.days < 366);
}

test "normalizeTime clamps non-finite and negative" {
    const inf = normalizeTime(std.math.inf(f64));
    try std.testing.expectEqual(@as(u32, 0), inf.years);
    const neg = normalizeTime(-100.0);
    try std.testing.expectEqual(@as(u32, 0), neg.years);
}

test "ToStringTime BigValue" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatTime(normalizeTime(500001.0), &writer);
    try std.testing.expectEqualStrings("5 days 18 hr 53 min 21.000 sec", std.Io.Writer.buffered(&writer));
}

test "ToStringTime Hours" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatTime(normalizeTime(7000.0), &writer);
    try std.testing.expectEqualStrings("1 hr 56 min 40.000 sec", std.Io.Writer.buffered(&writer));
}

test "ToStringTime Minutes" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const time: f64 = 200.0;
    const result = normalizeTime(time);
    try formatTime(result, &writer);
    try std.testing.expectEqualStrings("3 min 20.000 sec", std.Io.Writer.buffered(&writer));
    try std.testing.expectEqual(time, result.total_seconds);
}

test "ToStringTime Seconds" {
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try formatTime(normalizeTime(20.0), &writer);
    try std.testing.expectEqualStrings("20.000 sec", std.Io.Writer.buffered(&writer));
}

test "elapsedSince advances on this host" {
    const io = std.testing.io;
    const t0 = std.Io.Clock.awake.now(io);
    // Busy-wait until the monotonic clock moves (avoids std.Io sleep).
    while (t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds < std.time.ns_per_ms) {}
    const elapsed = elapsedSince(io, t0);
    try std.testing.expect(elapsed.total_seconds > 0);
}
