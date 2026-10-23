const std = @import("std");
const builtin = @import("builtin");

/// Set Windows console input/output code page to UTF-8 so digests and paths
/// print correctly. No-op on non-Windows.
pub fn setupConsoleUtf8() void {
    if (comptime builtin.target.os.tag != .windows) return;
    const kernel32 = struct {
        extern "kernel32" fn SetConsoleOutputCP(wCodePageID: u32) callconv(.winapi) i32;
        extern "kernel32" fn SetConsoleCP(wCodePageID: u32) callconv(.winapi) i32;
    };
    _ = kernel32.SetConsoleOutputCP(65001);
    _ = kernel32.SetConsoleCP(65001);
}

/// Architecture suffix for copyright / help banners (`hc` and `l2h`).
pub fn archSuffix() []const u8 {
    return switch (builtin.target.cpu.arch) {
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
pub fn productBanner(gpa: std.mem.Allocator, app_name: []const u8) ![]u8 {
    return gpa.print("{s} {s} {s}\n{s}", .{
        app_name,
        productVersion(),
        archSuffix(),
        COPYRIGHT_NOTICE,
    });
}

/// Prints `"\n" ++ banner ++ "\n\n"`.
pub fn printProductBanner(out: *std.Io.Writer, app_name: []const u8) !void {
    try out.print("\n{s} {s} {s}\n{s}\n\n", .{
        app_name,
        productVersion(),
        archSuffix(),
        COPYRIGHT_NOTICE,
    });
}

pub const BINARY_THOUSAND: u64 = 1024;

/// Brute-force alphabet when no dictionary is given (`hc hash`, l2h `Hash.dict`).
pub const DEFAULT_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

pub const SizeUnit = enum(u8) {
    bytes = 0,
    kbytes = 1,
    mbytes = 2,
    gbytes = 3,
    tbytes = 4,
    pbytes = 5,
    ebytes = 6,

    /// Display suffix used by `formatSize`.
    pub fn suffix(self: SizeUnit) []const u8 {
        return switch (self) {
            .bytes => "bytes",
            .kbytes => "Kb",
            .mbytes => "Mb",
            .gbytes => "Gb",
            .tbytes => "Tb",
            .pbytes => "Pb",
            .ebytes => "Eb",
        };
    }
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
    /// The value does not fit `years`; `formatTime` prints it as a lower bound.
    saturated: bool = false,
};

pub fn normalizeSize(size: u64) FileSize {
    if (size == 0) return .{};
    // Clamp to the last unit so the tag stays valid for any BINARY_THOUSAND:
    // an out-of-range @fromBackingInt is a panic in safe builds and UB in fast ones.
    const per_unit = comptime std.math.log2_int(u64, BINARY_THOUSAND);
    const ix: u8 = @min(std.math.log2_int(u64, size) / per_unit, @backingInt(SizeUnit.ebytes));
    const unit: SizeUnit = @fromBackingInt(@intCast(ix));
    return .{
        .unit = unit,
        .size_in_bytes = size,
        .size = if (unit == .bytes) 0 else @as(f64, @floatFromInt(size)) /
            std.math.pow(f64, @floatFromInt(BINARY_THOUSAND), @floatFromInt(ix)),
    };
}

pub fn normalizeTime(seconds: f64) Time {
    var result: Time = .{};
    result.total_seconds = seconds;
    const SECS_PER_YEAR: u64 = 31536000;
    // Long-password estimates (pow(dictlen, passmax) for -x 12+) can exceed
    // what the u32 years field holds, or overflow to +inf (-x 200). Report
    // those as saturated instead of clamping into fields that would then
    // disagree with each other. The bound (~1.35e17 s) is far below 2^63, so
    // @intFromFloat below cannot trap in Debug/ReleaseSafe or be UB in
    // ReleaseFast.
    const max_secs: f64 = @floatFromInt(@as(u64, std.math.maxInt(u32)) * SECS_PER_YEAR);
    if (seconds >= max_secs) {
        result.years = std.math.maxInt(u32);
        result.saturated = true;
        return result;
    }
    // Probe timing can be negative/NaN on a clock failure.
    if (!(seconds >= 0)) return result;
    const total_u: u64 = @intFromFloat(seconds);

    result.years = @intCast(@min(total_u / SECS_PER_YEAR, std.math.maxInt(u32)));
    result.days = @intCast((total_u % SECS_PER_YEAR) / 86400);
    result.hours = @intCast(((total_u % SECS_PER_YEAR) % 86400) / 3600);
    result.minutes = @intCast((total_u % 3600) / 60);
    const whole_secs: f64 = @floatFromInt((total_u % 3600) % 60);
    result.seconds = whole_secs + (seconds - @floor(seconds));
    return result;
}

pub fn formatSize(size: u64, w: *std.Io.Writer) !void {
    const n = normalizeSize(size);
    if (n.unit != .bytes) {
        try w.print("{d:.2} {s} ({d} {s})", .{ n.size, n.unit.suffix(), n.size_in_bytes, SizeUnit.bytes.suffix() });
    } else {
        try w.print("{d} {s}", .{ n.size_in_bytes, SizeUnit.bytes.suffix() });
    }
}

pub fn formatTime(time: Time, w: *std.Io.Writer) !void {
    if (time.saturated) return w.print("more than {d} years", .{time.years});
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
    if (std.mem.startsWith(u8, tok, "--") and std.mem.findScalar(u8, tok, '=') == null) {
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
    gpa: std.mem.Allocator,
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

    const out = try gpa.alloc([:0]const u8, argv.len - merged);
    errdefer gpa.free(out);

    var oi: usize = 0;
    var i: usize = 0;
    while (i < argv.len) {
        if (i + 1 < argv.len and should_attach(argv[i], argv[i + 1])) {
            out[oi] = try gpa.printSentinel("{s}={s}", .{ argv[i], argv[i + 1] }, 0);
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
    saved: if (builtin.target.os.tag == .windows) std.os.windows.HANDLE else std.posix.fd_t,

    pub fn begin() !YazapStdoutRedirect {
        if (builtin.target.os.tag == .windows) {
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
        if (builtin.target.os.tag == .windows) {
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
    // Arrange
    const shorts = [_]u8{ 'q', 'f' };
    const longs = [_][]const u8{ "query", "file" };

    // Act

    // Assert
    try std.testing.expect(isBareNamedOption("-q", &shorts, &longs));
    try std.testing.expect(isBareNamedOption("--query", &shorts, &longs));
    try std.testing.expect(!isBareNamedOption("-q=x", &shorts, &longs));
    try std.testing.expect(!isBareNamedOption("-qx", &shorts, &longs));
    try std.testing.expect(!isBareNamedOption("--query=x", &shorts, &longs));
    try std.testing.expect(!isBareNamedOption("-z", &shorts, &longs));
}

test "trimQuotes strips surrounding quotes" {
    // Arrange

    // Act

    // Assert
    try std.testing.expectEqualStrings("foo", trimQuotes("\"foo\""));
    try std.testing.expectEqualStrings("foo", trimQuotes("'foo'"));
    try std.testing.expectEqualStrings("foo", trimQuotes("foo"));
    try std.testing.expectEqualStrings("", trimQuotes("\"\""));
    try std.testing.expectEqualStrings("foo", trimQuotes("\"'foo'\""));
}

test "normalizeSize bytes" {
    // Arrange

    // Act
    const s = normalizeSize(512);

    // Assert
    try std.testing.expectEqual(SizeUnit.bytes, s.unit);
    try std.testing.expectEqual(@as(u64, 512), s.size_in_bytes);
}

test "normalizeSize Kb" {
    // Arrange

    // Act
    const s = normalizeSize(2048);

    // Assert
    try std.testing.expectEqual(SizeUnit.kbytes, s.unit);
    try std.testing.expectEqual(@as(f64, 2.0), s.size);
}

test "normalizeSize Mb" {
    // Arrange

    // Act
    const s = normalizeSize(5 * 1024 * 1024);

    // Assert
    try std.testing.expectEqual(SizeUnit.mbytes, s.unit);
    try std.testing.expectEqual(@as(f64, 5.0), s.size);
}

test "formatSize small" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatSize(512, &writer);

    // Assert
    try std.testing.expectEqualStrings("512 bytes", std.Io.Writer.buffered(&writer));
}

test "formatSize big" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatSize(1572864, &writer);

    // Assert
    try std.testing.expectEqualStrings("1.50 Mb (1572864 bytes)", std.Io.Writer.buffered(&writer));
}

test "SizeToString KBytesBoundary" {
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatSize(1024, &writer);

    // Assert
    try std.testing.expectEqualStrings("1.00 Kb (1024 bytes)", std.Io.Writer.buffered(&writer));
}

test "SizeToString KBytes" {
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatSize(BINARY_THOUSAND * 2 + 10, &writer);

    // Assert
    try std.testing.expectEqualStrings("2.01 Kb (2058 bytes)", std.Io.Writer.buffered(&writer));
}

test "SizeToString BytesZero" {
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatSize(0, &writer);

    // Assert
    try std.testing.expectEqualStrings("0 bytes", std.Io.Writer.buffered(&writer));
}

test "SizeToString Bytes" {
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatSize(20, &writer);

    // Assert
    try std.testing.expectEqualStrings("20 bytes", std.Io.Writer.buffered(&writer));
}

test "SizeToString MaxValue" {
    // Arrange
    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatSize(std.math.maxInt(u64), &writer);

    // Assert
    try std.testing.expectEqualStrings("16.00 Eb (18446744073709551615 bytes)", std.Io.Writer.buffered(&writer));
}

test "formatTime seconds only" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const t = normalizeTime(3.5);

    // Act
    try formatTime(t, &writer);

    // Assert
    try std.testing.expectEqualStrings("3.500 sec", std.Io.Writer.buffered(&writer));
}

test "formatTime with minutes" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const t = normalizeTime(125.0);

    // Act
    try formatTime(t, &writer);

    // Assert
    try std.testing.expectEqualStrings("2 min 5.000 sec", std.Io.Writer.buffered(&writer));
}

test "ToStringTime BigValueYears" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatTime(normalizeTime(50000001.0), &writer);

    // Assert
    try std.testing.expectEqualStrings("1 years 213 days 16 hr 53 min 21.000 sec", std.Io.Writer.buffered(&writer));
}

test "normalizeTime does not trap on overflow estimates" {
    // Arrange
    // pow(dictlen, passmax) for -x 12+ exceeds maxInt(u64); previously this
    // trapped @intFromFloat in Debug/ReleaseSafe. It must clamp instead.
    const huge = @as(f64, 3.0e21);

    // Act
    const t = normalizeTime(huge);

    // Assert
    try std.testing.expect(t.years >= 100_000_000); // the u32 years field saturates
    try std.testing.expect(t.days < 366);
}

test "normalizeTime clamps +inf like a huge estimate; NaN and negative are zero" {
    // Arrange
    // pow(dictlen, passmax) overflows f64 to +inf for long passwords (-x 200);
    // that is the largest estimate, not an instant one.

    // Act
    const inf = normalizeTime(std.math.inf(f64));
    const huge = normalizeTime(3.0e21);
    const nan = normalizeTime(std.math.nan(f64));
    const neg = normalizeTime(-100.0);

    // Assert
    try std.testing.expect(inf.saturated);
    try std.testing.expect(huge.saturated);
    try std.testing.expectEqual(huge.years, inf.years);
    try std.testing.expectEqual(huge.days, inf.days);
    try std.testing.expect(!nan.saturated);
    try std.testing.expectEqual(@as(u32, 0), nan.years);
    try std.testing.expectEqual(@as(f64, 0), nan.seconds);
    try std.testing.expectEqual(@as(u32, 0), neg.years);
}

test "normalizeTime saturates only past the u32 years range" {
    // Arrange
    const secs_per_year: f64 = 31536000;
    const max_years: f64 = @floatFromInt(std.math.maxInt(u32));

    // Act
    const below = normalizeTime((max_years - 1) * secs_per_year);
    const above = normalizeTime((max_years + 1) * secs_per_year);

    // Assert
    try std.testing.expect(!below.saturated);
    try std.testing.expectEqual(@as(u32, std.math.maxInt(u32) - 1), below.years);
    try std.testing.expectEqual(@as(u32, 0), below.days);
    try std.testing.expect(above.saturated);
    try std.testing.expectEqual(@as(u32, 0), above.days);
}

test "formatTime prints a saturated time as a lower bound" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatTime(normalizeTime(std.math.inf(f64)), &writer);

    // Assert
    try std.testing.expectEqualStrings("more than 4294967295 years", std.Io.Writer.buffered(&writer));
}

test "ToStringTime BigValue" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatTime(normalizeTime(500001.0), &writer);

    // Assert
    try std.testing.expectEqualStrings("5 days 18 hr 53 min 21.000 sec", std.Io.Writer.buffered(&writer));
}

test "ToStringTime Hours" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatTime(normalizeTime(7000.0), &writer);

    // Assert
    try std.testing.expectEqualStrings("1 hr 56 min 40.000 sec", std.Io.Writer.buffered(&writer));
}

test "ToStringTime Minutes" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const time: f64 = 200.0;
    const result = normalizeTime(time);

    // Act
    try formatTime(result, &writer);

    // Assert
    try std.testing.expectEqualStrings("3 min 20.000 sec", std.Io.Writer.buffered(&writer));
    try std.testing.expectEqual(time, result.total_seconds);
}

test "ToStringTime Seconds" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);

    // Act
    try formatTime(normalizeTime(20.0), &writer);

    // Assert
    try std.testing.expectEqualStrings("20.000 sec", std.Io.Writer.buffered(&writer));
}

test "elapsedSince advances on this host" {
    // Arrange
    const io = std.testing.io;
    const t0 = std.Io.Clock.awake.now(io);
    // Busy-wait until the monotonic clock moves (avoids std.Io sleep).
    while (t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds < std.time.ns_per_ms) {}

    // Act
    const elapsed = elapsedSince(io, t0);

    // Assert
    try std.testing.expect(elapsed.total_seconds > 0);
}
