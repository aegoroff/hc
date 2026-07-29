const std = @import("std");
const builtin = @import("builtin");

pub const BINARY_THOUSAND: u64 = 1024;
pub const INT64_BITS_COUNT: u8 = 64;

pub const SizeUnit = enum(u8) {
    bytes = 0,
    kbytes = 1,
    mbytes = 2,
    gbytes = 3,
    tbytes = 4,
    pbytes = 5,
    ebytes = 6,
    zbytes = 7,
    ybytes = 8,
    bbytes = 9,
    gpbytes = 10,
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
    "bytes", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb", "Bb", "GPb",
};

var span_seconds: f64 = 0.0;
var timer_start_ns: i128 = 0;

pub fn getProcessorCount() u32 {
    return @intCast(std.Thread.getCpuCount() catch 1);
}

fn ilog(x: u64) u64 {
    var n: u64 = INT64_BITS_COUNT;
    var c: u32 = INT64_BITS_COUNT / 2;
    var v = x;
    while (true) {
        const y = v >> @intCast(c);
        if (y != 0) {
            n -= c;
            v = y;
        }
        if (c == 0) break;
        c >>= 1;
    }
    n -= v >> (INT64_BITS_COUNT - 1);
    return (INT64_BITS_COUNT - 1) - (n - v);
}

pub fn normalizeSize(size: u64) FileSize {
    var result: FileSize = .{};
    result.size_in_bytes = size;
    result.unit = if (size == 0)
        .bytes
    else
        @enumFromInt(@as(u8, @intCast(ilog(size) / ilog(BINARY_THOUSAND))));
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
    const total_u: u64 = if (!std.math.isFinite(seconds) or seconds < 0)
        0
    else if (seconds >= @as(f64, @floatFromInt(CLAMP_SECS)))
        CLAMP_SECS
    else
        @intFromFloat(seconds);
    const SECS_PER_YEAR = 31536000;

    result.years = @intCast(@min(total_u / SECS_PER_YEAR, std.math.maxInt(u32)));
    result.days = @intCast((total_u % SECS_PER_YEAR) / 86400);
    result.hours = @intCast(((total_u % 31536000) % 86400) / 3600);
    result.minutes = @intCast((total_u % 3600) / 60);
    result.seconds = @floatFromInt((total_u % 3600) % 60);

    const tmp = result.seconds;
    // Use u64/f64 for the product — years * SECS_PER_YEAR overflows u32 for long estimates
    // (e.g. "May take approximately: 3000 years …").
    result.seconds += seconds - (@as(f64, @floatFromInt(@as(u64, result.years) * SECS_PER_YEAR)) +
        @as(f64, @floatFromInt(@as(u64, result.days) * 86400)) +
        @as(f64, @floatFromInt(@as(u64, result.hours) * 3600)) +
        @as(f64, @floatFromInt(@as(u64, result.minutes) * 60)) + result.seconds);
    if (result.seconds > 60) {
        result.seconds = tmp;
    }
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
    if (time.years != 0) {
        try w.print("{d} years {d} days {d} hr {d} min {d:.3} sec", .{ time.years, time.days, time.hours, time.minutes, time.seconds });
        return;
    }
    if (time.days != 0) {
        try w.print("{d} days {d} hr {d} min {d:.3} sec", .{ time.days, time.hours, time.minutes, time.seconds });
        return;
    }
    if (time.hours != 0) {
        try w.print("{d} hr {d} min {d:.3} sec", .{ time.hours, time.minutes, time.seconds });
        return;
    }
    if (time.minutes != 0) {
        try w.print("{d} min {d:.3} sec", .{ time.minutes, time.seconds });
        return;
    }
    try w.print("{d:.3} sec", .{time.seconds});
}

pub fn newLine(w: *std.Io.Writer) !void {
    try w.writeAll("\n");
}

fn nowNs() i128 {
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            var ts: linux.timespec = .{ .sec = 0, .nsec = 0 };
            _ = linux.clock_gettime(.MONOTONIC, &ts);
            return @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
        },
        .windows => {
            // Match classic srclib QueryPerformanceCounter path (Zig Io.Clock is
            // unavailable here without an std.Io context).
            const windows = std.os.windows;
            var freq: windows.LARGE_INTEGER = undefined;
            var counter: windows.LARGE_INTEGER = undefined;
            if (!windows.ntdll.RtlQueryPerformanceFrequency(&freq).toBool()) return 0;
            if (!windows.ntdll.RtlQueryPerformanceCounter(&counter).toBool()) return 0;
            if (freq <= 0) return 0;
            return @divTrunc(@as(i128, counter) * std.time.ns_per_s, freq);
        },
        else => {
            // Match std.Io.Threaded nowPosix (std.posix.clock_gettime removed in Zig 0.16).
            var ts: std.posix.timespec = .{ .sec = 0, .nsec = 0 };
            switch (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts))) {
                .SUCCESS => return @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec),
                else => return 0,
            }
        },
    }
}

pub fn startTimer() void {
    timer_start_ns = nowNs();
}

pub fn stopTimer() void {
    const finish = nowNs();
    span_seconds = @as(f64, @floatFromInt(finish - timer_start_ns)) / 1_000_000_000.0;
}

pub fn readElapsedTime() Time {
    return normalizeTime(span_seconds);
}

pub fn getFileName(path: []const u8) []const u8 {
    if (path.len == 0) return path;
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |idx| {
        return path[idx + 1 ..];
    }
    if (std.mem.lastIndexOfScalar(u8, path, '\\')) |idx| {
        return path[idx + 1 ..];
    }
    return path;
}

test "ilog floor(log2)" {
    try std.testing.expectEqual(@as(u64, 0), ilog(1));
    try std.testing.expectEqual(@as(u64, 9), ilog(512));
    try std.testing.expectEqual(@as(u64, 10), ilog(1024));
    try std.testing.expectEqual(@as(u64, 19), ilog(524288));
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

test "startTimer/stopTimer advances on this host" {
    startTimer();
    // Busy-wait until the monotonic clock moves (avoids std.Io sleep).
    const start = nowNs();
    while (nowNs() - start < std.time.ns_per_ms) {}
    stopTimer();
    const elapsed = readElapsedTime();
    try std.testing.expect(elapsed.total_seconds > 0);
}

test "getFileName extracts basename" {
    try std.testing.expectEqualStrings("file.txt", getFileName("/path/to/file.txt"));
    try std.testing.expectEqualStrings("file.txt", getFileName("file.txt"));
    try std.testing.expectEqualStrings("f", getFileName("a\\b\\c\\f"));
}
