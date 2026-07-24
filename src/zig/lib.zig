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

pub const default_seps: []const u8 = "\t\n\x0b\x0c\r ";

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

pub fn htoi(ptr: []const u8) u32 {
    var value: u32 = 0;
    for (ptr) |ch| {
        if (ch >= '0' and ch <= '9') {
            value = (value << 4) + (ch - '0');
        } else if (ch >= 'A' and ch <= 'F') {
            value = (value << 4) + (ch - 'A') + 10;
        } else if (ch >= 'a' and ch <= 'f') {
            value = (value << 4) + (ch - 'a') + 10;
        } else if (value > 0) {
            return value;
        }
    }
    return value;
}

pub fn hexToBytes(str: []const u8, bytes: []u8) void {
    const to = @min(bytes.len, str.len / 2);
    var i: usize = 0;
    while (i < to) : (i += 1) {
        bytes[i] = @intCast(htoi(str[i * 2 .. i * 2 + 2]));
    }
}

pub fn normalizeTime(seconds: f64) Time {
    var result: Time = .{};
    result.total_seconds = seconds;
    const total_u: u64 = @intFromFloat(seconds);

    result.years = @intCast(total_u / 31536000);
    result.days = @intCast((total_u % 31536000) / 86400);
    result.hours = @intCast(((total_u % 31536000) % 86400) / 3600);
    result.minutes = @intCast((total_u % 3600) / 60);
    result.seconds = @floatFromInt((total_u % 3600) % 60);

    const tmp = result.seconds;
    result.seconds += seconds - (@as(f64, @floatFromInt(result.years * 31536000)) +
        @as(f64, @floatFromInt(result.days * 86400)) +
        @as(f64, @floatFromInt(result.hours * 3600)) +
        @as(f64, @floatFromInt(result.minutes * 60)) + result.seconds);
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
    if (builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var ts: linux.timespec = .{ .sec = 0, .nsec = 0 };
        _ = linux.clock_gettime(.MONOTONIC, &ts);
        return @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
    }
    return 0;
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

pub fn countDigitsIn(x: f64) u32 {
    var result: u32 = 0;
    var n: i64 = @intFromFloat(x);
    while (true) {
        result += 1;
        const div = @divTrunc(n, 10);
        n = div;
        if (n <= 0) break;
    }
    return result;
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

pub fn ltrim(str: []u8, seps: []const u8) []u8 {
    const s = if (seps.len == 0) default_seps else seps;
    var i: usize = 0;
    while (i < str.len and std.mem.indexOfScalar(u8, s, str[i]) != null) : (i += 1) {}
    if (i == str.len) {
        str[0] = 0;
        return str[0..0];
    }
    if (i > 0) {
        std.mem.copyForwards(u8, str[0 .. str.len - i], str[i..]);
        str[str.len - i] = 0;
        return str[0 .. str.len - i];
    }
    return str;
}

pub fn rtrim(str: []u8, seps: []const u8) []u8 {
    const s = if (seps.len == 0) default_seps else seps;
    var len = str.len;
    while (len > 0 and std.mem.indexOfScalar(u8, s, str[len - 1]) != null) {
        len -= 1;
        str[len] = 0;
    }
    return str[0..len];
}

pub fn trim(str: []u8, seps: []const u8) []u8 {
    return ltrim(rtrim(str, seps), seps);
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

test "htoi parses hex pairs" {
    try std.testing.expectEqual(@as(u32, 0xAB), htoi("AB"));
    try std.testing.expectEqual(@as(u32, 0xff), htoi("ff"));
    try std.testing.expectEqual(@as(u32, 0x00), htoi("00"));
}

test "hexToBytes converts string" {
    var bytes: [4]u8 = undefined;
    hexToBytes("deadbeef", &bytes);
    try std.testing.expectEqualSlices(u8, &.{ 0xde, 0xad, 0xbe, 0xef }, &bytes);
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

test "getFileName extracts basename" {
    try std.testing.expectEqualStrings("file.txt", getFileName("/path/to/file.txt"));
    try std.testing.expectEqualStrings("file.txt", getFileName("file.txt"));
    try std.testing.expectEqualStrings("f", getFileName("a\\b\\c\\f"));
}

test "trim strips whitespace" {
    var buf = "  hello  ".*;
    const got = trim(&buf, default_seps);
    try std.testing.expectEqualStrings("hello", got);
}

test "countDigitsIn" {
    try std.testing.expectEqual(@as(u32, 1), countDigitsIn(0));
    try std.testing.expectEqual(@as(u32, 3), countDigitsIn(100));
}
