//! Dictionary template expansion for brute-force (port of prbf_prepare_dictionary).

const std = @import("std");

pub const DIGITS = "0123456789";
pub const DIGITS_TPL = "0-9";
pub const LOW_CASE = "abcdefghijklmnopqrstuvwxyz";
pub const LOW_CASE_TPL = "a-z";
pub const UPPER_CASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const UPPER_CASE_TPL = "A-Z";
pub const ASCII_TPL = "ASCII";
pub const DEFAULT_ALPHABET = DIGITS ++ LOW_CASE ++ UPPER_CASE;

const ascii_first: u8 = '!';
const ascii_last: u8 = '~';

/// Expand dict templates (`0-9`, `a-z`, `A-Z`, `ASCII`) and dedupe bytes.
/// Caller owns the returned NUL-terminated slice.
pub fn prepareDictionary(allocator: std.mem.Allocator, dict: []const u8) ![:0]u8 {
    if (std.mem.indexOf(u8, dict, ASCII_TPL) != null) {
        const len = @as(usize, ascii_last - ascii_first) + 1;
        const tmp = try allocator.allocSentinel(u8, len, 0);
        var i: usize = 0;
        var sym: u8 = ascii_first;
        while (sym <= ascii_last) : (sym += 1) {
            tmp[i] = sym;
            i += 1;
        }
        return tmp;
    }

    var buf: ?[]u8 = null;
    defer if (buf) |b| allocator.free(b);
    var current: []const u8 = dict;

    inline for (.{
        .{ DIGITS_TPL, DIGITS },
        .{ LOW_CASE_TPL, LOW_CASE },
        .{ UPPER_CASE_TPL, UPPER_CASE },
    }) |pair| {
        if (std.mem.indexOf(u8, current, pair[0]) != null) {
            const replaced = try strReplace(allocator, current, pair[0], pair[1]);
            if (buf) |b| allocator.free(b);
            buf = replaced;
            current = replaced;
        }
    }

    var seen = [_]bool{false} ** 256;
    var unique: usize = 0;
    for (current) |c| {
        if (!seen[c]) {
            seen[c] = true;
            unique += 1;
        }
    }

    const out = try allocator.allocSentinel(u8, unique, 0);
    @memset(seen[0..], false);
    var ir: usize = 0;
    for (current) |c| {
        if (!seen[c]) {
            out[ir] = c;
            ir += 1;
            seen[c] = true;
        }
    }
    return out;
}

fn strReplace(allocator: std.mem.Allocator, orig: []const u8, rep: []const u8, with: []const u8) ![]u8 {
    if (rep.len == 0) return try allocator.dupe(u8, orig);

    var count: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, orig, i, rep)) |pos| {
        count += 1;
        i = pos + rep.len;
    }
    if (count == 0) return try allocator.dupe(u8, orig);

    const result_len = orig.len + count * with.len - count * rep.len;
    const result = try allocator.alloc(u8, result_len);
    var out_i: usize = 0;
    var src_i: usize = 0;
    var left = count;
    while (left > 0) : (left -= 1) {
        const pos = std.mem.indexOfPos(u8, orig, src_i, rep).?;
        const front = pos - src_i;
        @memcpy(result[out_i..][0..front], orig[src_i..][0..front]);
        out_i += front;
        @memcpy(result[out_i..][0..with.len], with);
        out_i += with.len;
        src_i = pos + rep.len;
    }
    const rem = orig.len - src_i;
    @memcpy(result[out_i..][0..rem], orig[src_i..]);
    return result;
}

/// Zero-extend ANSI bytes to UTF-16LE code units (NTLM / use_wide_string path).
pub fn ansiToWide(allocator: std.mem.Allocator, ansi: []const u8) ![]u16 {
    const out = try allocator.alloc(u16, ansi.len);
    for (ansi, 0..) |b, i| {
        out[i] = b;
    }
    return out;
}

/// Collapse wide code units back to ANSI bytes (ASCII dictionary alphabet).
pub fn wideToAnsi(allocator: std.mem.Allocator, wide: []const u16) ![]u8 {
    const out = try allocator.alloc(u8, wide.len);
    for (wide, 0..) |c, i| {
        out[i] = @truncate(c);
    }
    return out;
}

test "prepareDictionary ASCII" {
    const d = try prepareDictionary(std.testing.allocator, "ASCII");
    defer std.testing.allocator.free(d);
    try std.testing.expectEqual(@as(usize, 95), d.len);
    try std.testing.expectEqual(@as(u8, '!'), d[0]);
    try std.testing.expectEqual(@as(u8, '~'), d[94]);
}

test "prepareDictionary digit class" {
    const d = try prepareDictionary(std.testing.allocator, "0-9");
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualStrings(DIGITS, d);
}

test "prepareDictionary mixed dedupe" {
    const d = try prepareDictionary(std.testing.allocator, "0-9abc0");
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualStrings("0123456789abc", d);
}

test "wide roundtrip 123" {
    const w = try ansiToWide(std.testing.allocator, "123");
    defer std.testing.allocator.free(w);
    try std.testing.expectEqual(@as(u16, '1'), w[0]);
    const a = try wideToAnsi(std.testing.allocator, w);
    defer std.testing.allocator.free(a);
    try std.testing.expectEqualStrings("123", a);
}
