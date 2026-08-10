//! Dictionary template expansion for brute-force.

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
            const len = std.mem.replacementSize(u8, current, pair[0], pair[1]);
            const replaced = try allocator.alloc(u8, len);
            _ = std.mem.replace(u8, current, pair[0], pair[1], replaced);
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

test "prepareDictionary ASCII" {
    const d = try prepareDictionary(std.testing.allocator, "ASCII");
    defer std.testing.allocator.free(d);
    // '!'..'~' inclusive → 94 printable ASCII bytes.
    try std.testing.expectEqual(@as(usize, 94), d.len);
    try std.testing.expectEqual(@as(u8, '!'), d[0]);
    try std.testing.expectEqual(@as(u8, '~'), d[93]);
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
