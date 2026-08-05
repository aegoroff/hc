//! Decode a string / byte-string token from the lexer into payload bytes.
//! Ordinary `'…'` / `"…"` are raw. `b'…'` / `b"…"` support `\xNN`, `\\`, `\'`, `\"`, `\n`, `\r`, `\t`.

const std = @import("std");

pub const Error = error{InvalidStringEscape};

/// Decode lexer token `raw` into owned payload bytes.
pub fn decode(allocator: std.mem.Allocator, raw: []const u8) (Error || std.mem.Allocator.Error)![]u8 {
    if (isByteLiteral(raw)) return unescape(allocator, raw[1..]);
    return try allocator.dupe(u8, try stripQuotes(raw));
}

fn isByteLiteral(raw: []const u8) bool {
    return raw.len >= 3 and raw[0] == 'b' and (raw[1] == '\'' or raw[1] == '"');
}

fn stripQuotes(raw: []const u8) Error![]const u8 {
    if (raw.len < 2) return error.InvalidStringEscape;
    const q = raw[0];
    if ((q != '\'' and q != '"') or raw[raw.len - 1] != q) return error.InvalidStringEscape;
    return raw[1 .. raw.len - 1];
}

fn unescape(allocator: std.mem.Allocator, quoted: []const u8) (Error || std.mem.Allocator.Error)![]u8 {
    const inner = try stripQuotes(quoted);
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    try list.ensureTotalCapacity(allocator, inner.len);

    var i: usize = 0;
    while (i < inner.len) {
        if (inner[i] != '\\') {
            try list.append(allocator, inner[i]);
            i += 1;
            continue;
        }
        i += 1;
        if (i >= inner.len) return error.InvalidStringEscape;
        switch (inner[i]) {
            '\\' => try list.append(allocator, '\\'),
            '\'' => try list.append(allocator, '\''),
            '"' => try list.append(allocator, '"'),
            'n' => try list.append(allocator, '\n'),
            'r' => try list.append(allocator, '\r'),
            't' => try list.append(allocator, '\t'),
            'x' => {
                if (i + 2 >= inner.len) return error.InvalidStringEscape;
                const hex = inner[i + 1 .. i + 3];
                const byte = std.fmt.parseInt(u8, hex, 16) catch return error.InvalidStringEscape;
                try list.append(allocator, byte);
                i += 3;
                continue;
            },
            else => return error.InvalidStringEscape,
        }
        i += 1;
    }
    return try list.toOwnedSlice(allocator);
}

test "decode plain string is raw" {
    const a = std.testing.allocator;
    const got = try decode(a, "\"abc\"");
    defer a.free(got);
    try std.testing.expectEqualStrings("abc", got);

    const win = try decode(a, "'c:\\Windows'");
    defer a.free(win);
    try std.testing.expectEqualStrings("c:\\Windows", win);
}

test "decode empty" {
    const a = std.testing.allocator;
    const got = try decode(a, "''");
    defer a.free(got);
    try std.testing.expectEqualStrings("", got);
}

test "decode byte literal hex" {
    const a = std.testing.allocator;
    const dq = try decode(a, "b\"\\xDE\\xAD\\xBE\\xEF\"");
    defer a.free(dq);
    try std.testing.expectEqualSlices(u8, &.{ 0xDE, 0xAD, 0xBE, 0xEF }, dq);

    const sq = try decode(a, "b'\\xDE\\xAD'");
    defer a.free(sq);
    try std.testing.expectEqualSlices(u8, &.{ 0xDE, 0xAD }, sq);
}

test "decode byte literal common escapes" {
    const a = std.testing.allocator;
    const got = try decode(a, "b'a\\n\\r\\t\\\\\\'\\\"b'");
    defer a.free(got);
    try std.testing.expectEqualSlices(u8, &.{ 'a', '\n', '\r', '\t', '\\', '\'', '"', 'b' }, got);
}

test "decode rejects bad byte escapes" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "b\"\\q\""));
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "b\"\\xA\""));
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "b\"\\xZZ\""));
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "b\"\\\""));
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "abc"));
}

test "plain string keeps backslash-x text" {
    const a = std.testing.allocator;
    const got = try decode(a, "\"\\xZZ\"");
    defer a.free(got);
    try std.testing.expectEqualStrings("\\xZZ", got);
}
