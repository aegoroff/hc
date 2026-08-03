const std = @import("std");

/// Decode a quoted string token from the lexer into payload bytes.
/// Supports `\xNN`, `\\`, `\'`, `\"`, `\n`, `\r`, `\t`.
pub const Error = error{InvalidStringEscape};

pub fn decode(allocator: std.mem.Allocator, raw: []const u8) (Error || std.mem.Allocator.Error)![]u8 {
    if (raw.len < 2) return error.InvalidStringEscape;
    const q = raw[0];
    if ((q != '\'' and q != '"') or raw[raw.len - 1] != q) return error.InvalidStringEscape;

    const inner = raw[1 .. raw.len - 1];
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

test "decode plain string" {
    const a = std.testing.allocator;
    const got = try decode(a, "\"abc\"");
    defer a.free(got);
    try std.testing.expectEqualStrings("abc", got);
}

test "decode empty" {
    const a = std.testing.allocator;
    const got = try decode(a, "''");
    defer a.free(got);
    try std.testing.expectEqualStrings("", got);
}

test "decode hex bytes" {
    const a = std.testing.allocator;
    const got = try decode(a, "\"\\xDE\\xAD\\xBE\\xEF\"");
    defer a.free(got);
    try std.testing.expectEqualSlices(u8, &.{ 0xDE, 0xAD, 0xBE, 0xEF }, got);
}

test "decode common escapes" {
    const a = std.testing.allocator;
    const got = try decode(a, "'a\\n\\r\\t\\\\\\'\\\"b'");
    defer a.free(got);
    try std.testing.expectEqualSlices(u8, &.{ 'a', '\n', '\r', '\t', '\\', '\'', '"', 'b' }, got);
}

test "decode rejects bad escapes" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "\"\\q\""));
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "\"\\xA\""));
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "\"\\xZZ\""));
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "\"\\\""));
    try std.testing.expectError(error.InvalidStringEscape, decode(a, "abc"));
}
