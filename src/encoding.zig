//! Encoding helpers ported from src/srclib/encoding.c (BOM detection and UTF-8
//! validation). Full code-page conversion stays Windows/APR-specific and is
//! not required for the Zig hc binary.

const std = @import("std");

pub const Bom = enum {
    unknown,
    utf8,
    utf16le,
    utf16be,
    utf32be,
};

const BomSig = struct {
    bom: Bom,
    signature: []const u8,
};

const boms = [_]BomSig{
    .{ .bom = .utf8, .signature = "\xEF\xBB\xBF" },
    .{ .bom = .utf16le, .signature = "\xFF\xFE" },
    .{ .bom = .utf16be, .signature = "\xFE\xFF" },
    .{ .bom = .utf32be, .signature = "\x00\x00\xFE\xFF" },
};

pub fn detectBomMemory(buffer: []const u8, offset: *usize) Bom {
    for (boms) |b| {
        if (buffer.len >= b.signature.len and std.mem.eql(u8, buffer[0..b.signature.len], b.signature)) {
            offset.* = b.signature.len;
            return b.bom;
        }
    }
    offset.* = 0;
    return .unknown;
}

pub fn isValidUtf8(str: []const u8) bool {
    var i: usize = 0;
    while (i < str.len) {
        const b = str[i];
        var num: usize = undefined;
        var cp: u32 = undefined;
        if ((b & 0x80) == 0x00) {
            cp = b & 0x7F;
            num = 1;
        } else if ((b & 0xE0) == 0xC0) {
            cp = b & 0x1F;
            num = 2;
        } else if ((b & 0xF0) == 0xE0) {
            cp = b & 0x0F;
            num = 3;
        } else if ((b & 0xF8) == 0xF0) {
            cp = b & 0x07;
            num = 4;
        } else {
            return false;
        }
        if (i + num > str.len) return false;
        var j: usize = 1;
        while (j < num) : (j += 1) {
            if ((str[i + j] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (str[i + j] & 0x3F);
        }
        if ((num == 2 and cp < 0x80) or
            (num == 3 and cp < 0x800) or
            (num == 4 and cp < 0x10000) or
            cp > 0x10FFFF or
            (cp >= 0xD800 and cp <= 0xDFFF))
        {
            return false;
        }
        i += num;
    }
    return true;
}

test "DetectBomUtf8" {
    var offset: usize = 0;
    const result = detectBomMemory("\xEF\xBB\xBF\xd1\x82\xd0\xb5\xd1\x81\xd1\x82", &offset);
    try std.testing.expectEqual(Bom.utf8, result);
    try std.testing.expectEqual(@as(usize, 3), offset);
}

test "DetectBomUtf16le" {
    var offset: usize = 0;
    const result = detectBomMemory("\xFF\xFE\x00\x00\x00\x00\x00\xd1\x81\xd1\x82", &offset);
    try std.testing.expectEqual(Bom.utf16le, result);
    try std.testing.expectEqual(@as(usize, 2), offset);
}

test "DetectBomUtf16be" {
    var offset: usize = 0;
    const result = detectBomMemory("\xFE\xFF\x00\x00\x00\x00\x00\xd1\x81\xd1\x82", &offset);
    try std.testing.expectEqual(Bom.utf16be, result);
    try std.testing.expectEqual(@as(usize, 2), offset);
}

test "DetectBomUtf32be" {
    var offset: usize = 0;
    const result = detectBomMemory("\x00\x00\xFE\xFF\x00\x00\x00\xd1\x81\xd1\x82", &offset);
    try std.testing.expectEqual(Bom.utf32be, result);
    try std.testing.expectEqual(@as(usize, 4), offset);
}

test "DetectBomNoBom" {
    var offset: usize = 0;
    const result = detectBomMemory("\xd1\x82\xd0\xb5\xd1\x81\xd1\x82", &offset);
    try std.testing.expectEqual(Bom.unknown, result);
    try std.testing.expectEqual(@as(usize, 0), offset);
}

test "IsValidUtf8Success" {
    try std.testing.expect(isValidUtf8("тест"));
}

test "IsValidUtf8Fail" {
    // Invalid continuation — high bit set without a valid sequence start.
    try std.testing.expect(!isValidUtf8(&.{ 0xC0, 0xAF }));
}
