//! GoogleTest HashTest parity: digest of "123" for every registered algorithm.
const std = @import("std");
const hashes = @import("hashes");

const Case = struct { name: []const u8, expected: []const u8 };

// Expected digests of the ASCII string "123" (upper-case hex), from HashTest.h.
const cases = [_]Case{
    .{ .name = "md5", .expected = "202CB962AC59075B964B07152D234B70" },
    .{ .name = "sha1", .expected = "40BD001563085FC35165329EA1FF5C5ECBDBBEEF" },
    .{ .name = "sha256", .expected = "A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3" },
    .{ .name = "tiger", .expected = "A86807BB96A714FE9B22425893E698334CD71E36B0EEF2BE" },
    .{ .name = "blake3", .expected = "B3D4F8803F7E24B8F389B072E75477CDBCFBE074080FB5E500E53E26E054158E" },
    .{ .name = "md2", .expected = "EF1FEDF5D32EAD6B7AAF687DE4ED1B71" },
    .{ .name = "gost", .expected = "5EF18489617BA2D8D2D7E0DA389AAA4FF022AD01A39512A4FEA1A8C45E439148" },
    .{ .name = "crc32", .expected = "884863D2" },
    .{ .name = "crc32c", .expected = "107B2FB2" },
    .{ .name = "md4", .expected = "C58CDA49F00748A3BC0FCFA511D516CB" },
    .{ .name = "whirlpool", .expected = "344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F" },
};

fn expectHashUpper(name: []const u8, expected_upper: []const u8) !void {
    const h = hashes.getHash(name) orelse return error.UnknownHash;
    var digest: [64]u8 align(8) = std.mem.zeroes([64]u8);
    hashes.compute(h, "123", &digest);
    var hex: [128]u8 = undefined;
    const n = h.hash_length;
    for (digest[0..n], 0..) |b, i| {
        _ = std.fmt.bufPrint(hex[i * 2 ..][0..2], "{X:0>2}", .{b}) catch unreachable;
    }
    try std.testing.expectEqualStrings(expected_upper, hex[0 .. n * 2]);
}

test "HashTest Str123 sample algorithms" {
    for (cases) |c| {
        try expectHashUpper(c.name, c.expected);
    }
}
