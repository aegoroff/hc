const std = @import("std");

const c = @cImport({
    @cInclude("sph_tiger.h");
    @cInclude("blake3.h");
});

const tiger_empty_hex = "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3";
const blake3_empty_hex = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";

fn tigerHash(input: []const u8, out: *[24]u8) void {
    var ctx: c.sph_tiger_context = std.mem.zeroes(c.sph_tiger_context);
    c.sph_tiger_init(&ctx);
    if (input.len != 0) c.sph_tiger(&ctx, input.ptr, input.len);
    c.sph_tiger_close(&ctx, out);
}

fn blake3Hash(input: []const u8, out: *[32]u8) void {
    var hasher: c.blake3_hasher = std.mem.zeroes(c.blake3_hasher);
    c.blake3_hasher_init(&hasher);
    if (input.len != 0) c.blake3_hasher_update(&hasher, input.ptr, input.len);
    c.blake3_hasher_finalize(&hasher, out, out.len);
}

pub fn main() void {
    var t: [24]u8 align(8) = undefined;
    tigerHash("", &t);
    std.debug.print("tiger192(\"\") = {x}\n", .{t});

    var b3: [32]u8 align(8) = undefined;
    blake3Hash("", &b3);
    std.debug.print("blake3(\"\") = {x}\n", .{b3});
}

test "tiger192 of empty string matches reference vector" {
    var digest: [24]u8 align(8) = undefined;
    tigerHash("", &digest);
    const got = std.fmt.bytesToHex(digest, .lower);
    try std.testing.expectEqualStrings(tiger_empty_hex, &got);
}

test "blake3 of empty string matches reference vector" {
    var digest: [32]u8 align(8) = undefined;
    blake3Hash("", &digest);
    const got = std.fmt.bytesToHex(digest, .lower);
    try std.testing.expectEqualStrings(blake3_empty_hex, &got);
}
