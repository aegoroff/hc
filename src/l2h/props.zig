//! Builtin property access on a range-kind receiver (semantics §4.3).
//! Record fields are not listed here — they are resolved by name on the value.

const std = @import("std");
const hashes = @import("hashes");
const plan = @import("plan.zig");

pub const Access = enum {
    /// Absolute or relative path of the file / dir / hash source.
    path,
    /// File basename only (no directory) — for SFV-style output (§4.3 / §4.7).
    name,
    /// Byte length of the file (or string).
    size,
    /// File hash window start (semantics §4.5).
    offset,
    /// File hash window length (semantics §4.5).
    limit,
    /// Whether the file can be opened and stated as a regular file (§4.3).
    readable,
    /// `prop` is a known hash algorithm name.
    hash_algo,
};

/// Look up a builtin property for `recv`. `null` means unknown/disallowed.
pub fn lookup(recv: plan.SourceKind, prop: []const u8) ?Access {
    return switch (recv) {
        .file => {
            if (std.mem.eql(u8, prop, "path")) return .path;
            if (std.mem.eql(u8, prop, "name")) return .name;
            if (std.mem.eql(u8, prop, "size")) return .size;
            if (std.mem.eql(u8, prop, "offset")) return .offset;
            if (std.mem.eql(u8, prop, "limit")) return .limit;
            if (std.mem.eql(u8, prop, "readable")) return .readable;
            if (hashes.getHash(prop) != null) return .hash_algo;
            return null;
        },
        .string => {
            if (std.mem.eql(u8, prop, "size")) return .size;
            if (hashes.getHash(prop) != null) return .hash_algo;
            return null;
        },
        .hash => {
            if (hashes.getHash(prop) != null) return .hash_algo;
            return null;
        },
        .dir => {
            if (std.mem.eql(u8, prop, "path")) return .path;
            return null;
        },
    };
}

test "lookup matches semantics catalog for range kinds" {
    // Arrange / Act / Assert
    try std.testing.expectEqual(@as(?Access, .path), lookup(.file, "path"));
    try std.testing.expectEqual(@as(?Access, .name), lookup(.file, "name"));
    try std.testing.expectEqual(@as(?Access, .size), lookup(.file, "size"));
    try std.testing.expectEqual(@as(?Access, .offset), lookup(.file, "offset"));
    try std.testing.expectEqual(@as(?Access, .limit), lookup(.file, "limit"));
    try std.testing.expectEqual(@as(?Access, .readable), lookup(.file, "readable"));
    try std.testing.expectEqual(@as(?Access, .hash_algo), lookup(.file, "md5"));
    try std.testing.expect(lookup(.file, "nope") == null);

    try std.testing.expectEqual(@as(?Access, .size), lookup(.string, "size"));
    try std.testing.expectEqual(@as(?Access, .hash_algo), lookup(.string, "sha1"));
    try std.testing.expect(lookup(.string, "path") == null);
    try std.testing.expect(lookup(.string, "name") == null);
    try std.testing.expect(lookup(.string, "limit") == null);
    try std.testing.expect(lookup(.string, "offset") == null);
    try std.testing.expect(lookup(.string, "readable") == null);

    try std.testing.expectEqual(@as(?Access, .hash_algo), lookup(.hash, "md5"));
    try std.testing.expect(lookup(.hash, "size") == null);

    try std.testing.expectEqual(@as(?Access, .path), lookup(.dir, "path"));
    try std.testing.expect(lookup(.dir, "tree") == null);
    try std.testing.expect(lookup(.dir, "size") == null);
    try std.testing.expect(lookup(.file, "tree") == null);
}
