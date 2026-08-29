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
    /// Restore alphabet on `Hash` (§4.4).
    hash_dict,
    /// Restore min length on `Hash` (§4.4).
    hash_min,
    /// Restore max length on `Hash` (§4.4).
    hash_max,
    /// Whether restore skips the timing probe on `Hash` (§4.4).
    hash_no_probe,
    /// `prop` is a known hash algorithm name.
    hash_algo,
};

const FILE_PROPS = std.StaticStringMap(Access).initComptime(.{
    .{ "path", .path },
    .{ "name", .name },
    .{ "size", .size },
    .{ "offset", .offset },
    .{ "limit", .limit },
    .{ "readable", .readable },
});

const STRING_PROPS = std.StaticStringMap(Access).initComptime(.{
    .{ "size", .size },
});

const DIR_PROPS = std.StaticStringMap(Access).initComptime(.{
    .{ "path", .path },
});

const HASH_PROPS = std.StaticStringMap(Access).initComptime(.{
    .{ "dict", .hash_dict },
    .{ "min", .hash_min },
    .{ "max", .hash_max },
    .{ "noProbe", .hash_no_probe },
});

/// Look up a builtin property for `recv`. `null` means unknown/disallowed.
pub fn lookup(recv: plan.SourceKind, prop: []const u8) ?Access {
    const named: ?Access = switch (recv) {
        .file => FILE_PROPS.get(prop),
        .string => STRING_PROPS.get(prop),
        .dir => DIR_PROPS.get(prop),
        .hash => HASH_PROPS.get(prop),
    };
    if (named) |a| return a;
    return switch (recv) {
        .file, .string, .hash => if (hashes.getHash(prop) != null) .hash_algo else null,
        .dir => null,
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

    try std.testing.expectEqual(@as(?Access, .hash_dict), lookup(.hash, "dict"));
    try std.testing.expectEqual(@as(?Access, .hash_min), lookup(.hash, "min"));
    try std.testing.expectEqual(@as(?Access, .hash_max), lookup(.hash, "max"));
    try std.testing.expectEqual(@as(?Access, .hash_no_probe), lookup(.hash, "noProbe"));
    try std.testing.expectEqual(@as(?Access, .hash_algo), lookup(.hash, "md5"));
    try std.testing.expect(lookup(.hash, "size") == null);
    try std.testing.expect(lookup(.hash, "offset") == null);

    try std.testing.expectEqual(@as(?Access, .path), lookup(.dir, "path"));
    try std.testing.expect(lookup(.dir, "tree") == null);
    try std.testing.expect(lookup(.dir, "size") == null);
    try std.testing.expect(lookup(.file, "tree") == null);
}
