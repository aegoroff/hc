const std = @import("std");
const hashes = @import("hashes");
const plan = @import("plan.zig");
const value = @import("value.zig");

/// Builtin property access on a range-kind receiver (semantics §4.3).
/// Record fields are not listed here — they are resolved by name on the value.
pub const Access = enum {
    path,
    size,
    /// `prop` is a known hash algorithm name.
    hash_algo,
};

pub const ResultKind = enum { string, int };

pub fn resultKind(access: Access) ResultKind {
    return switch (access) {
        .path, .hash_algo => .string,
        .size => .int,
    };
}

/// Range kind of a runtime value, if it can carry builtin properties.
pub fn ofValue(v: value.Value) ?plan.SourceKind {
    return switch (v) {
        .string => .string,
        .file => .file,
        .dir => .dir,
        .hash => .hash,
        else => null,
    };
}

/// Look up a builtin property for `recv`. `null` means unknown/disallowed.
pub fn lookup(recv: plan.SourceKind, prop: []const u8) ?Access {
    return switch (recv) {
        .file => {
            if (std.mem.eql(u8, prop, "path")) return .path;
            if (std.mem.eql(u8, prop, "size")) return .size;
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
    try std.testing.expectEqual(@as(?Access, .size), lookup(.file, "size"));
    try std.testing.expectEqual(@as(?Access, .hash_algo), lookup(.file, "md5"));
    try std.testing.expect(lookup(.file, "nope") == null);

    try std.testing.expectEqual(@as(?Access, .size), lookup(.string, "size"));
    try std.testing.expectEqual(@as(?Access, .hash_algo), lookup(.string, "sha1"));
    try std.testing.expect(lookup(.string, "path") == null);

    try std.testing.expectEqual(@as(?Access, .hash_algo), lookup(.hash, "md5"));
    try std.testing.expect(lookup(.hash, "size") == null);

    try std.testing.expectEqual(@as(?Access, .path), lookup(.dir, "path"));
    try std.testing.expect(lookup(.dir, "size") == null);

    try std.testing.expectEqual(ResultKind.string, resultKind(.path));
    try std.testing.expectEqual(ResultKind.int, resultKind(.size));
    try std.testing.expectEqual(ResultKind.string, resultKind(.hash_algo));
}

test "ofValue maps range-kind values only" {
    try std.testing.expectEqual(@as(?plan.SourceKind, .string), ofValue(value.Value.plainStr("x")));
    try std.testing.expectEqual(@as(?plan.SourceKind, .file), ofValue(.{ .file = "a" }));
    try std.testing.expectEqual(@as(?plan.SourceKind, .dir), ofValue(.{ .dir = "d" }));
    try std.testing.expectEqual(@as(?plan.SourceKind, .hash), ofValue(.{ .hash = "00" }));
    try std.testing.expect(ofValue(.{ .int = 1 }) == null);
    try std.testing.expect(ofValue(.{ .bool = true }) == null);
}
