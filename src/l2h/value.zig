const std = @import("std");

/// Runtime values for the l2h IR (see docs/l2h-semantics.md).
/// String payload. `is_digest` marks hash-property results (§5): equality / join /
/// orderby use case-insensitive compare when either side is a digest.
pub const Str = struct {
    bytes: []const u8,
    is_digest: bool = false,

    /// §5: case-insensitive when either side is a hash-property digest.
    pub fn compare(a: Str, b: Str) std.math.Order {
        if (a.is_digest or b.is_digest) return std.ascii.orderIgnoreCase(a.bytes, b.bytes);
        return std.mem.order(u8, a.bytes, b.bytes);
    }
};

/// File binding with optional hash window (§4.5). Defaults match `hc` file mode.
pub const FileVal = struct {
    path: []const u8,
    /// Bytes to hash from `offset`; `maxInt(i64)` means whole file (hc default).
    limit: i64 = std.math.maxInt(i64),
    offset: i64 = 0,
};

/// Directory binding with optional depth-limited enumeration (§3.4 / §4.6).
pub const DirVal = struct {
    path: []const u8,
    /// Max relative directory depth for `from file f in d` (regular files only).
    /// `0` = flat (default); `n` = descend at most `n` levels; `null` = unlimited.
    /// Set by `d.tree()` / `d.tree(n)`, which return a new Dir with this field.
    max_depth: ?u32 = 0,
    /// When true, unreadable subdirectories are skipped during walk (§4.6).
    /// Set by `d.skipErrors()`.
    skip_errors: bool = false,
};

pub const Value = union(enum) {
    string: Str,
    file: FileVal,
    dir: DirVal,
    hash: []const u8,
    int: i64,
    bool: bool,
    record: *Record,
    seq: *Seq,

    pub fn plainStr(bytes: []const u8) Value {
        return .{ .string = .{ .bytes = bytes } };
    }

    pub fn digestStr(bytes: []const u8) Value {
        return .{ .string = .{ .bytes = bytes, .is_digest = true } };
    }

    pub fn filePath(path: []const u8) Value {
        return .{ .file = .{ .path = path } };
    }

    pub fn dupe(self: Value, allocator: std.mem.Allocator) std.mem.Allocator.Error!Value {
        return switch (self) {
            .string => |s| .{ .string = .{
                .bytes = try allocator.dupe(u8, s.bytes),
                .is_digest = s.is_digest,
            } },
            .file => |f| .{ .file = .{
                .path = try allocator.dupe(u8, f.path),
                .limit = f.limit,
                .offset = f.offset,
            } },
            .dir => |d| .{ .dir = .{
                .path = try allocator.dupe(u8, d.path),
                .max_depth = d.max_depth,
                .skip_errors = d.skip_errors,
            } },
            .hash => |h| .{ .hash = try allocator.dupe(u8, h) },
            .int, .bool => self,
            .record => |r| blk: {
                const fields = try allocator.alloc(RecordField, r.fields.len);
                for (r.fields, 0..) |f, i| {
                    // Field names live in the query plan (same as Env keys).
                    fields[i] = .{ .name = f.name, .value = try f.value.dupe(allocator) };
                }
                const rec = try allocator.create(Record);
                rec.* = .{ .fields = fields };
                break :blk .{ .record = rec };
            },
            .seq => |s| blk: {
                const items = try allocator.alloc(Value, s.items.len);
                for (s.items, 0..) |item, i| items[i] = try item.dupe(allocator);
                const seq = try allocator.create(Seq);
                seq.* = .{ .items = items };
                break :blk .{ .seq = seq };
            },
        };
    }
};

pub const RecordField = struct {
    name: []const u8,
    value: Value,
};

pub const Record = struct {
    fields: []RecordField,

    pub fn get(self: *const Record, name: []const u8) ?Value {
        for (self.fields) |f| {
            if (std.mem.eql(u8, f.name, name)) return f.value;
        }
        return null;
    }
};

pub const Seq = struct {
    items: []Value,
};

/// One query-row: range variable bindings.
pub const Env = struct {
    map: std.StringHashMapUnmanaged(Value) = .empty,

    pub fn deinit(self: *Env, allocator: std.mem.Allocator) void {
        self.map.deinit(allocator);
    }

    pub fn put(self: *Env, allocator: std.mem.Allocator, name: []const u8, value: Value) !void {
        try self.map.put(allocator, name, value);
    }

    pub fn get(self: *const Env, name: []const u8) ?Value {
        return self.map.get(name);
    }

    /// Shallow copy of bindings: keys and value payloads are shared with `self`.
    /// Use when the source env outlives the copy (e.g. cloning into a row arena
    /// from an outer env already persisted in the parent allocator).
    pub fn clone(self: *const Env, allocator: std.mem.Allocator) std.mem.Allocator.Error!Env {
        var out: Env = .{};
        errdefer out.deinit(allocator);
        var it = self.map.iterator();
        while (it.next()) |e| {
            try out.map.put(allocator, e.key_ptr.*, e.value_ptr.*);
        }
        return out;
    }

    /// Deep copy of bindings: values are `Value.dupe`'d into `allocator`.
    /// Keys still alias the query plan. Use to freeze an env across row-arena resets.
    pub fn dupe(self: *const Env, allocator: std.mem.Allocator) std.mem.Allocator.Error!Env {
        var out: Env = .{};
        errdefer out.deinit(allocator);
        var it = self.map.iterator();
        while (it.next()) |e| {
            // Range names live in the query plan; only values need copying out of the row arena.
            try out.map.put(allocator, e.key_ptr.*, try e.value_ptr.*.dupe(allocator));
        }
        return out;
    }
};

test "record get by auto-name" {
    // Arrange
    var fields = [_]RecordField{
        .{ .name = "md5", .value = Value.plainStr("abc") },
        .{ .name = "size", .value = .{ .int = 3 } },
    };
    var rec: Record = .{ .fields = &fields };

    // Act
    const got = rec.get("md5").?.string.bytes;
    const missing = rec.get("nope") == null;

    // Assert
    try std.testing.expectEqualStrings("abc", got);
    try std.testing.expect(missing);
}

test "Str.compare digests are case-insensitive; plain strings are not" {
    const dig_a: Str = .{ .bytes = "Ab", .is_digest = true };
    const dig_b: Str = .{ .bytes = "ab", .is_digest = false };
    const plain_a: Str = .{ .bytes = "Ab" };
    const plain_b: Str = .{ .bytes = "ab" };
    try std.testing.expectEqual(std.math.Order.eq, dig_a.compare(dig_b));
    try std.testing.expectEqual(std.math.Order.lt, plain_a.compare(plain_b)); // 'A' < 'a'
}
