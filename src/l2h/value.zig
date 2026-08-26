//! Runtime values for the l2h IR (see docs/l2h-semantics.md).

const std = @import("std");
const plan = @import("plan.zig");

/// String payload. `is_digest` marks hash-property results (§5.3): equality / join /
/// orderby use case-insensitive compare when either side is a digest.
pub const Str = struct {
    bytes: []const u8,
    is_digest: bool = false,

    /// §5.3: case-insensitive when either side is a hash-property digest.
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

    /// Same path/limit, new hash-window start (§4.5 `offset(n)`).
    pub fn withOffset(self: FileVal, offset: i64) FileVal {
        return .{ .path = self.path, .limit = self.limit, .offset = offset };
    }

    /// Same path/offset, new max bytes to hash (§4.5 `limit(n)`).
    pub fn withLimit(self: FileVal, limit: i64) FileVal {
        return .{ .path = self.path, .limit = limit, .offset = self.offset };
    }
};

/// Hash restore binding with optional crack knobs (§4.4). Defaults match `hc hash`.
pub const HashVal = struct {
    digest: []const u8,
    /// When null, restore uses `hc`'s default alphabet (digits + a-z + A-Z).
    dictionary: ?[]const u8 = null,
    /// Restore min/max length; `hc hash` defaults. Methods require `n ≥ 1` (§4.4).
    min: i32 = 1,
    max: i32 = 10,
    no_probe: bool = false,

    pub fn withDict(self: HashVal, dictionary: []const u8) HashVal {
        var copy = self;
        copy.dictionary = dictionary;
        return copy;
    }

    pub fn withMin(self: HashVal, min: i32) HashVal {
        var copy = self;
        copy.min = min;
        return copy;
    }

    pub fn withMax(self: HashVal, max: i32) HashVal {
        var copy = self;
        copy.max = max;
        return copy;
    }

    pub fn withNoProbe(self: HashVal) HashVal {
        var copy = self;
        copy.no_probe = true;
        return copy;
    }
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

    /// Same path/skip_errors, new depth limit (§4.6 `tree()` / `tree(n)`).
    pub fn withTree(self: DirVal, max_depth: ?u32) DirVal {
        return .{ .path = self.path, .max_depth = max_depth, .skip_errors = self.skip_errors };
    }

    /// Same path/max_depth, skip unreadable subdirs (§4.6 `skipErrors()`).
    pub fn withSkipErrors(self: DirVal) DirVal {
        return .{ .path = self.path, .max_depth = self.max_depth, .skip_errors = true };
    }
};

pub const Value = union(enum) {
    string: Str,
    file: FileVal,
    dir: DirVal,
    hash: HashVal,
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

    pub fn hashDigest(digest: []const u8) Value {
        return .{ .hash = .{ .digest = digest } };
    }

    /// Range kind if this value can carry builtin properties (§4.3); else `null`.
    pub fn sourceKind(self: Value) ?plan.SourceKind {
        return switch (self) {
            .string => .string,
            .file => .file,
            .dir => .dir,
            .hash => .hash,
            else => null,
        };
    }

    /// Equality for join / group / `==` keys (§5.3). Only int/bool/string;
    /// any tag mismatch is `error.TypeMismatch` (no coercion).
    pub fn eql(self: Value, other: Value) error{TypeMismatch}!bool {
        return switch (self) {
            .int => |x| {
                if (other != .int) return error.TypeMismatch;
                return x == other.int;
            },
            .bool => |x| {
                if (other != .bool) return error.TypeMismatch;
                return x == other.bool;
            },
            .string => |x| {
                if (other != .string) return error.TypeMismatch;
                return x.compare(other.string) == .eq;
            },
            else => error.TypeMismatch,
        };
    }

    /// Total order for `orderby` (§5). Comparable pairs: int, string, bool.
    pub fn compare(self: Value, other: Value) error{TypeMismatch}!std.math.Order {
        if (self == .int and other == .int) return std.math.order(self.int, other.int);
        if (self == .string and other == .string) return self.string.compare(other.string);
        if (self == .bool and other == .bool) {
            return std.math.order(@intFromBool(self.bool), @intFromBool(other.bool));
        }
        return error.TypeMismatch;
    }

    /// Write string/int/bool text into `w` (formatters / sink). Other tags → TypeMismatch.
    pub fn writeScalar(self: Value, w: *std.Io.Writer) (error{TypeMismatch} || std.Io.Writer.Error)!void {
        switch (self) {
            .string => |s| try w.writeAll(s.bytes),
            .int => |n| try w.print("{d}", .{n}),
            .bool => |b| try w.writeAll(if (b) "true" else "false"),
            else => return error.TypeMismatch,
        }
    }

    /// Deep-copy this value into `allocator` (strings, paths, record/seq payloads).
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
            .hash => |h| .{ .hash = .{
                .digest = try allocator.dupe(u8, h.digest),
                .dictionary = if (h.dictionary) |d| try allocator.dupe(u8, d) else null,
                .min = h.min,
                .max = h.max,
                .no_probe = h.no_probe,
            } },
            .int, .bool => self,
            .record => |r| blk: {
                const fields = try allocator.alloc(RecordField, r.fields.len);
                for (r.fields, 0..) |f, i| {
                    // Names must be owned too: script `into` outlives the plan arena
                    // that originally held field names from `select { … }`.
                    fields[i] = .{
                        .name = try allocator.dupe(u8, f.name),
                        .value = try f.value.dupe(allocator),
                    };
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

test "Value.sourceKind maps range-kind values only" {
    try std.testing.expectEqual(@as(?plan.SourceKind, .string), Value.plainStr("x").sourceKind());
    try std.testing.expectEqual(@as(?plan.SourceKind, .file), Value.filePath("a").sourceKind());
    const dir_v: Value = .{ .dir = .{ .path = "d" } };
    const hash_v = Value.hashDigest("00");
    try std.testing.expectEqual(@as(?plan.SourceKind, .dir), dir_v.sourceKind());
    try std.testing.expectEqual(@as(?plan.SourceKind, .hash), hash_v.sourceKind());
    try std.testing.expect((Value{ .int = 1 }).sourceKind() == null);
    try std.testing.expect((Value{ .bool = true }).sourceKind() == null);
}

test "Str.compare digests are case-insensitive; plain strings are not" {
    const dig_a: Str = .{ .bytes = "Ab", .is_digest = true };
    const dig_b: Str = .{ .bytes = "ab", .is_digest = false };
    const plain_a: Str = .{ .bytes = "Ab" };
    const plain_b: Str = .{ .bytes = "ab" };
    try std.testing.expectEqual(std.math.Order.eq, dig_a.compare(dig_b));
    try std.testing.expectEqual(std.math.Order.lt, plain_a.compare(plain_b)); // 'A' < 'a'
}

test "Value.eql and Value.compare for scalars" {
    try std.testing.expect(try Value.eql(.{ .int = 1 }, .{ .int = 1 }));
    try std.testing.expect(!try Value.eql(.{ .int = 1 }, .{ .int = 2 }));
    try std.testing.expectError(error.TypeMismatch, Value.eql(.{ .int = 1 }, .{ .bool = true }));
    try std.testing.expectError(error.TypeMismatch, Value.eql(Value.plainStr("a"), .{ .int = 1 }));
    try std.testing.expectError(error.TypeMismatch, Value.eql(Value.filePath("p"), Value.filePath("p")));

    try std.testing.expectEqual(std.math.Order.lt, try Value.compare(.{ .int = 1 }, .{ .int = 2 }));
    try std.testing.expectEqual(std.math.Order.lt, try Value.compare(.{ .bool = false }, .{ .bool = true }));
    try std.testing.expectEqual(std.math.Order.eq, try Value.compare(Value.plainStr("a"), Value.plainStr("a")));
    try std.testing.expectError(error.TypeMismatch, Value.compare(.{ .int = 1 }, Value.plainStr("a")));
}

test "Value.writeScalar formats string int bool only" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();

    try Value.plainStr("hi").writeScalar(&out.writer);
    try (@as(Value, .{ .int = -42 })).writeScalar(&out.writer);
    try (@as(Value, .{ .bool = true })).writeScalar(&out.writer);
    try std.testing.expectEqualStrings("hi-42true", out.writer.buffered());

    try std.testing.expectError(error.TypeMismatch, Value.filePath("p").writeScalar(&out.writer));
}

test "HashVal with* copy helpers" {
    const h: HashVal = .{ .digest = "aa", .dictionary = "xy", .min = 2, .max = 5, .no_probe = false };
    const d = h.withDict("ab");
    try std.testing.expectEqualStrings("ab", d.dictionary.?);
    try std.testing.expectEqual(@as(i32, 2), d.min);
    const m = h.withMin(3);
    try std.testing.expectEqual(@as(i32, 3), m.min);
    try std.testing.expectEqual(@as(i32, 5), m.max);
    const x = h.withMax(7);
    try std.testing.expectEqual(@as(i32, 7), x.max);
    try std.testing.expect(h.withNoProbe().no_probe);
    try std.testing.expectEqualStrings("xy", h.withMax(7).dictionary.?);
}

test "FileVal and DirVal with* copy helpers" {
    const f: FileVal = .{ .path = "/a", .limit = 10, .offset = 2 };
    try std.testing.expectEqual(@as(i64, 5), f.withOffset(5).offset);
    try std.testing.expectEqual(@as(i64, 10), f.withOffset(5).limit);
    try std.testing.expectEqual(@as(i64, 3), f.withLimit(3).limit);
    try std.testing.expectEqual(@as(i64, 2), f.withLimit(3).offset);

    const d: DirVal = .{ .path = "/d", .max_depth = 0, .skip_errors = false };
    const tree = d.withTree(null);
    try std.testing.expect(tree.max_depth == null);
    try std.testing.expect(!tree.skip_errors);
    const skip = d.withSkipErrors();
    try std.testing.expect(skip.skip_errors);
    try std.testing.expectEqual(@as(?u32, 0), skip.max_depth);
}

test "Value.dupe record owns field names past source arena" {
    // Arrange — names allocated in a short-lived arena (like the plan arena).
    var src_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const src = src_arena.allocator();
    const name = try src.dupe(u8, "path");
    const fields = try src.alloc(RecordField, 1);
    fields[0] = .{ .name = name, .value = Value.plainStr("p") };
    const rec = try src.create(Record);
    rec.* = .{ .fields = fields };

    var dst_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer dst_arena.deinit();

    // Act
    const owned = try (@as(Value, .{ .record = rec })).dupe(dst_arena.allocator());
    src_arena.deinit();

    // Assert — field name must remain readable after source arena is gone.
    try std.testing.expectEqualStrings("path", owned.record.fields[0].name);
    try std.testing.expectEqualStrings("p", owned.record.fields[0].value.string.bytes);
}
