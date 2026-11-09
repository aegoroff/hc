//! Runtime values for the l2h IR (see docs/l2h-semantics.md).

const std = @import("std");
const lib = @import("lib");
const plan = @import("plan.zig");

/// String payload. `is_digest` marks hash-property results (§5.3): equality / join /
/// orderby use case-insensitive compare when either side is a digest.
pub const Str = struct {
    bytes: []const u8,
    is_digest: bool = false,
    /// Path text (`f.path`, `f.name`). The sink spells newline and CR as `\n` / `\r`.
    is_path: bool = false,

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
    is_base64: bool = false,

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

    /// Interpret the stored digest as Base64 when restoring.
    pub fn fromBase64(self: HashVal) HashVal {
        var copy = self;
        copy.is_base64 = true;
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

    /// Path or file-name text. Printing escapes newline and CR (§7).
    pub fn pathStr(bytes: []const u8) Value {
        return .{ .string = .{ .bytes = bytes, .is_path = true } };
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
            .string => |s| if (s.is_path)
                try lib.writeLineText(w, s.bytes, false)
            else
                try w.writeAll(s.bytes),
            .int => |n| try w.print("{d}", .{n}),
            .bool => |b| try w.writeAll(if (b) "true" else "false"),
            else => return error.TypeMismatch,
        }
    }

    /// Deep-copy this value into `gpa` (strings, paths, record/seq payloads).
    /// A failed copy frees every allocation it already made.
    pub fn dupe(self: Value, gpa: std.mem.Allocator) std.mem.Allocator.Error!Value {
        return switch (self) {
            .string => |s| .{ .string = .{
                .bytes = try gpa.dupe(u8, s.bytes),
                .is_digest = s.is_digest,
                .is_path = s.is_path,
            } },
            .file => |f| .{ .file = .{
                .path = try gpa.dupe(u8, f.path),
                .limit = f.limit,
                .offset = f.offset,
            } },
            .dir => |d| .{ .dir = .{
                .path = try gpa.dupe(u8, d.path),
                .max_depth = d.max_depth,
                .skip_errors = d.skip_errors,
            } },
            .hash => |h| blk: {
                const digest = try gpa.dupe(u8, h.digest);
                errdefer gpa.free(digest);
                const dictionary = if (h.dictionary) |d| try gpa.dupe(u8, d) else null;
                break :blk .{ .hash = .{
                    .digest = digest,
                    .dictionary = dictionary,
                    .min = h.min,
                    .max = h.max,
                    .no_probe = h.no_probe,
                    .is_base64 = h.is_base64,
                } };
            },
            .int, .bool => self,
            .record => |r| blk: {
                const fields = try gpa.alloc(RecordField, r.fields.len);
                errdefer gpa.free(fields);
                var initialized: usize = 0;
                errdefer for (fields[0..initialized]) |done| {
                    gpa.free(done.name);
                    done.value.free(gpa);
                };
                for (r.fields, 0..) |f, i| {
                    // Names must be owned too: script `into` outlives the plan arena
                    // that originally held field names from `select { … }`.
                    const name = try gpa.dupe(u8, f.name);
                    errdefer gpa.free(name);
                    const copied = try f.value.dupe(gpa);
                    fields[i] = .{ .name = name, .value = copied };
                    initialized += 1;
                }
                const rec = try gpa.create(Record);
                rec.* = .{ .fields = fields };
                break :blk .{ .record = rec };
            },
            .seq => |s| blk: {
                const items = try gpa.alloc(Value, s.items.len);
                errdefer gpa.free(items);
                var initialized: usize = 0;
                errdefer for (items[0..initialized]) |done| done.free(gpa);
                for (s.items, 0..) |item, i| {
                    items[i] = try item.dupe(gpa);
                    initialized += 1;
                }
                const seq = try gpa.create(Seq);
                seq.* = .{ .items = items };
                break :blk .{ .seq = seq };
            },
        };
    }

    /// Release memory from `dupe`. `int` and `bool` are no-ops.
    /// Payloads must be owned by `gpa`; row values that alias a plan or a
    /// script binding must not be freed here.
    pub fn free(self: Value, gpa: std.mem.Allocator) void {
        switch (self) {
            .string => |s| gpa.free(s.bytes),
            .file => |f| gpa.free(f.path),
            .dir => |d| gpa.free(d.path),
            .hash => |h| {
                gpa.free(h.digest);
                if (h.dictionary) |dictionary| gpa.free(dictionary);
            },
            .int, .bool => {},
            .record => |r| {
                for (r.fields) |f| {
                    gpa.free(f.name);
                    f.value.free(gpa);
                }
                gpa.free(r.fields);
                gpa.destroy(r);
            },
            .seq => |s| {
                for (s.items) |item| item.free(gpa);
                gpa.free(s.items);
                gpa.destroy(s);
            },
        }
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

/// One query-row: range variable bindings, plus a shared script scope.
pub const Env = struct {
    /// Row-owned bindings (range, `let`, continuation `into`). Copied by `dupe`.
    map: std.StringHashMapUnmanaged(Value) = .empty,
    /// Persistent script-scope (`into id;`). Not owned. `clone` / `dupe` share
    /// this pointer and do not copy its values into the row.
    script: ?*Env = null,

    pub fn deinit(self: *Env, gpa: std.mem.Allocator) void {
        self.map.deinit(gpa);
    }

    /// Free script-scope entries whose keys and values were allocated with `gpa`,
    /// then free the map. Row envs alias plan keys; use `deinit` for those.
    pub fn deinitOwned(self: *Env, gpa: std.mem.Allocator) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            gpa.free(e.key_ptr.*);
            e.value_ptr.free(gpa);
        }
        self.deinit(gpa);
        self.* = .{};
    }

    pub fn put(self: *Env, gpa: std.mem.Allocator, name: []const u8, value: Value) !void {
        try self.map.put(gpa, name, value);
    }

    /// Row bindings shadow script-scope names.
    pub fn get(self: *const Env, name: []const u8) ?Value {
        if (self.map.get(name)) |v| return v;
        const script = self.script orelse return null;
        return script.map.get(name);
    }

    /// Shallow copy of row bindings: keys and value payloads are shared with `self`.
    /// The script scope is the same pointer. Use when the source env outlives the
    /// copy (e.g. cloning into a row arena from an outer env already persisted).
    pub fn clone(self: *const Env, gpa: std.mem.Allocator) std.mem.Allocator.Error!Env {
        var out: Env = .{ .script = self.script };
        errdefer out.deinit(gpa);
        var it = self.map.iterator();
        while (it.next()) |e| {
            try out.map.put(gpa, e.key_ptr.*, e.value_ptr.*);
        }
        return out;
    }

    /// Deep copy of row bindings: those values are `Value.dupe`'d into `gpa`.
    /// Script-scope values stay shared. Keys still alias the query plan.
    /// Use to freeze an env across row-arena resets.
    pub fn dupe(self: *const Env, gpa: std.mem.Allocator) std.mem.Allocator.Error!Env {
        var out: Env = .{ .script = self.script };
        errdefer out.deinit(gpa);
        var it = self.map.iterator();
        while (it.next()) |e| {
            // Range names live in the query plan; only row values need copying
            // out of the row arena. Script values outlive every row.
            try out.map.put(gpa, e.key_ptr.*, try e.value_ptr.*.dupe(gpa));
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
    // Arrange

    // Act

    // Assert
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
    // Arrange
    const dig_a: Str = .{ .bytes = "Ab", .is_digest = true };
    const dig_b: Str = .{ .bytes = "ab", .is_digest = false };
    const plain_a: Str = .{ .bytes = "Ab" };
    const plain_b: Str = .{ .bytes = "ab" };

    // Act

    // Assert
    try std.testing.expectEqual(std.math.Order.eq, dig_a.compare(dig_b));
    try std.testing.expectEqual(std.math.Order.lt, plain_a.compare(plain_b)); // 'A' < 'a'
}

test "Value.eql and Value.compare for scalars" {
    // Arrange

    // Act

    // Assert
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
    // Arrange
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();

    // Act
    try Value.plainStr("hi").writeScalar(&out.writer);
    try (@as(Value, .{ .int = -42 })).writeScalar(&out.writer);
    try (@as(Value, .{ .bool = true })).writeScalar(&out.writer);

    // Assert
    try std.testing.expectEqualStrings("hi-42true", out.writer.buffered());

    try std.testing.expectError(error.TypeMismatch, Value.filePath("p").writeScalar(&out.writer));

    var path_out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer path_out.deinit();
    try Value.pathStr("a\nb|c").writeScalar(&path_out.writer);
    try std.testing.expectEqualStrings("a\\nb|c", path_out.writer.buffered());
}

test "HashVal with* copy helpers" {
    // Arrange
    const h: HashVal = .{ .digest = "aa", .dictionary = "xy", .min = 2, .max = 5, .no_probe = false };

    // Act
    const d = h.withDict("ab");

    // Assert
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
    // Arrange
    const f: FileVal = .{ .path = "/a", .limit = 10, .offset = 2 };

    // Act

    // Assert
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

fn dupeNestedValue(gpa: std.mem.Allocator) !void {
    var inner_items = [_]Value{
        Value.plainStr("a"),
        Value.plainStr("b"),
    };
    var inner: Seq = .{ .items = &inner_items };
    var fields = [_]RecordField{
        .{ .name = "path", .value = Value.pathStr("a/b") },
        .{ .name = "items", .value = .{ .seq = &inner } },
    };
    var rec: Record = .{ .fields = &fields };
    var items = [_]Value{
        .{ .record = &rec },
        .{ .hash = .{ .digest = "ab", .dictionary = "xyz" } },
        Value.filePath("f"),
    };
    var seq: Seq = .{ .items = &items };

    const owned = try (Value{ .seq = &seq }).dupe(gpa);
    owned.free(gpa);
}

test "Value.dupe frees partial record and seq copies on OOM" {
    // Arrange: `dupeNestedValue` builds a seq of a two-field record (one field
    // is itself a seq) plus a hash and a file.

    // Act / Assert: each allocation fails once; partial names, fields, and items are freed.
    try std.testing.checkAllAllocationFailures(std.testing.allocator, dupeNestedValue, .{});
}

test "Value.dupe HashVal frees digest if dictionary dupe fails" {
    // Arrange
    const h: Value = .{ .hash = .{
        .digest = "aa",
        .dictionary = "xy",
        .min = 1,
        .max = 10,
    } };
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });

    // Act
    const err = h.dupe(failing.allocator());

    // Assert
    try std.testing.expectError(error.OutOfMemory, err);
}

test "Env.dupe shares script values and row bindings shadow them" {
    // Arrange
    var script: Env = .{};
    defer script.deinit(std.testing.allocator);
    const items = try std.testing.allocator.alloc(Value, 1);
    defer std.testing.allocator.free(items);
    items[0] = .{ .int = 7 };
    const seq = try std.testing.allocator.create(Seq);
    defer std.testing.allocator.destroy(seq);
    seq.* = .{ .items = items };
    try script.put(std.testing.allocator, "saved", .{ .seq = seq });

    var row: Env = .{ .script = &script };
    defer row.deinit(std.testing.allocator);
    try row.put(std.testing.allocator, "f", .{ .int = 2 });

    var shadowed: Env = .{ .script = &script };
    defer shadowed.deinit(std.testing.allocator);
    try shadowed.put(std.testing.allocator, "saved", .{ .int = 3 });

    // Act
    var copied = try row.dupe(std.testing.allocator);
    defer copied.deinit(std.testing.allocator);
    var copied_shadow = try shadowed.dupe(std.testing.allocator);
    defer copied_shadow.deinit(std.testing.allocator);

    // Assert
    try std.testing.expect(row.get("saved").?.seq == seq);
    try std.testing.expectEqual(@as(i64, 2), row.get("f").?.int);
    try std.testing.expectEqual(@as(i64, 3), shadowed.get("saved").?.int);
    try std.testing.expect(copied.script == &script);
    try std.testing.expect(copied.get("saved").?.seq == seq);
    try std.testing.expectEqual(@as(i64, 2), copied.get("f").?.int);
    try std.testing.expect(copied_shadow.script == &script);
    try std.testing.expectEqual(@as(i64, 3), copied_shadow.get("saved").?.int);
}

test "Value.free releases a dupe'd seq of records and scalars" {
    // Arrange
    const gpa = std.testing.allocator;
    var fields = [_]RecordField{
        .{ .name = "path", .value = Value.pathStr("a/b") },
    };
    var rec: Record = .{ .fields = &fields };
    var items = [_]Value{
        .{ .record = &rec },
        .{ .hash = .{ .digest = "ab", .dictionary = "xyz" } },
        Value.filePath("f"),
        .{ .dir = .{ .path = "d" } },
        .{ .int = 1 },
        .{ .bool = true },
        Value.plainStr(""),
    };
    var seq: Seq = .{ .items = &items };

    // Act
    const owned = try (Value{ .seq = &seq }).dupe(gpa);
    owned.free(gpa);

    // Assert: `std.testing.allocator` reports anything `free` left behind.
}

test "Value.dupe record owns field names past source arena" {
    // Arrange
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

    // Assert
    try std.testing.expectEqualStrings("path", owned.record.fields[0].name);
    try std.testing.expectEqualStrings("p", owned.record.fields[0].value.string.bytes);
}
