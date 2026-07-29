const std = @import("std");

/// Runtime values for the l2h IR (see docs/l2h-semantics.md).

pub const Value = union(enum) {
    string: []const u8,
    file: []const u8,
    dir: []const u8,
    hash: []const u8,
    int: i64,
    bool: bool,
    record: *Record,
    seq: *Seq,

    pub fn kindName(self: Value) []const u8 {
        return @tagName(self);
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

    pub fn clone(self: *const Env, allocator: std.mem.Allocator) !Env {
        var out: Env = .{};
        var it = self.map.iterator();
        while (it.next()) |e| {
            try out.map.put(allocator, e.key_ptr.*, e.value_ptr.*);
        }
        return out;
    }
};

test "record get by auto-name" {
    // Arrange
    var fields = [_]RecordField{
        .{ .name = "md5", .value = .{ .string = "abc" } },
        .{ .name = "size", .value = .{ .int = 3 } },
    };
    var rec: Record = .{ .fields = &fields };

    // Act
    const got = rec.get("md5").?.string;
    const missing = rec.get("nope") == null;

    // Assert
    try std.testing.expectEqualStrings("abc", got);
    try std.testing.expect(missing);
}
