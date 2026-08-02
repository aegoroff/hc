const std = @import("std");
const value = @import("value.zig");

/// Record formatters for method calls (semantics §4.7).
/// Same four-space separator as `hc` SFV / checksum output.
pub const PAIR_SEPARATOR = "    ";

pub const Error = error{
    UnknownMethod,
    InvalidMethodArity,
    InvalidMethodReceiver,
    InvalidMethodFields,
    TypeMismatch,
} || std.mem.Allocator.Error || std.Io.Writer.Error;

/// Formatters. `sfv` / `checksum` look up fields **by name** and always emit a
/// fixed order (independent of declaration order in `{…}`):
/// - `sfv`: requires `name` + one other field → `name    digest` (`hc --sfv`)
/// - `checksum`: requires `path` + one other field → `digest    path` (`hc -c`)
pub const Method = enum {
    sfv,
    checksum,
    json,
    json_pretty,
    csv,
    spaced,
    tabbed,
};

/// Look up a formatter by name; `null` if unknown.
pub fn lookup(name: []const u8) ?Method {
    if (std.mem.eql(u8, name, "sfv")) return .sfv;
    if (std.mem.eql(u8, name, "checksum")) return .checksum;
    if (std.mem.eql(u8, name, "json")) return .json;
    if (std.mem.eql(u8, name, "jsonPretty")) return .json_pretty;
    if (std.mem.eql(u8, name, "csv")) return .csv;
    if (std.mem.eql(u8, name, "spaced")) return .spaced;
    if (std.mem.eql(u8, name, "tabbed")) return .tabbed;
    return null;
}

/// Expected argument count for `m` (all formatters are nullary for now).
pub fn arity(m: Method) usize {
    _ = m;
    return 0;
}

/// Label field required by pair formatters (`name` for `sfv`, `path` for `checksum`).
pub fn pairLabelField(m: Method) ?[]const u8 {
    return switch (m) {
        .sfv => "name",
        .checksum => "path",
        else => null,
    };
}

/// Compile-time / runtime check: pair methods need exactly two fields including the label.
pub fn validatePairFields(m: Method, field_names: []const []const u8) Error!void {
    const label = pairLabelField(m) orelse return;
    if (field_names.len != 2) return error.InvalidMethodFields;
    for (field_names) |n| {
        if (std.mem.eql(u8, n, label)) return;
    }
    return error.InvalidMethodFields;
}

fn delimiter(m: Method) ?[]const u8 {
    return switch (m) {
        .csv => ",",
        .spaced => " ",
        .tabbed => "\t",
        .sfv, .checksum, .json, .json_pretty => null,
    };
}

/// Format `rec` with method `m` and evaluated `args`. Returns owned bytes.
pub fn call(
    allocator: std.mem.Allocator,
    m: Method,
    rec: *const value.Record,
    args: []const value.Value,
) Error![]u8 {
    if (args.len != arity(m)) return error.InvalidMethodArity;

    return switch (m) {
        .sfv => try formatSfv(allocator, rec),
        .checksum => try formatChecksum(allocator, rec),
        .csv, .spaced, .tabbed => try joinFields(allocator, rec, delimiter(m).?),
        .json => try formatJson(allocator, rec, false),
        .json_pretty => try formatJson(allocator, rec, true),
    };
}

fn formatSfv(allocator: std.mem.Allocator, rec: *const value.Record) Error![]u8 {
    const pair = try splitLabeledPair(rec, "name");
    return try joinTwo(allocator, pair.label, pair.other);
}

fn formatChecksum(allocator: std.mem.Allocator, rec: *const value.Record) Error![]u8 {
    const pair = try splitLabeledPair(rec, "path");
    return try joinTwo(allocator, pair.other, pair.label);
}

const LabeledPair = struct { label: value.Value, other: value.Value };

fn splitLabeledPair(rec: *const value.Record, label_name: []const u8) Error!LabeledPair {
    if (rec.fields.len != 2) return error.InvalidMethodFields;
    const label_val = rec.get(label_name) orelse return error.InvalidMethodFields;
    const other = if (std.mem.eql(u8, rec.fields[0].name, label_name))
        rec.fields[1].value
    else
        rec.fields[0].value;
    return .{ .label = label_val, .other = other };
}

fn joinTwo(allocator: std.mem.Allocator, a: value.Value, b: value.Value) Error![]u8 {
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    try appendScalar(allocator, &list, a);
    try list.appendSlice(allocator, PAIR_SEPARATOR);
    try appendScalar(allocator, &list, b);
    return try list.toOwnedSlice(allocator);
}

fn joinFields(allocator: std.mem.Allocator, rec: *const value.Record, sep: []const u8) Error![]u8 {
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    for (rec.fields, 0..) |f, i| {
        if (i > 0) try list.appendSlice(allocator, sep);
        try appendScalar(allocator, &list, f.value);
    }
    return try list.toOwnedSlice(allocator);
}

fn appendScalar(allocator: std.mem.Allocator, list: *std.ArrayList(u8), v: value.Value) Error!void {
    switch (v) {
        .string => |s| try list.appendSlice(allocator, s.bytes),
        .int => |n| {
            var buf: [32]u8 = undefined;
            const sl = std.fmt.bufPrint(&buf, "{d}", .{n}) catch unreachable;
            try list.appendSlice(allocator, sl);
        },
        .bool => |b| try list.appendSlice(allocator, if (b) "true" else "false"),
        else => return error.TypeMismatch,
    }
}

/// Whether `m` accepts nested `Record` / `Seq` values (JSON only).
pub fn allowsNestedValues(m: Method) bool {
    return switch (m) {
        .json, .json_pretty => true,
        else => false,
    };
}

fn formatJson(allocator: std.mem.Allocator, rec: *const value.Record, pretty: bool) Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    var w: std.json.Stringify = .{
        .writer = &out.writer,
        .options = .{ .whitespace = if (pretty) .indent_2 else .minified },
    };
    try writeJsonRecord(&w, rec);
    return try out.toOwnedSlice();
}

fn writeJsonRecord(w: *std.json.Stringify, rec: *const value.Record) Error!void {
    try w.beginObject();
    for (rec.fields) |f| {
        try w.objectField(f.name);
        try writeJsonValue(w, f.value);
    }
    try w.endObject();
}

fn writeJsonValue(w: *std.json.Stringify, v: value.Value) Error!void {
    switch (v) {
        .string => |s| try w.write(s.bytes),
        .int => |n| try w.write(n),
        .bool => |b| try w.write(b),
        .record => |rec| try writeJsonRecord(w, rec),
        .seq => |s| {
            try w.beginArray();
            for (s.items) |item| try writeJsonValue(w, item);
            try w.endArray();
        },
        .file, .dir, .hash => return error.TypeMismatch,
    }
}

test "lookup and arity" {
    try std.testing.expectEqual(@as(?Method, .sfv), lookup("sfv"));
    try std.testing.expectEqual(@as(?Method, .checksum), lookup("checksum"));
    try std.testing.expectEqual(@as(?Method, .json), lookup("json"));
    try std.testing.expectEqual(@as(?Method, .json_pretty), lookup("jsonPretty"));
    try std.testing.expect(lookup("Sfv") == null);
    try std.testing.expectEqual(@as(usize, 0), arity(.sfv));
    try std.testing.expectEqual(@as(usize, 0), arity(.json));
}

test "sfv emits name then digest regardless of field order" {
    var fields = [_]value.RecordField{
        .{ .name = "crc32", .value = value.Value.plainStr("00000000") },
        .{ .name = "name", .value = value.Value.plainStr("a.txt") },
    };
    var rec: value.Record = .{ .fields = &fields };
    const s = try call(std.testing.allocator, .sfv, &rec, &.{});
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings("a.txt    00000000", s);
}

test "checksum emits digest then path regardless of field order" {
    var fields = [_]value.RecordField{
        .{ .name = "path", .value = value.Value.plainStr("/tmp/a.txt") },
        .{ .name = "crc32", .value = value.Value.plainStr("00000000") },
    };
    var rec: value.Record = .{ .fields = &fields };
    const c = try call(std.testing.allocator, .checksum, &rec, &.{});
    defer std.testing.allocator.free(c);
    try std.testing.expectEqualStrings("00000000    /tmp/a.txt", c);
}

test "sfv rejects missing name field" {
    var fields = [_]value.RecordField{
        .{ .name = "path", .value = value.Value.plainStr("/tmp/a.txt") },
        .{ .name = "crc32", .value = value.Value.plainStr("00000000") },
    };
    var rec: value.Record = .{ .fields = &fields };
    try std.testing.expectError(error.InvalidMethodFields, call(std.testing.allocator, .sfv, &rec, &.{}));
}

test "json and jsonPretty" {
    var fields = [_]value.RecordField{
        .{ .name = "a", .value = value.Value.plainStr("x") },
        .{ .name = "n", .value = .{ .int = 1 } },
    };
    var rec: value.Record = .{ .fields = &fields };
    const compact = try call(std.testing.allocator, .json, &rec, &.{});
    defer std.testing.allocator.free(compact);
    try std.testing.expectEqualStrings("{\"a\":\"x\",\"n\":1}", compact);

    const pretty = try call(std.testing.allocator, .json_pretty, &rec, &.{});
    defer std.testing.allocator.free(pretty);
    try std.testing.expectEqualStrings("{\n  \"a\": \"x\",\n  \"n\": 1\n}", pretty);
}

test "json nests records and sequences" {
    var inner_fields = [_]value.RecordField{
        .{ .name = "md5", .value = value.Value.plainStr("aa") },
    };
    var inner: value.Record = .{ .fields = &inner_fields };
    var items = [_]value.Value{value.Value.plainStr("x")};
    var seq: value.Seq = .{ .items = &items };
    var outer_fields = [_]value.RecordField{
        .{ .name = "path", .value = value.Value.plainStr("/a") },
        .{ .name = "hashes", .value = .{ .record = &inner } },
        .{ .name = "tags", .value = .{ .seq = &seq } },
    };
    var outer: value.Record = .{ .fields = &outer_fields };
    const s = try call(std.testing.allocator, .json, &outer, &.{});
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(
        "{\"path\":\"/a\",\"hashes\":{\"md5\":\"aa\"},\"tags\":[\"x\"]}",
        s,
    );
}

test "delimited joins" {
    var fields = [_]value.RecordField{
        .{ .name = "a", .value = value.Value.plainStr("one") },
        .{ .name = "b", .value = value.Value.plainStr("two") },
    };
    var rec: value.Record = .{ .fields = &fields };
    const csv = try call(std.testing.allocator, .csv, &rec, &.{});
    defer std.testing.allocator.free(csv);
    try std.testing.expectEqualStrings("one,two", csv);
    const sp = try call(std.testing.allocator, .spaced, &rec, &.{});
    defer std.testing.allocator.free(sp);
    try std.testing.expectEqualStrings("one two", sp);
    const tb = try call(std.testing.allocator, .tabbed, &rec, &.{});
    defer std.testing.allocator.free(tb);
    try std.testing.expectEqualStrings("one\ttwo", tb);
}

test "sfv rejects wrong field count" {
    var fields = [_]value.RecordField{
        .{ .name = "a", .value = value.Value.plainStr("x") },
    };
    var rec: value.Record = .{ .fields = &fields };
    try std.testing.expectError(error.InvalidMethodFields, call(std.testing.allocator, .sfv, &rec, &.{}));
}
