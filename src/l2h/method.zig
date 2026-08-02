const std = @import("std");
const hashes = @import("hashes");
const value = @import("value.zig");

/// Method catalog for `recv.method(args…)` — Record formatters (§4.7) and
/// hash-check on File/String (§4.8). Analogous to `props.zig` for properties.
/// Receiver may be an identifier or a record literal `{…}`.
/// Same four-space separator as `hc` SFV / checksum output.
pub const PAIR_SEPARATOR = "    ";

pub const Error = error{
    UnknownMethod,
    InvalidMethodArity,
    InvalidMethodReceiver,
    InvalidMethodFields,
    TypeMismatch,
} || std.mem.Allocator.Error || std.Io.Writer.Error;

/// Record formatters. `sfv` / `checksum` look up fields **by name** and always
/// emit a fixed order (independent of declaration order in `{…}`):
/// - `sfv`: requires `name` + one other field → `name    digest` (`hc --sfv`)
/// - `checksum`: requires `path` + one other field → `digest    path` (`hc -c`)
pub const Formatter = enum {
    sfv,
    checksum,
    json,
    json_pretty,
    csv,
    spaced,
    tabbed,
};

/// Resolved method kind. Formatters take precedence over hash-check if names collide.
pub const Kind = union(enum) {
    formatter: Formatter,
    /// Algorithm name is the call's method name (`m.name`); arity 1, result `Bool`.
    hash_check,
};

pub const ResultKind = enum { string, bool };

/// Look up a method by name; `null` if unknown.
pub fn lookup(name: []const u8) ?Kind {
    if (lookupFormatter(name)) |f| return .{ .formatter = f };
    if (hashes.getHash(name) != null) return .hash_check;
    return null;
}

fn lookupFormatter(name: []const u8) ?Formatter {
    if (std.mem.eql(u8, name, "sfv")) return .sfv;
    if (std.mem.eql(u8, name, "checksum")) return .checksum;
    if (std.mem.eql(u8, name, "json")) return .json;
    if (std.mem.eql(u8, name, "jsonPretty")) return .json_pretty;
    if (std.mem.eql(u8, name, "csv")) return .csv;
    if (std.mem.eql(u8, name, "spaced")) return .spaced;
    if (std.mem.eql(u8, name, "tabbed")) return .tabbed;
    return null;
}

pub fn arity(k: Kind) usize {
    return switch (k) {
        .formatter => 0,
        .hash_check => 1,
    };
}

pub fn resultKind(k: Kind) ResultKind {
    return switch (k) {
        .formatter => .string,
        .hash_check => .bool,
    };
}

/// Label field required by pair formatters (`name` for `sfv`, `path` for `checksum`).
pub fn pairLabelField(f: Formatter) ?[]const u8 {
    return switch (f) {
        .sfv => "name",
        .checksum => "path",
        else => null,
    };
}

/// Compile-time / runtime check: pair methods need exactly two fields including the label.
pub fn validatePairFields(f: Formatter, field_names: []const []const u8) Error!void {
    const label = pairLabelField(f) orelse return;
    if (field_names.len != 2) return error.InvalidMethodFields;
    for (field_names) |n| {
        if (std.mem.eql(u8, n, label)) return;
    }
    return error.InvalidMethodFields;
}

/// Whether `f` accepts nested `Record` / `Seq` values (JSON only).
pub fn allowsNestedValues(f: Formatter) bool {
    return switch (f) {
        .json, .json_pretty => true,
        else => false,
    };
}

fn delimiter(f: Formatter) ?[]const u8 {
    return switch (f) {
        .csv => ",",
        .spaced => " ",
        .tabbed => "\t",
        .sfv, .checksum, .json, .json_pretty => null,
    };
}

/// Format `rec` with formatter `f` and evaluated `args`. Returns owned bytes.
pub fn callFormatter(
    allocator: std.mem.Allocator,
    f: Formatter,
    rec: *const value.Record,
    args: []const value.Value,
) Error![]u8 {
    if (args.len != 0) return error.InvalidMethodArity;

    return switch (f) {
        .sfv => try formatSfv(allocator, rec),
        .checksum => try formatChecksum(allocator, rec),
        .csv, .spaced, .tabbed => try joinFields(allocator, rec, delimiter(f).?),
        .json => try formatJson(allocator, rec, false),
        .json_pretty => try formatJson(allocator, rec, true),
    };
}

/// Case-insensitive digest equality for hash-check (§4.8 / §5.2).
pub fn digestsEqual(actual_hex: []const u8, expected: value.Str) bool {
    const actual = value.Str{ .bytes = actual_hex, .is_digest = true };
    return cmpDigest(actual, expected) == .eq;
}

fn cmpDigest(a: value.Str, b: value.Str) std.math.Order {
    // Same rule as interpret.cmpStr when either side is a digest.
    if (a.is_digest or b.is_digest) {
        const n = @min(a.bytes.len, b.bytes.len);
        for (0..n) |i| {
            const ca = std.ascii.toLower(a.bytes[i]);
            const cb = std.ascii.toLower(b.bytes[i]);
            if (ca < cb) return .lt;
            if (ca > cb) return .gt;
        }
        return std.math.order(a.bytes.len, b.bytes.len);
    }
    return std.mem.order(u8, a.bytes, b.bytes);
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

test "lookup kind covers formatters and hash-check" {
    try std.testing.expectEqual(Kind{ .formatter = .sfv }, lookup("sfv").?);
    try std.testing.expectEqual(Kind{ .formatter = .checksum }, lookup("checksum").?);
    try std.testing.expectEqual(Kind{ .formatter = .json }, lookup("json").?);
    try std.testing.expectEqual(Kind{ .formatter = .json_pretty }, lookup("jsonPretty").?);
    try std.testing.expect(lookup("Sfv") == null);
    try std.testing.expectEqual(@as(?Kind, .hash_check), lookup("md5"));
    try std.testing.expectEqual(@as(?Kind, .hash_check), lookup("sha1"));
    try std.testing.expect(lookup("nope") == null);

    try std.testing.expectEqual(@as(usize, 0), arity(.{ .formatter = .sfv }));
    try std.testing.expectEqual(@as(usize, 1), arity(.hash_check));
    try std.testing.expectEqual(ResultKind.string, resultKind(.{ .formatter = .json }));
    try std.testing.expectEqual(ResultKind.bool, resultKind(.hash_check));
}

test "sfv emits name then digest regardless of field order" {
    var fields = [_]value.RecordField{
        .{ .name = "crc32", .value = value.Value.plainStr("00000000") },
        .{ .name = "name", .value = value.Value.plainStr("a.txt") },
    };
    var rec: value.Record = .{ .fields = &fields };
    const s = try callFormatter(std.testing.allocator, .sfv, &rec, &.{});
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings("a.txt    00000000", s);
}

test "checksum emits digest then path regardless of field order" {
    var fields = [_]value.RecordField{
        .{ .name = "path", .value = value.Value.plainStr("/tmp/a.txt") },
        .{ .name = "crc32", .value = value.Value.plainStr("00000000") },
    };
    var rec: value.Record = .{ .fields = &fields };
    const c = try callFormatter(std.testing.allocator, .checksum, &rec, &.{});
    defer std.testing.allocator.free(c);
    try std.testing.expectEqualStrings("00000000    /tmp/a.txt", c);
}

test "sfv rejects missing name field" {
    var fields = [_]value.RecordField{
        .{ .name = "path", .value = value.Value.plainStr("/tmp/a.txt") },
        .{ .name = "crc32", .value = value.Value.plainStr("00000000") },
    };
    var rec: value.Record = .{ .fields = &fields };
    try std.testing.expectError(error.InvalidMethodFields, callFormatter(std.testing.allocator, .sfv, &rec, &.{}));
}

test "json and jsonPretty" {
    var fields = [_]value.RecordField{
        .{ .name = "a", .value = value.Value.plainStr("x") },
        .{ .name = "n", .value = .{ .int = 1 } },
    };
    var rec: value.Record = .{ .fields = &fields };
    const compact = try callFormatter(std.testing.allocator, .json, &rec, &.{});
    defer std.testing.allocator.free(compact);
    try std.testing.expectEqualStrings("{\"a\":\"x\",\"n\":1}", compact);

    const pretty = try callFormatter(std.testing.allocator, .json_pretty, &rec, &.{});
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
    const s = try callFormatter(std.testing.allocator, .json, &outer, &.{});
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
    const csv = try callFormatter(std.testing.allocator, .csv, &rec, &.{});
    defer std.testing.allocator.free(csv);
    try std.testing.expectEqualStrings("one,two", csv);
    const sp = try callFormatter(std.testing.allocator, .spaced, &rec, &.{});
    defer std.testing.allocator.free(sp);
    try std.testing.expectEqualStrings("one two", sp);
    const tb = try callFormatter(std.testing.allocator, .tabbed, &rec, &.{});
    defer std.testing.allocator.free(tb);
    try std.testing.expectEqualStrings("one\ttwo", tb);
}

test "sfv rejects wrong field count" {
    var fields = [_]value.RecordField{
        .{ .name = "a", .value = value.Value.plainStr("x") },
    };
    var rec: value.Record = .{ .fields = &fields };
    try std.testing.expectError(error.InvalidMethodFields, callFormatter(std.testing.allocator, .sfv, &rec, &.{}));
}

test "digestsEqual is case-insensitive" {
    try std.testing.expect(digestsEqual("abc", value.Str{ .bytes = "ABC", .is_digest = false }));
    try std.testing.expect(!digestsEqual("abc", value.Str{ .bytes = "abd", .is_digest = false }));
}
