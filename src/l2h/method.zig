//! Method catalog for `recv.method(args…)` — Record formatters (§4.7),
//! hash-check on File/String (§4.8), Dir walk helpers (§4.6), File
//! hash-window helpers (§4.5), and Seq cardinality (§4.9). Analogous to
//! `props.zig` for properties. Receiver may be an identifier or a record
//! literal `{…}`.

const std = @import("std");
const hashes = @import("hashes");
const modes = @import("modes");
const value = @import("value.zig");

pub const Error = error{
    InvalidMethodArity,
    InvalidMethodFields,
    TypeMismatch,
} || std.mem.Allocator.Error || std.Io.Writer.Error;

/// Record formatters. `sfv` / `checksum` look up fields **by name** and always
/// emit a fixed order (independent of declaration order in `{…}`):
/// - `sfv`: requires `name` + one other field → `name    digest` (`hc --sfv`)
/// - `checksum`: requires `path` + one other field → `digest path` (`hc -c`)
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
    /// `Dir.tree()` / `Dir.tree(n)` — same path, depth limit on the Dir (§4.6).
    dir_tree,
    /// `Dir.skipErrors()` — skip unreadable subdirectories while walking (§4.6).
    dir_skip_errors,
    /// `File.offset(n)` — same path, start byte for hashing (§4.5).
    file_offset,
    /// `File.limit(n)` — same path, max bytes to hash from offset (§4.5).
    file_limit,
    /// `Hash.dict(s)` — same digest, new restore alphabet (§4.4).
    hash_dict,
    /// `Hash.min(n)` — same digest, new restore min length (§4.4).
    hash_min,
    /// `Hash.max(n)` — same digest, new restore max length (§4.4).
    hash_max,
    /// `Hash.noProbe()` — same digest, skip restore timing probe (§4.4).
    hash_noprobe,
    /// `Seq.count()` — number of elements in a materialized sequence (§4.9).
    seq_count,
};

/// Allowed argument count range for a method kind.
pub const Arity = struct {
    min: usize,
    max: usize,
};

const FORMATTERS = std.StaticStringMap(Formatter).initComptime(.{
    .{ "sfv", .sfv },
    .{ "checksum", .checksum },
    .{ "json", .json },
    .{ "jsonPretty", .json_pretty },
    .{ "csv", .csv },
    .{ "spaced", .spaced },
    .{ "tabbed", .tabbed },
});

const BUILTIN_METHODS = std.StaticStringMap(Kind).initComptime(.{
    .{ "tree", .dir_tree },
    .{ "skipErrors", .dir_skip_errors },
    .{ "offset", .file_offset },
    .{ "limit", .file_limit },
    .{ "dict", .hash_dict },
    .{ "min", .hash_min },
    .{ "max", .hash_max },
    .{ "noProbe", .hash_noprobe },
    .{ "count", .seq_count },
});

/// Look up a method by name; `null` if unknown.
pub fn lookup(name: []const u8) ?Kind {
    if (FORMATTERS.get(name)) |f| return .{ .formatter = f };
    if (BUILTIN_METHODS.get(name)) |k| return k;
    if (hashes.getHash(name) != null) return .hash_check;
    return null;
}

/// Allowed argument count range for a method kind.
pub fn arityRange(k: Kind) Arity {
    return switch (k) {
        .formatter, .dir_skip_errors, .seq_count, .hash_noprobe => .{ .min = 0, .max = 0 },
        .hash_check, .file_offset, .file_limit, .hash_dict, .hash_min, .hash_max => .{ .min = 1, .max = 1 },
        .dir_tree => .{ .min = 0, .max = 1 },
    };
}

pub fn arityOk(k: Kind, n: usize) bool {
    const r = arityRange(k);
    return n >= r.min and n <= r.max;
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

/// Case-insensitive digest equality for hash-check (§4.8 / §5.3).
pub fn digestsEqual(actual_hex: []const u8, expected: value.Str) bool {
    const actual = value.Str{ .bytes = actual_hex, .is_digest = true };
    return actual.compare(expected) == .eq;
}

fn formatSfv(allocator: std.mem.Allocator, rec: *const value.Record) Error![]u8 {
    const pair = try splitLabeledPair(rec, "name");
    return try joinTwo(allocator, pair.label, pair.other, modes.types.SFV_SEPARATOR);
}

fn formatChecksum(allocator: std.mem.Allocator, rec: *const value.Record) Error![]u8 {
    const pair = try splitLabeledPair(rec, "path");
    return try joinTwo(allocator, pair.other, pair.label, modes.types.CHECKSUM_SEPARATOR);
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

fn joinTwo(allocator: std.mem.Allocator, a: value.Value, b: value.Value, sep: []const u8) Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    try a.writeScalar(&out.writer);
    try out.writer.writeAll(sep);
    try b.writeScalar(&out.writer);
    return try out.toOwnedSlice();
}

fn joinFields(allocator: std.mem.Allocator, rec: *const value.Record, sep: []const u8) Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    for (rec.fields, 0..) |f, i| {
        if (i > 0) try out.writer.writeAll(sep);
        try f.value.writeScalar(&out.writer);
    }
    return try out.toOwnedSlice();
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

test "lookup kind covers formatters, dir_tree, file window, seq_count, and hash-check" {
    try std.testing.expectEqual(Kind{ .formatter = .sfv }, lookup("sfv").?);
    try std.testing.expectEqual(Kind{ .formatter = .checksum }, lookup("checksum").?);
    try std.testing.expectEqual(Kind{ .formatter = .json }, lookup("json").?);
    try std.testing.expectEqual(Kind{ .formatter = .json_pretty }, lookup("jsonPretty").?);
    try std.testing.expect(lookup("Sfv") == null);
    try std.testing.expectEqual(@as(?Kind, .dir_tree), lookup("tree"));
    try std.testing.expectEqual(@as(?Kind, .dir_skip_errors), lookup("skipErrors"));
    try std.testing.expectEqual(@as(?Kind, .file_offset), lookup("offset"));
    try std.testing.expectEqual(@as(?Kind, .file_limit), lookup("limit"));
    try std.testing.expectEqual(@as(?Kind, .hash_dict), lookup("dict"));
    try std.testing.expectEqual(@as(?Kind, .hash_min), lookup("min"));
    try std.testing.expectEqual(@as(?Kind, .hash_max), lookup("max"));
    try std.testing.expectEqual(@as(?Kind, .hash_noprobe), lookup("noProbe"));
    try std.testing.expectEqual(@as(?Kind, .seq_count), lookup("count"));
    try std.testing.expectEqual(@as(?Kind, .hash_check), lookup("md5"));
    try std.testing.expectEqual(@as(?Kind, .hash_check), lookup("sha1"));
    try std.testing.expect(lookup("nope") == null);

    try std.testing.expectEqual(Arity{ .min = 0, .max = 0 }, arityRange(.{ .formatter = .sfv }));
    try std.testing.expectEqual(Arity{ .min = 0, .max = 1 }, arityRange(.dir_tree));
    try std.testing.expectEqual(Arity{ .min = 0, .max = 0 }, arityRange(.dir_skip_errors));
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, arityRange(.file_offset));
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, arityRange(.file_limit));
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, arityRange(.hash_dict));
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, arityRange(.hash_min));
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, arityRange(.hash_max));
    try std.testing.expectEqual(Arity{ .min = 0, .max = 0 }, arityRange(.hash_noprobe));
    try std.testing.expectEqual(Arity{ .min = 0, .max = 0 }, arityRange(.seq_count));
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, arityRange(.hash_check));
    try std.testing.expect(arityOk(.dir_tree, 0));
    try std.testing.expect(arityOk(.dir_tree, 1));
    try std.testing.expect(!arityOk(.dir_tree, 2));
    try std.testing.expect(arityOk(.file_offset, 1));
    try std.testing.expect(!arityOk(.file_offset, 0));
    try std.testing.expect(!arityOk(.file_limit, 2));
    try std.testing.expect(arityOk(.seq_count, 0));
    try std.testing.expect(!arityOk(.seq_count, 1));
    try std.testing.expect(!arityOk(.hash_check, 0));
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
    try std.testing.expectEqualStrings("00000000 /tmp/a.txt", c);
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
