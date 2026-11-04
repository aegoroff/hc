const std = @import("std");
const hashes = @import("hashes");
const lib = @import("lib");
const builtins = @import("builtins.zig");
const value = @import("value.zig");
const expr = @import("expr.zig");

const Ctx = builtins.Ctx;
const Error = builtins.Error;
const Value = value.Value;
const Env = value.Env;
const Expr = expr.Expr;

/// Write a select/group result value to the sink (unwraps singleton Seq).
pub fn print(ctx: Ctx, v: Value) Error!void {
    return printValue(ctx, v);
}

pub fn select(ctx: Ctx, e: *const Expr, env: *Env, v: Value) Error!void {
    // Hash.<algo> restore already printed via evalProp (§4.4).
    if (isHashRestoreSelect(e, env)) return;
    return printValue(ctx, v);
}

fn printValue(ctx: Ctx, v: Value) Error!void {
    switch (v) {
        .string, .int, .bool, .file, .dir, .hash => try sinkLine(ctx, v),
        .record => |rec| {
            // Exactly one line per field (§7): no expanding nested Seq/Record.
            for (rec.fields) |f| try sinkFieldLine(ctx, f.value);
        },
        .seq => |s| {
            for (s.items) |item| try printValue(ctx, item);
        },
    }
}

/// One sink line for a Record field value (scalars / path-like only).
fn sinkFieldLine(ctx: Ctx, v: Value) Error!void {
    switch (v) {
        .string, .int, .bool, .file, .dir, .hash => try sinkLine(ctx, v),
        .record, .seq => return error.TypeMismatch,
    }
}

fn sinkLine(ctx: Ctx, v: Value) Error!void {
    switch (v) {
        .string, .int, .bool => try v.writeScalar(ctx.out),
        .file => |f| try lib.writeLineText(ctx.out, f.path, false),
        .dir => |d| try lib.writeLineText(ctx.out, d.path, false),
        .hash => |h| try ctx.out.writeAll(h.digest),
        .record, .seq => unreachable,
    }
    try ctx.out.writeAll("\n");
    // Progressive output: main() uses a large stdout buffer; flush each sunk line.
    ctx.out.flush() catch return error.WriteFailed;
}

/// True when `e` is a terminal Hash restore algo projection (`x.md5`,
/// `x.noProbe().md5`, `r.h.md5`, …). Non-algo Hash props (`min` / `max` /
/// `dict` / `noProbe`) and File/String digests are false — those still need a
/// sink line.
fn isHashRestoreSelect(e: *const Expr, env: *Env) bool {
    if (e.kind != .prop) return false;
    const p = e.kind.prop;
    if (p.access) |a| {
        if (a != .hash_algo) return false;
    } else if (hashes.getHash(p.prop) == null) {
        return false;
    }
    return exprIsHash(p.recv, env);
}

/// Walk a Hash binding, `dict`/`min`/`max`/`noProbe` chain, or record-field
/// path without evaluating (re-eval would restore twice).
fn exprIsHash(e: *const Expr, env: *Env) bool {
    return switch (e.kind) {
        .name => |n| if (env.get(n)) |v| v == .hash else false,
        .method => |m| switch (m.kind) {
            .hash_dict, .hash_min, .hash_max, .hash_noprobe => exprIsHash(m.recv, env),
            else => false,
        },
        .prop => |p| blk: {
            // Builtin props never return Hash (algo→string, min→int, …).
            // Record fields leave `access` null (§4.3 / compile).
            if (p.access != null) break :blk false;
            const rec = exprAsRecord(p.recv, env) orelse break :blk false;
            const field = rec.get(p.prop) orelse break :blk false;
            break :blk field == .hash;
        },
        else => false,
    };
}

/// Resolve a record value from a name or nested record-field path (no eval).
fn exprAsRecord(e: *const Expr, env: *Env) ?*value.Record {
    return switch (e.kind) {
        .name => |n| {
            const v = env.get(n) orelse return null;
            return if (v == .record) v.record else null;
        },
        .prop => |p| {
            if (p.access != null) return null;
            const parent = exprAsRecord(p.recv, env) orelse return null;
            const field = parent.get(p.prop) orelse return null;
            return if (field == .record) field.record else null;
        },
        else => null,
    };
}

fn testCtx(gpa: std.mem.Allocator, out: *std.Io.Writer) Ctx {
    return .{
        .allocator = gpa,
        .io = std.testing.io,
        .out = out,
    };
}

test "sink keeps a path newline on one line" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(std.testing.allocator, &writer);

    // Act
    try print(ctx, Value.filePath("a\nb|c"));

    // Assert
    try std.testing.expectEqualStrings("a\\nb|c\n", std.Io.Writer.buffered(&writer));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, std.Io.Writer.buffered(&writer), "\n"));
}

test "sink record prints two lines" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    var fields = [_]value.RecordField{
        .{ .name = "md5", .value = Value.plainStr("aa") },
        .{ .name = "sha1", .value = Value.plainStr("bb") },
    };
    var rec: value.Record = .{ .fields = &fields };
    // Act
    try print(ctx, .{ .record = &rec });
    // Assert
    try std.testing.expectEqualStrings("aa\nbb\n", std.Io.Writer.buffered(&writer));
}

test "sink record rejects Seq field" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    const items = try a.alloc(Value, 1);
    items[0] = Value.plainStr("x");
    const seq = try a.create(value.Seq);
    seq.* = .{ .items = items };
    var fields = [_]value.RecordField{
        .{ .name = "key", .value = .{ .int = 1 } },
        .{ .name = "items", .value = .{ .seq = seq } },
    };
    var rec: value.Record = .{ .fields = &fields };

    // Act

    // Assert
    try std.testing.expectError(error.TypeMismatch, print(ctx, .{ .record = &rec }));
}
