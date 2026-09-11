//! Range-kind builtins (docs/l2h-semantics.md §4): typeOf + eval.
//! Name catalogs stay in `props.zig` / `method.zig`; formatters stay in `method.zig`.
//! Compile and interpret read `spec(Kind)` instead of switching on Kind in parallel.

const std = @import("std");
const hashes = @import("hashes");
const modes = @import("modes");
const value = @import("value.zig");
const props = @import("props.zig");
const method = @import("method.zig");
const diag = @import("diag.zig");
const expr = @import("expr.zig");

const Value = value.Value;
const Access = props.Access;
const Kind = method.Kind;

pub const Ctx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
};

pub const Error = error{
    UnknownProperty,
    UnknownHash,
    InvalidHashDigest,
    InvalidMethodArity,
    InvalidMethodReceiver,
    InvalidMethodFields,
    TypeMismatch,
    IoFailure,
    WriteFailed,
    Overflow,
    /// Negative `limit(n)` / `offset(n)` argument (§4.5).
    InvalidWindow,
    /// `Hash.min(n)` / `Hash.max(n)` with `n < 1` (§4.4).
    InvalidRestoreBound,
    /// `Hash.min` greater than `Hash.max` after a bound method (§4.4).
    InvalidRestoreRange,
    /// Restore `Hash.max` exceeds the brute-force length cap (§4.4).
    InvalidRestoreLength,
    /// Negative `tree(n)` depth (§4.6).
    InvalidTreeDepth,
    /// File hash window starts past EOF (§4.5).
    OffsetTooBig,
    /// UTF-16-widening algorithm (e.g. NTLM) got a non-UTF-8 `String` payload (§4.3).
    InvalidStringPayload,
} || std.mem.Allocator.Error;

pub const TypeTag = enum { string, int, bool, file, dir, hash };

pub const RecvClass = enum { record, file_or_string, dir, file, hash, seq };

pub const ArgSpec = enum { none, int, string, optional_int };

/// Allowed argument count. Independent of `ArgSpec` so a later method can take
/// e.g. two ints without inventing a new arg tag first.
pub const Arity = struct {
    min: usize,
    max: usize,
};

/// Compile-time signature of a method kind.
pub const MethodSpec = struct {
    result: TypeTag,
    recv: RecvClass,
    args: ArgSpec,
    arity: Arity,

    pub fn arityOk(self: MethodSpec, n: usize) bool {
        return n >= self.arity.min and n <= self.arity.max;
    }
};

pub fn typeOfProp(access: Access) TypeTag {
    return switch (access) {
        .path, .name, .hash_algo, .hash_dict => .string,
        .size, .offset, .limit, .hash_min, .hash_max => .int,
        .readable, .hash_no_probe => .bool,
    };
}

pub fn spec(kind: Kind) MethodSpec {
    return switch (kind) {
        .formatter => .{ .result = .string, .recv = .record, .args = .none, .arity = .{ .min = 0, .max = 0 } },
        .hash_check => .{ .result = .bool, .recv = .file_or_string, .args = .string, .arity = .{ .min = 1, .max = 1 } },
        .dir_tree => .{ .result = .dir, .recv = .dir, .args = .optional_int, .arity = .{ .min = 0, .max = 1 } },
        .dir_skip_errors => .{ .result = .dir, .recv = .dir, .args = .none, .arity = .{ .min = 0, .max = 0 } },
        .file_offset, .file_limit => .{ .result = .file, .recv = .file, .args = .int, .arity = .{ .min = 1, .max = 1 } },
        .hash_dict => .{ .result = .hash, .recv = .hash, .args = .string, .arity = .{ .min = 1, .max = 1 } },
        .hash_min, .hash_max => .{ .result = .hash, .recv = .hash, .args = .int, .arity = .{ .min = 1, .max = 1 } },
        .hash_noprobe => .{ .result = .hash, .recv = .hash, .args = .none, .arity = .{ .min = 0, .max = 0 } },
        .seq_count => .{ .result = .int, .recv = .seq, .args = .none, .arity = .{ .min = 0, .max = 0 } },
    };
}

const failSpan = diag.failSpan;

fn hashHexOfBytes(ctx: Ctx, algo: []const u8, bytes: []const u8) Error![]const u8 {
    const def = hashes.getHash(algo) orelse return error.UnknownHash;
    var digest: [modes.types.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([modes.types.MAX_DIGEST_SIZE]u8);
    hashes.createStringDigest(def, bytes, digest[0..def.hash_length], ctx.allocator) catch |err| return switch (err) {
        error.InvalidUtf8 => error.InvalidStringPayload,
        error.OutOfMemory => error.OutOfMemory,
    };
    var hex_buf: [modes.types.MAX_DIGEST_SIZE * 2]u8 = undefined;
    const hex = modes.types.hashToHex(digest[0..def.hash_length], true, &hex_buf);
    return try ctx.allocator.dupe(u8, hex);
}

fn hashHexOfFile(ctx: Ctx, algo: []const u8, file: value.FileVal) Error![]const u8 {
    const def = hashes.getHash(algo) orelse return error.UnknownHash;
    const digest = modes.file.createFileDigest(def, file.path, .{
        .offset = file.offset,
        .limit = file.limit,
    }, ctx.io) catch |err| switch (err) {
        error.OffsetPastEof => {
            diag.noteIoPath(file.path);
            return error.OffsetTooBig;
        },
        error.OutOfMemory => return error.OutOfMemory,
        error.OpenFailed, error.StatFailed, error.ReadFailed => return diag.ioFail(file.path),
    };
    var hex_buf: [modes.types.MAX_DIGEST_SIZE * 2]u8 = undefined;
    const hex = modes.types.hashToHex(digest.slice(), true, &hex_buf);
    return try ctx.allocator.dupe(u8, hex);
}

fn fileSize(ctx: Ctx, path: []const u8) Error!i64 {
    var file = std.Io.Dir.cwd().openFile(ctx.io, path, .{}) catch return diag.ioFail(path);
    defer file.close(ctx.io);
    const st = file.stat(ctx.io) catch return diag.ioFail(path);
    return std.math.cast(i64, st.size) orelse return error.Overflow;
}

fn fileIsReadable(ctx: Ctx, path: []const u8) bool {
    var file = std.Io.Dir.cwd().openFile(ctx.io, path, .{}) catch return false;
    defer file.close(ctx.io);
    const st = file.stat(ctx.io) catch return false;
    return st.kind == .file;
}

fn restoreHash(ctx: Ctx, algo: []const u8, h: value.HashVal) Error!void {
    const def = hashes.getHash(algo) orelse return error.UnknownHash;
    var target: [modes.types.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([modes.types.MAX_DIGEST_SIZE]u8);
    modes.types.parseSearchHash(h.digest, false, def, &target) catch return error.InvalidHashDigest;
    const password = modes.hash.restore(def, target[0..def.hash_length], .{
        .dictionary = h.dictionary,
        .min = h.min,
        .max = h.max,
        .no_probe = h.no_probe,
    }, ctx.allocator, ctx.io, ctx.out) catch |err| return switch (err) {
        error.InvalidRange => error.InvalidRestoreRange,
        error.PassmaxTooBig => error.InvalidRestoreLength,
        error.OutOfMemory => error.OutOfMemory,
        error.WriteFailed => error.WriteFailed,
        else => error.IoFailure,
    };
    if (password) |p| ctx.allocator.free(p);
}

fn hashWithBound(sp: expr.Span, recv: value.HashVal, set_min: bool, n: i64) Error!value.HashVal {
    if (n < 1) return failSpan(sp, error.InvalidRestoreBound);
    const bound = std.math.cast(i32, n) orelse return failSpan(sp, error.Overflow);
    const h = if (set_min) recv.withMin(bound) else recv.withMax(bound);
    if (h.min > h.max) return failSpan(sp, error.InvalidRestoreRange);
    return h;
}

/// Demand-driven property access (semantics §4).
/// `baked` is the compile-time builtin when known; null defers to runtime lookup
/// (record fields, or recv typed `.unknown` at compile).
pub fn evalProp(ctx: Ctx, recv: Value, prop: []const u8, baked: ?Access, sp: expr.Span) Error!Value {
    if (recv == .record) {
        return recv.record.get(prop) orelse failSpan(sp, error.UnknownProperty);
    }
    const access = baked orelse blk: {
        const kind = recv.sourceKind() orelse return failSpan(sp, error.UnknownProperty);
        break :blk props.lookup(kind, prop) orelse return failSpan(sp, error.UnknownProperty);
    };
    return switch (access) {
        .path => switch (recv) {
            .file => |f| Value.plainStr(f.path),
            .dir => |d| Value.plainStr(d.path),
            else => unreachable,
        },
        .name => switch (recv) {
            .file => |f| Value.plainStr(std.fs.path.basenameWindows(f.path)),
            else => unreachable,
        },
        .size => switch (recv) {
            .file => |f| .{ .int = fileSize(ctx, f.path) catch |err| return failSpan(sp, err) },
            .string => |s| .{ .int = std.math.cast(i64, s.bytes.len) orelse return failSpan(sp, error.Overflow) },
            else => unreachable,
        },
        .offset => switch (recv) {
            .file => |f| .{ .int = f.offset },
            else => unreachable,
        },
        .limit => switch (recv) {
            .file => |f| .{ .int = f.limit },
            else => unreachable,
        },
        .readable => switch (recv) {
            .file => |f| .{ .bool = fileIsReadable(ctx, f.path) },
            else => unreachable,
        },
        .hash_dict => switch (recv) {
            .hash => |h| Value.plainStr(h.dictionary orelse modes.defaultAlphabet),
            else => unreachable,
        },
        .hash_min => switch (recv) {
            .hash => |h| .{ .int = h.min },
            else => unreachable,
        },
        .hash_max => switch (recv) {
            .hash => |h| .{ .int = h.max },
            else => unreachable,
        },
        .hash_no_probe => switch (recv) {
            .hash => |h| .{ .bool = h.no_probe },
            else => unreachable,
        },
        .hash_algo => switch (recv) {
            .file => |f| Value.digestStr(hashHexOfFile(ctx, prop, f) catch |err| return failSpan(sp, err)),
            .string => |s| Value.digestStr(hashHexOfBytes(ctx, prop, s.bytes) catch |err| return failSpan(sp, err)),
            .hash => |h| blk: {
                restoreHash(ctx, prop, h) catch |err| return failSpan(sp, err);
                break :blk Value.digestStr(h.digest);
            },
            else => unreachable,
        },
    };
}

/// Apply a method to an already-evaluated receiver and scalar args (§4).
pub fn evalMethod(
    ctx: Ctx,
    kind: Kind,
    name: []const u8,
    recv: Value,
    args: []const Value,
    sp: expr.Span,
) Error!Value {
    if (!spec(kind).arityOk(args.len)) return failSpan(sp, error.InvalidMethodArity);
    return switch (kind) {
        .formatter => |f| {
            const rec = switch (recv) {
                .record => |r| r,
                else => return failSpan(sp, error.InvalidMethodReceiver),
            };
            const bytes = method.callFormatter(ctx.allocator, f, rec, args) catch |err| return failSpan(sp, switch (err) {
                error.InvalidMethodArity => error.InvalidMethodArity,
                error.InvalidMethodFields => error.InvalidMethodFields,
                error.TypeMismatch => error.TypeMismatch,
                error.OutOfMemory => error.OutOfMemory,
                else => error.WriteFailed,
            });
            return Value.plainStr(bytes);
        },
        .hash_check => {
            if (args.len != 1 or args[0] != .string) return failSpan(sp, error.TypeMismatch);
            const actual_hex = switch (recv) {
                .file => |file| hashHexOfFile(ctx, name, file) catch |err| return failSpan(sp, err),
                .string => |s| hashHexOfBytes(ctx, name, s.bytes) catch |err| return failSpan(sp, err),
                else => return failSpan(sp, error.InvalidMethodReceiver),
            };
            return .{ .bool = method.digestsEqual(actual_hex, args[0].string) };
        },
        .dir_tree => {
            if (recv != .dir) return failSpan(sp, error.InvalidMethodReceiver);
            const max_depth: ?u32 = if (args.len == 0) null else blk: {
                if (args[0] != .int) return failSpan(sp, error.TypeMismatch);
                if (args[0].int < 0) return failSpan(sp, error.InvalidTreeDepth);
                break :blk std.math.cast(u32, args[0].int) orelse return failSpan(sp, error.Overflow);
            };
            return .{ .dir = recv.dir.withTree(max_depth) };
        },
        .dir_skip_errors => {
            if (recv != .dir) return failSpan(sp, error.InvalidMethodReceiver);
            return .{ .dir = recv.dir.withSkipErrors() };
        },
        .file_offset, .file_limit => {
            if (recv != .file) return failSpan(sp, error.InvalidMethodReceiver);
            if (args.len != 1 or args[0] != .int) return failSpan(sp, error.TypeMismatch);
            if (args[0].int < 0) return failSpan(sp, error.InvalidWindow);
            const f = if (kind == .file_offset)
                recv.file.withOffset(args[0].int)
            else
                recv.file.withLimit(args[0].int);
            return .{ .file = f };
        },
        .hash_dict => {
            if (recv != .hash) return failSpan(sp, error.InvalidMethodReceiver);
            if (args.len != 1 or args[0] != .string) return failSpan(sp, error.TypeMismatch);
            return .{ .hash = recv.hash.withDict(args[0].string.bytes) };
        },
        .hash_min, .hash_max => {
            if (recv != .hash) return failSpan(sp, error.InvalidMethodReceiver);
            if (args.len != 1 or args[0] != .int) return failSpan(sp, error.TypeMismatch);
            const h = try hashWithBound(sp, recv.hash, kind == .hash_min, args[0].int);
            return .{ .hash = h };
        },
        .hash_noprobe => {
            if (recv != .hash) return failSpan(sp, error.InvalidMethodReceiver);
            return .{ .hash = recv.hash.withNoProbe() };
        },
        .seq_count => {
            if (recv != .seq) return failSpan(sp, error.InvalidMethodReceiver);
            const n = std.math.cast(i64, recv.seq.items.len) orelse return failSpan(sp, error.Overflow);
            return .{ .int = n };
        },
    };
}

test "typeOfProp and spec match §4 result kinds" {
    // Arrange

    // Act

    // Assert
    try std.testing.expectEqual(TypeTag.int, typeOfProp(.offset));
    try std.testing.expectEqual(TypeTag.string, typeOfProp(.hash_algo));
    try std.testing.expectEqual(TypeTag.file, spec(.file_offset).result);
    try std.testing.expectEqual(TypeTag.hash, spec(.hash_max).result);
    try std.testing.expectEqual(RecvClass.file, spec(.file_limit).recv);
    try std.testing.expectEqual(ArgSpec.int, spec(.file_offset).args);
}

test "spec arity min/max is checked at call" {
    // Arrange

    // Act

    // Assert
    try std.testing.expectEqual(Arity{ .min = 0, .max = 0 }, spec(.{ .formatter = .sfv }).arity);
    try std.testing.expectEqual(Arity{ .min = 0, .max = 1 }, spec(.dir_tree).arity);
    try std.testing.expectEqual(Arity{ .min = 0, .max = 0 }, spec(.dir_skip_errors).arity);
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, spec(.file_offset).arity);
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, spec(.file_limit).arity);
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, spec(.hash_dict).arity);
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, spec(.hash_min).arity);
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, spec(.hash_max).arity);
    try std.testing.expectEqual(Arity{ .min = 0, .max = 0 }, spec(.hash_noprobe).arity);
    try std.testing.expectEqual(Arity{ .min = 0, .max = 0 }, spec(.seq_count).arity);
    try std.testing.expectEqual(Arity{ .min = 1, .max = 1 }, spec(.hash_check).arity);
    try std.testing.expect(spec(.dir_tree).arityOk(0));
    try std.testing.expect(spec(.dir_tree).arityOk(1));
    try std.testing.expect(!spec(.dir_tree).arityOk(2));
    try std.testing.expect(spec(.file_offset).arityOk(1));
    try std.testing.expect(!spec(.file_offset).arityOk(0));
    try std.testing.expect(!spec(.file_limit).arityOk(2));
    try std.testing.expect(spec(.seq_count).arityOk(0));
    try std.testing.expect(!spec(.seq_count).arityOk(1));
    try std.testing.expect(!spec(.hash_check).arityOk(0));
}

test "evalMethod rejects wrong arity" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx: Ctx = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
        .out = &writer,
    };
    const extra = [_]Value{.{ .int = 1 }};

    // Act

    // Assert
    try std.testing.expectError(
        error.InvalidMethodArity,
        evalMethod(ctx, .seq_count, "count", Value.plainStr(""), &extra, .{}),
    );
}

test "evalMethod File.offset sets the window" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx: Ctx = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
        .out = &writer,
    };
    const args = [_]Value{.{ .int = 2 }};

    // Act
    const got = try evalMethod(ctx, .file_offset, "offset", Value.filePath("x"), &args, .{});

    // Assert
    try std.testing.expectEqual(@as(i64, 2), got.file.offset);
    try std.testing.expectEqualStrings("x", got.file.path);
}

test "evalMethod Hash.max less than min is InvalidRestoreRange" {
    // Arrange
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx: Ctx = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
        .out = &writer,
    };
    const recv = Value{ .hash = .{
        .digest = "D41D8CD98F00B204E9800998ECF8427E",
        .min = 5,
        .max = 10,
    } };
    const args = [_]Value{.{ .int = 2 }};

    // Act

    // Assert
    try std.testing.expectError(
        error.InvalidRestoreRange,
        evalMethod(ctx, .hash_max, "max", recv, &args, .{}),
    );
}
