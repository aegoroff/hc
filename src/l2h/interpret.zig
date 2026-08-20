//! Interpret compiled l2h query plans (docs/l2h-semantics.md).
//! Pipeline: pull operators over the compiled plan (`open` / `next` / `close`).

const std = @import("std");
const hashes = @import("hashes");
const modes = @import("modes");
const state = @import("state.zig");
const value = @import("value.zig");
const diag = @import("diag.zig");
const expr = @import("expr.zig");
const compile = @import("compile.zig");
const plan = @import("plan.zig");
const props = @import("props.zig");
const method = @import("method.zig");
const re_match = @import("match_re.zig");

const Value = value.Value;
const Env = value.Env;
const Expr = expr.Expr;
const BinaryOp = expr.BinaryOp;

pub const Error = error{
    UndefinedName,
    TypeMismatch,
    UnknownProperty,
    UnknownHash,
    InvalidHashDigest,
    DuplicateField,
    InvalidMethodArity,
    InvalidMethodReceiver,
    InvalidMethodFields,
    IoFailure,
    WriteFailed,
    Overflow,
    QueryTooDeep,
    /// Negative `limit(n)` / `offset(n)` argument (§4.5).
    InvalidWindow,
    /// Negative `tree(n)` depth (§4.6).
    InvalidTreeDepth,
    /// Invalid regex pattern for `~` / `!~` (§5.3 / §8).
    BadRegex,
    /// File hash window starts past EOF (§4.5).
    OffsetTooBig,
    /// UTF-16-widening algorithm (e.g. NTLM) got a non-UTF-8 `String` payload (§4.3).
    InvalidStringPayload,
} || std.mem.Allocator.Error;

pub const Ctx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
};

fn runEnv(ctx: Ctx) modes.RunEnv {
    return .{ .io = ctx.io, .allocator = ctx.allocator, .out = ctx.out };
}

fn failExpr(e: *const Expr, err: Error) Error {
    return failSpan(e.span, err);
}

fn failSpan(sp: expr.Span, err: Error) Error {
    diag.noteSpan(sp);
    return err;
}

fn ioFail(path: []const u8) Error {
    diag.noteIoPath(path);
    return error.IoFailure;
}

/// Map errors from `modes.hashRun` / `builtinInit` without collapsing digest
/// parse failures into the file/dir I/O message.
fn mapHashRestoreError(err: anyerror) Error {
    return switch (err) {
        error.InvalidArgument => error.InvalidHashDigest,
        error.UnknownHash => error.UnknownHash,
        error.OutOfMemory => error.OutOfMemory,
        error.WriteFailed => error.WriteFailed,
        else => error.IoFailure,
    };
}

// --- property evaluation ----------------------------------------------------

fn hashHexOfBytes(ctx: Ctx, algo: []const u8, bytes: []const u8) Error![]const u8 {
    const def = hashes.getHash(algo) orelse return error.UnknownHash;
    var digest: [modes.types.MAX_DIGEST_SIZE]u8 align(8) = std.mem.zeroes([modes.types.MAX_DIGEST_SIZE]u8);
    modes.str.hashFromString(bytes, def, digest[0..def.hash_length], ctx.allocator) catch |err| return switch (err) {
        error.InvalidArgument => error.InvalidStringPayload,
        error.OutOfMemory => error.OutOfMemory,
        error.WriteFailed => error.WriteFailed,
        error.UnknownHash => error.UnknownHash,
    };
    var hex_buf: [modes.types.MAX_DIGEST_SIZE * 2]u8 = undefined;
    const hex = modes.types.hashToHex(digest[0..def.hash_length], true, &hex_buf);
    return try ctx.allocator.dupe(u8, hex);
}

fn hashHexOfFile(ctx: Ctx, algo: []const u8, file: value.FileVal) Error![]const u8 {
    const def = hashes.getHash(algo) orelse return error.UnknownHash;
    const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = algo };
    var fctx: modes.FileCtx = .{
        .opts = .{
            .builtin = &bctx,
            .limit = file.limit,
            .offset = file.offset,
        },
        .file_path = file.path,
    };
    const result = modes.file.calculateFile(file.path, &fctx, runEnv(ctx), def) catch return ioFail(file.path);
    if (result.offset_error != null) {
        diag.noteIoPath(file.path);
        return error.OffsetTooBig;
    }
    if (result.open_error != null or result.info_error != null or result.hash_error != null)
        return ioFail(file.path);
    var hex_buf: [modes.types.MAX_DIGEST_SIZE * 2]u8 = undefined;
    const hex = modes.types.hashToHex(result.digest[0..result.digest_len], true, &hex_buf);
    return try ctx.allocator.dupe(u8, hex);
}

fn fileSize(ctx: Ctx, path: []const u8) Error!i64 {
    var file = std.Io.Dir.cwd().openFile(ctx.io, path, .{}) catch return ioFail(path);
    defer file.close(ctx.io);
    const st = file.stat(ctx.io) catch return ioFail(path);
    // usize (u64 on 64-bit) -> i64: a >2^63-byte file would overflow. Surface a
    // clean error instead of trapping (Debug/ReleaseSafe) or UB (ReleaseFast).
    return std.math.cast(i64, st.size) orelse return error.Overflow;
}

fn fileIsReadable(ctx: Ctx, path: []const u8) bool {
    var file = std.Io.Dir.cwd().openFile(ctx.io, path, .{}) catch return false;
    defer file.close(ctx.io);
    const st = file.stat(ctx.io) catch return false;
    return st.kind == .file;
}

/// Demand-driven property access (semantics §4).
/// `baked` is the compile-time builtin when known; null defers to runtime lookup
/// (record fields, or recv typed `.unknown` at compile).
pub fn evalProp(ctx: Ctx, recv: Value, prop: []const u8, baked: ?props.Access, sp: expr.Span) Error!Value {
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
        .hash_algo => switch (recv) {
            .file => |f| Value.digestStr(hashHexOfFile(ctx, prop, f) catch |err| return failSpan(sp, err)),
            .string => |s| Value.digestStr(hashHexOfBytes(ctx, prop, s.bytes) catch |err| return failSpan(sp, err)),
            .hash => |digest| blk: {
                // Restore: side-effect to out, value is the digest.
                const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = prop };
                var hctx: modes.HashCtx = .{ .hash = digest };
                const env = runEnv(ctx);
                const h = modes.builtinInit(&bctx, env) catch |err| {
                    return failSpan(sp, mapHashRestoreError(err));
                };
                modes.hashRun(&hctx, env, h) catch |err| {
                    return failSpan(sp, mapHashRestoreError(err));
                };
                break :blk Value.digestStr(digest);
            },
            else => unreachable,
        },
    };
}

// --- expression evaluation --------------------------------------------------

fn cmpInt(op: BinaryOp, left: i64, right: i64) bool {
    return switch (op) {
        .eq => left == right,
        .neq => left != right,
        .gt => left > right,
        .ge => left >= right,
        .lt => left < right,
        .le => left <= right,
        else => unreachable,
    };
}

/// Unwrap a singleton Seq to a scalar value for comparisons / method args.
/// When `allow_named_seq` is false (compare / orderby / join keys), only a
/// nested-query Seq may unwrap; bare Seq is TypeMismatch.
fn unwrapScalar(e: *const Expr, v: Value, allow_named_seq: bool) Error!Value {
    if (!allow_named_seq and e.kind != .nested_query) {
        if (v == .seq) return failExpr(e, error.TypeMismatch);
        return v;
    }
    if (v != .seq) return v;
    if (v.seq.items.len != 1) return failExpr(e, error.TypeMismatch);
    return v.seq.items[0];
}

fn asBool(e: *const Expr, v: Value) Error!bool {
    return switch (v) {
        .bool => |b| b,
        .seq => |s| s.items.len > 0,
        else => failExpr(e, error.TypeMismatch),
    };
}

/// Evaluate expression `e` under `env` (semantics §5 / §9).
pub fn evalExpr(ctx: Ctx, e: *const Expr, env: *Env, depth: u32) Error!Value {
    return switch (e.kind) {
        .string_lit => |s| Value.plainStr(s),
        .int_lit => |n| .{ .int = n },
        .bool_lit => |b| .{ .bool = b },
        .nested_query => |q| {
            // Nested plan was compiled ahead of eval; bound the runtime stack
            // in step with the analysis-time limit.
            if (depth >= compile.MAX_QUERY_DEPTH) return failExpr(e, error.QueryTooDeep);
            const items = try evalQueryValues(ctx, q, env, depth + 1);
            const seq = try ctx.allocator.create(value.Seq);
            seq.* = .{ .items = items };
            return .{ .seq = seq };
        },
        .name => |n| env.get(n) orelse failExpr(e, error.UndefinedName),
        .prop => |p| {
            const recv = try evalExpr(ctx, p.recv, env, depth);
            return evalProp(ctx, recv, p.prop, p.access, e.span);
        },
        .method => |m| {
            if (!method.arityOk(m.kind, m.args.len)) return failExpr(e, error.InvalidMethodArity);

            switch (m.kind) {
                .formatter => |f| {
                    const recv = try evalExpr(ctx, m.recv, env, depth);
                    const rec = switch (recv) {
                        .record => |r| r,
                        else => return failExpr(e, error.InvalidMethodReceiver),
                    };

                    const args = try ctx.allocator.alloc(Value, m.args.len);
                    for (m.args, 0..) |arg, i| {
                        args[i] = try unwrapScalar(arg, try evalExpr(ctx, arg, env, depth), true);
                    }
                    const bytes = method.callFormatter(ctx.allocator, f, rec, args) catch |err| {
                        return failExpr(e, err);
                    };
                    return Value.plainStr(bytes);
                },
                .hash_check => {
                    const recv = try evalExpr(ctx, m.recv, env, depth);
                    const expected_v = try unwrapScalar(m.args[0], try evalExpr(ctx, m.args[0], env, depth), true);
                    if (expected_v != .string) return failExpr(e, error.TypeMismatch);

                    const actual_hex = switch (recv) {
                        .file => |file| hashHexOfFile(ctx, m.name, file) catch |err| return failExpr(e, err),
                        .string => |s| hashHexOfBytes(ctx, m.name, s.bytes) catch |err| return failExpr(e, err),
                        else => return failExpr(e, error.InvalidMethodReceiver),
                    };
                    return .{ .bool = method.digestsEqual(actual_hex, expected_v.string) };
                },
                .dir_tree => {
                    const recv = try evalExpr(ctx, m.recv, env, depth);
                    if (recv != .dir) return failExpr(e, error.InvalidMethodReceiver);
                    const max_depth: ?u32 = if (m.args.len == 0) null else blk: {
                        const arg_v = try unwrapScalar(m.args[0], try evalExpr(ctx, m.args[0], env, depth), true);
                        if (arg_v != .int) return failExpr(e, error.TypeMismatch);
                        if (arg_v.int < 0) return failExpr(e, error.InvalidTreeDepth);
                        break :blk std.math.cast(u32, arg_v.int) orelse return failExpr(e, error.Overflow);
                    };
                    return .{ .dir = recv.dir.withTree(max_depth) };
                },
                .dir_skip_errors => {
                    const recv = try evalExpr(ctx, m.recv, env, depth);
                    if (recv != .dir) return failExpr(e, error.InvalidMethodReceiver);
                    return .{ .dir = recv.dir.withSkipErrors() };
                },
                .file_offset, .file_limit => {
                    const recv = try evalExpr(ctx, m.recv, env, depth);
                    if (recv != .file) return failExpr(e, error.InvalidMethodReceiver);
                    const arg_v = try unwrapScalar(m.args[0], try evalExpr(ctx, m.args[0], env, depth), true);
                    if (arg_v != .int) return failExpr(e, error.TypeMismatch);
                    if (arg_v.int < 0) return failExpr(e, error.InvalidWindow);
                    const f = if (m.kind == .file_offset)
                        recv.file.withOffset(arg_v.int)
                    else
                        recv.file.withLimit(arg_v.int);
                    return .{ .file = f };
                },
                .seq_count => {
                    const recv = try evalExpr(ctx, m.recv, env, depth);
                    if (recv != .seq) return failExpr(e, error.InvalidMethodReceiver);
                    const n = std.math.cast(i64, recv.seq.items.len) orelse return failExpr(e, error.Overflow);
                    return .{ .int = n };
                },
            }
        },
        .not => |arg| {
            const v = try evalExpr(ctx, arg, env, depth);
            return .{ .bool = !(try asBool(arg, v)) };
        },
        .binary => |b| {
            switch (b.op) {
                .and_ => {
                    const l = try evalExpr(ctx, b.left, env, depth);
                    if (!(try asBool(b.left, l))) return .{ .bool = false };
                    const r = try evalExpr(ctx, b.right, env, depth);
                    return .{ .bool = try asBool(b.right, r) };
                },
                .or_ => {
                    const l = try evalExpr(ctx, b.left, env, depth);
                    if (try asBool(b.left, l)) return .{ .bool = true };
                    const r = try evalExpr(ctx, b.right, env, depth);
                    return .{ .bool = try asBool(b.right, r) };
                },
                .match, .not_match => {
                    const l = try unwrapScalar(b.left, try evalExpr(ctx, b.left, env, depth), false);
                    const r = try unwrapScalar(b.right, try evalExpr(ctx, b.right, env, depth), false);
                    if (l != .string or r != .string) return failExpr(e, error.TypeMismatch);
                    const matched = re_match.matchRe(r.string.bytes, l.string.bytes) catch
                        return failExpr(e, error.BadRegex);
                    return .{ .bool = if (b.op == .match) matched else !matched };
                },
                .eq, .neq => {
                    const l = try unwrapScalar(b.left, try evalExpr(ctx, b.left, env, depth), false);
                    const r = try unwrapScalar(b.right, try evalExpr(ctx, b.right, env, depth), false);
                    const eq = l.eql(r) catch |err| return failExpr(e, err);
                    return .{ .bool = if (b.op == .eq) eq else !eq };
                },
                .gt, .ge, .lt, .le => {
                    const l = try unwrapScalar(b.left, try evalExpr(ctx, b.left, env, depth), false);
                    const r = try unwrapScalar(b.right, try evalExpr(ctx, b.right, env, depth), false);
                    if (l != .int or r != .int) return failExpr(e, error.TypeMismatch);
                    return .{ .bool = cmpInt(b.op, l.int, r.int) };
                },
            }
        },
        .record => |fields| {
            const out_fields = try ctx.allocator.alloc(value.RecordField, fields.len);
            var seen: std.StringHashMapUnmanaged(void) = .empty;
            defer seen.deinit(ctx.allocator);
            for (fields, 0..) |f, i| {
                if ((try seen.fetchPut(ctx.allocator, f.name, {})) != null)
                    return failExpr(f.expr, error.DuplicateField);
                out_fields[i] = .{
                    .name = f.name,
                    .value = try evalExpr(ctx, f.expr, env, depth),
                };
            }
            const rec = try ctx.allocator.create(value.Record);
            rec.* = .{ .fields = out_fields };
            return .{ .record = rec };
        },
    };
}

// --- sink -------------------------------------------------------------------

/// Write a select/group result value to the sink (unwraps singleton Seq).
pub fn sinkPrint(ctx: Ctx, v: Value) Error!void {
    switch (v) {
        .string, .int, .bool, .file, .dir, .hash => try sinkLine(ctx, v),
        .record => |rec| {
            // Exactly one line per field (§7): no expanding nested Seq/Record.
            for (rec.fields) |f| try sinkFieldLine(ctx, f.value);
        },
        .seq => |s| {
            for (s.items) |item| try sinkPrint(ctx, item);
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
        .file => |f| try ctx.out.writeAll(f.path),
        .dir => |d| try ctx.out.writeAll(d.path),
        .hash => |h| try ctx.out.writeAll(h),
        .record, .seq => unreachable,
    }
    try ctx.out.writeAll("\n");
    // Progressive output: main() uses a large stdout buffer; flush each sunk line.
    ctx.out.flush() catch return error.WriteFailed;
}

// --- sources ----------------------------------------------------------------

fn openAs(ctx: Ctx, kind: plan.SourceKind, path_or_payload: []const u8) Error!Value {
    switch (kind) {
        .string => return Value.plainStr(path_or_payload),
        .hash => return .{ .hash = path_or_payload },
        .file => {
            // §3.3: regular file only — openFile succeeds on directories on Linux.
            var f = std.Io.Dir.cwd().openFile(ctx.io, path_or_payload, .{}) catch return ioFail(path_or_payload);
            defer f.close(ctx.io);
            const st = f.stat(ctx.io) catch return ioFail(path_or_payload);
            if (st.kind != .file) return ioFail(path_or_payload);
            return .{ .file = .{ .path = path_or_payload } };
        },
        .dir => {
            var d = std.Io.Dir.cwd().openDir(ctx.io, path_or_payload, .{}) catch return ioFail(path_or_payload);
            d.close(ctx.io);
            return .{ .dir = .{ .path = path_or_payload } };
        },
    }
}

// --- directory file iterator (§3.4 / §4.6) ----------------------------------

/// Yields regular-file paths under a Dir one at a time (walk order, no sort).
const DirFileIter = struct {
    io: std.Io,
    dir: value.DirVal,
    root: std.Io.Dir,
    state: union(enum) {
        flat: std.Io.Dir.Iterator,
        tree: std.Io.Dir.SelectiveWalker,
    },

    fn init(allocator: std.mem.Allocator, io: std.Io, dir: value.DirVal) Error!DirFileIter {
        var root = std.Io.Dir.cwd().openDir(io, dir.path, .{ .iterate = true }) catch return ioFail(dir.path);
        errdefer root.close(io);
        if (dir.max_depth == 0) {
            return .{
                .io = io,
                .dir = dir,
                .root = root,
                .state = .{ .flat = root.iterate() },
            };
        }
        const walker = root.walkSelectively(allocator) catch return error.OutOfMemory;
        return .{
            .io = io,
            .dir = dir,
            .root = root,
            .state = .{ .tree = walker },
        };
    }

    fn deinit(self: *DirFileIter) void {
        switch (self.state) {
            .flat => {},
            .tree => |*w| w.deinit(),
        }
        self.root.close(self.io);
    }

    /// Owned path allocated with `path_allocator`, or `null` when exhausted.
    fn next(self: *DirFileIter, path_allocator: std.mem.Allocator) Error!?[]const u8 {
        switch (self.state) {
            .flat => |*it| {
                while (true) {
                    const maybe = it.next(self.io) catch return ioFail(self.dir.path);
                    const entry = maybe orelse return null;
                    // DT_UNKNOWN entries resolve via no-follow stat (§3.4).
                    if (modes.dir.effectiveEntryKind(self.root, self.io, entry.name, entry.kind) != .file) continue;
                    return try std.fs.path.join(path_allocator, &.{ self.dir.path, entry.name });
                }
            },
            .tree => |*walker| {
                while (true) {
                    const maybe = walker.next(self.io) catch {
                        if (self.dir.skip_errors) continue;
                        return ioFail(self.dir.path);
                    };
                    const entry = maybe orelse return null;
                    // DT_UNKNOWN entries resolve via no-follow stat (§3.4).
                    const kind = modes.dir.effectiveEntryKind(entry.dir, self.io, entry.basename, entry.kind);
                    if (kind == .directory) {
                        const unlimited = self.dir.max_depth == null;
                        const within = if (self.dir.max_depth) |n| entry.depth() <= n else false;
                        if (unlimited or within) {
                            walker.enter(self.io, entry) catch {
                                if (self.dir.skip_errors) continue;
                                const full = try std.fs.path.join(path_allocator, &.{ self.dir.path, entry.path });
                                return ioFail(full);
                            };
                        }
                        continue;
                    }
                    if (kind != .file) continue;
                    return try std.fs.path.join(path_allocator, &.{ self.dir.path, entry.path });
                }
            },
        }
    }
};

fn expectItem(kind: plan.SourceKind, item: Value) Error!void {
    const got = item.sourceKind() orelse return error.TypeMismatch;
    if (got != kind) return error.TypeMismatch;
}

/// Resolved `from`/`join` source (§3.3 / §3.4): stream or materialize via the same rules.
const BoundSource = union(enum) {
    /// Seq items already checked against the declared range kind.
    seq: []const Value,
    /// `from file f in <Dir>` / `d.tree()` — walk yields File paths.
    dir_files: value.DirVal,
    /// Singleton string path/digest opened as the range kind.
    one: Value,
};

fn bindSource(
    ctx: Ctx,
    kind: plan.SourceKind,
    source: *Expr,
    env: *Env,
    depth: u32,
) Error!BoundSource {
    const src_val = try evalExpr(ctx, source, env, depth);
    if (src_val == .seq) {
        for (src_val.seq.items) |item| {
            expectItem(kind, item) catch |err| return failExpr(source, err);
        }
        return .{ .seq = src_val.seq.items };
    }
    if (kind == .file and src_val == .dir) {
        return .{ .dir_files = src_val.dir };
    }
    // Singleton: string path/digest only — no File/Dir/Hash cross-kind coercion (§3.3).
    const payload = switch (src_val) {
        .string => |s| s.bytes,
        else => return failExpr(source, error.TypeMismatch),
    };
    const bound = openAs(ctx, kind, payload) catch |err| return failExpr(source, err);
    return .{ .one = bound };
}

fn collectDirFiles(ctx: Ctx, dir: value.DirVal) Error![]Value {
    var iter = try DirFileIter.init(ctx.allocator, ctx.io, dir);
    defer iter.deinit();
    var list: std.ArrayListUnmanaged(Value) = .empty;
    errdefer list.deinit(ctx.allocator);
    while (try iter.next(ctx.allocator)) |full| {
        try list.append(ctx.allocator, .{ .file = .{ .path = full } });
    }
    return try list.toOwnedSlice(ctx.allocator);
}

/// Materialize a bound source into a flat value list (join inners).
fn expandSourceValues(
    ctx: Ctx,
    kind: plan.SourceKind,
    source: *Expr,
    env: *Env,
    depth: u32,
) Error![]Value {
    return switch (try bindSource(ctx, kind, source, env, depth)) {
        .seq => |items| try ctx.allocator.dupe(Value, items),
        .dir_files => |dir| try collectDirFiles(ctx, dir),
        .one => |v| blk: {
            const slice = try ctx.allocator.alloc(Value, 1);
            slice[0] = v;
            break :blk slice;
        },
    };
}

// --- pull operator pipeline -------------------------------------------------

const DriveMode = union(enum) {
    sink,
    collect: *std.ArrayListUnmanaged(Value),
};

/// One value from a terminal producer; `env` is set by `project` for sinkSelect.
const Produced = struct {
    value: Value,
    env: ?*Env = null,
};

const PipeCtx = struct {
    io: std.Io,
    out: *std.Io.Writer,
    depth: u32,
    parent: std.mem.Allocator,
    row_arena: *std.heap.ArenaAllocator,
    /// Allocator for the current row (row arena while streaming files, else parent).
    row_alloc: std.mem.Allocator,
    /// Script env for terminal `into id;` binds (same pointer as drive outer).
    script: *Env,
    /// Allocator for values stored in `script` (survives query arenas).
    script_alloc: std.mem.Allocator,

    fn ctx(self: *PipeCtx) Ctx {
        return .{ .allocator = self.row_alloc, .io = self.io, .out = self.out };
    }
};

const Op = union(enum) {
    from: FromOp,
    where: WhereOp,
    let: LetOp,
    join: JoinOp,
    order_by: OrderByOp,
    group_into: GroupIntoOp,
    select_into: SelectIntoOp,
    project: ProjectOp,
    group_out: GroupOutOp,
    script_bind: ScriptBindOp,
};

/// Root (`child == null`) or nested `from`: expand source over one outer env at a time.
const FromOp = struct {
    from: *const plan.From,
    child: ?*Op,
    /// Drive outer env when `child == null` (root scan); cleared after take-once.
    script_outer: ?*Env = null,
    phase: enum { need_outer, in_dir, in_seq, in_one } = .need_outer,
    stable_outer: Env = .{},
    dir_iter: ?DirFileIter = null,
    seq_items: []const Value = &.{},
    seq_index: usize = 0,
    row: Env = .{},

    fn open(self: *FromOp, pc: *PipeCtx, outer: *Env) Error!void {
        self.phase = .need_outer;
        if (self.child) |c| {
            try opOpen(c, pc, outer);
        } else {
            self.script_outer = outer;
        }
    }

    fn nextEnv(self: *FromOp, pc: *PipeCtx) Error!?*Env {
        while (true) {
            switch (self.phase) {
                .need_outer => {
                    const outer: *Env = if (self.child) |c|
                        (try opNextEnv(c, pc)) orelse return null
                    else blk: {
                        const o = self.script_outer orelse return null;
                        self.script_outer = null;
                        break :blk o;
                    };
                    // Bind with parent, then persist Dir/Seq payloads there: they must
                    // outlive `row_arena.reset` in `.in_dir` / `.in_seq`. Env lookups
                    // (Dir via Seq/nested query/`into`) may still alias row-arena paths.
                    const bind_ctx: Ctx = .{ .allocator = pc.parent, .io = pc.io, .out = pc.out };
                    const bound = try bindSource(bind_ctx, self.from.kind, self.from.source, outer, pc.depth);
                    switch (bound) {
                        .dir_files => |dir| {
                            self.stable_outer = try outer.dupe(pc.parent);
                            const stable_dir: value.DirVal = .{
                                .path = try pc.parent.dupe(u8, dir.path),
                                .max_depth = dir.max_depth,
                                .skip_errors = dir.skip_errors,
                            };
                            errdefer pc.parent.free(stable_dir.path);
                            self.dir_iter = try DirFileIter.init(pc.parent, pc.io, stable_dir);
                            self.phase = .in_dir;
                        },
                        .seq => |items| {
                            self.stable_outer = try outer.dupe(pc.parent);
                            const owned = try pc.parent.alloc(Value, items.len);
                            errdefer pc.parent.free(owned);
                            for (items, 0..) |item, i| {
                                owned[i] = try item.dupe(pc.parent);
                            }
                            self.seq_items = owned;
                            self.seq_index = 0;
                            self.phase = .in_seq;
                        },
                        .one => |v| {
                            pc.row_alloc = pc.parent;
                            self.row = try outer.dupe(pc.parent);
                            try self.row.put(pc.parent, self.from.range, try v.dupe(pc.parent));
                            self.phase = .in_one;
                        },
                    }
                },
                .in_dir => {
                    const iter = &self.dir_iter.?;
                    _ = pc.row_arena.reset(.retain_capacity);
                    const ralloc = pc.row_arena.allocator();
                    pc.row_alloc = ralloc;
                    const path = (try iter.next(ralloc)) orelse {
                        iter.deinit();
                        self.dir_iter = null;
                        self.phase = .need_outer;
                        continue;
                    };
                    self.row = try self.stable_outer.clone(ralloc);
                    try self.row.put(ralloc, self.from.range, .{ .file = .{ .path = path } });
                    return &self.row;
                },
                .in_seq => {
                    if (self.seq_index >= self.seq_items.len) {
                        self.phase = .need_outer;
                        continue;
                    }
                    _ = pc.row_arena.reset(.retain_capacity);
                    const ralloc = pc.row_arena.allocator();
                    pc.row_alloc = ralloc;
                    const item = self.seq_items[self.seq_index];
                    self.seq_index += 1;
                    self.row = try self.stable_outer.clone(ralloc);
                    try self.row.put(ralloc, self.from.range, try item.dupe(ralloc));
                    return &self.row;
                },
                .in_one => {
                    self.phase = .need_outer;
                    return &self.row;
                },
            }
        }
    }

    fn close(self: *FromOp, pc: *PipeCtx) void {
        if (self.dir_iter) |*it| {
            it.deinit();
            self.dir_iter = null;
        }
        if (self.child) |c| opClose(c, pc);
    }
};

const WhereOp = struct {
    pred: *const Expr,
    child: *Op,

    fn open(self: *WhereOp, pc: *PipeCtx, outer: *Env) Error!void {
        try opOpen(self.child, pc, outer);
    }

    fn nextEnv(self: *WhereOp, pc: *PipeCtx) Error!?*Env {
        while (true) {
            const env = (try opNextEnv(self.child, pc)) orelse return null;
            const pred = try evalExpr(pc.ctx(), self.pred, env, pc.depth);
            if (try asBool(self.pred, pred)) return env;
        }
    }

    fn close(self: *WhereOp, pc: *PipeCtx) void {
        opClose(self.child, pc);
    }
};

const LetOp = struct {
    name: []const u8,
    expr: *const Expr,
    child: *Op,

    fn open(self: *LetOp, pc: *PipeCtx, outer: *Env) Error!void {
        try opOpen(self.child, pc, outer);
    }

    fn nextEnv(self: *LetOp, pc: *PipeCtx) Error!?*Env {
        const env = (try opNextEnv(self.child, pc)) orelse return null;
        const v = try evalExpr(pc.ctx(), self.expr, env, pc.depth);
        try env.put(pc.row_alloc, self.name, v);
        return env;
    }

    fn close(self: *LetOp, pc: *PipeCtx) void {
        opClose(self.child, pc);
    }
};

const SelectIntoOp = struct {
    name: []const u8,
    expr: *const Expr,
    child: *Op,
    cont: Env = .{},

    fn open(self: *SelectIntoOp, pc: *PipeCtx, outer: *Env) Error!void {
        try opOpen(self.child, pc, outer);
    }

    fn nextEnv(self: *SelectIntoOp, pc: *PipeCtx) Error!?*Env {
        const env = (try opNextEnv(self.child, pc)) orelse return null;
        const v = try evalExpr(pc.ctx(), self.expr, env, pc.depth);
        self.cont = .{};
        try self.cont.put(pc.row_alloc, self.name, v);
        return &self.cont;
    }

    fn close(self: *SelectIntoOp, pc: *PipeCtx) void {
        opClose(self.child, pc);
    }
};

/// True when `op` or its descendants bind `name` into the row env (range / let / into).
fn opBindsName(op: *const Op, name: []const u8) bool {
    switch (op.*) {
        .from => |*f| {
            if (std.mem.eql(u8, f.from.range, name)) return true;
            return if (f.child) |c| opBindsName(c, name) else false;
        },
        .where => |*w| return opBindsName(w.child, name),
        .let => |*l| return std.mem.eql(u8, l.name, name) or opBindsName(l.child, name),
        .join => |*j| {
            if (std.mem.eql(u8, j.join.range, name)) return true;
            if (j.join.group_into) |g| {
                if (std.mem.eql(u8, g, name)) return true;
            }
            return opBindsName(j.child, name);
        },
        .order_by => |*o| return opBindsName(o.child, name),
        .group_into => |*g| return std.mem.eql(u8, g.into_name, name) or opBindsName(g.child, name),
        .select_into => |*s| return std.mem.eql(u8, s.name, name) or opBindsName(s.child, name),
        .project => |*p| return opBindsName(p.child, name),
        .group_out => |*g| return opBindsName(g.child, name),
        .script_bind => |*s| return opBindsName(s.child, name),
    }
}

/// True when join `in` source does not depend on the current outer row.
/// Literals and unshadowed script-bound names are stable across outers; nested
/// queries always rematerialize (avoids a full plan free-name walk).
fn exprJoinSourceStable(e: *const Expr, script: *const Env, outer_ops: *const Op) bool {
    return switch (e.kind) {
        .string_lit, .int_lit, .bool_lit => true,
        .nested_query => false,
        // Pipeline range/let/into bindings shadow script; only pure script refs cache.
        .name => |n| script.get(n) != null and !opBindsName(outer_ops, n),
        .prop => |p| exprJoinSourceStable(p.recv, script, outer_ops),
        .method => |m| blk: {
            if (!exprJoinSourceStable(m.recv, script, outer_ops)) break :blk false;
            for (m.args) |a| {
                if (!exprJoinSourceStable(a, script, outer_ops)) break :blk false;
            }
            break :blk true;
        },
        .not => |inner| exprJoinSourceStable(inner, script, outer_ops),
        .binary => |b| exprJoinSourceStable(b.left, script, outer_ops) and exprJoinSourceStable(b.right, script, outer_ops),
        .record => |fields| blk: {
            for (fields) |f| {
                if (!exprJoinSourceStable(f.expr, script, outer_ops)) break :blk false;
            }
            break :blk true;
        },
    };
}

const JoinOp = struct {
    join: *const plan.Join,
    child: *Op,
    inners: []Value = &.{},
    /// When true, `inners` is reused across outer rows (source is script-stable).
    inners_cached: bool = false,
    inner_index: usize = 0,
    outer_env: ?*Env = null,
    /// Cached `normalize(outer_key)` for the current outer row.
    outer_key_val: Value = .{ .bool = false },
    /// Scratch / yield env: one clone of the outer, range binding overwritten per candidate.
    row: Env = .{},
    /// Group-join yield env: parent-owned clone of the outer row plus the `into`
    /// binding. The child row env's map lives in the row arena, which the next
    /// pull recycles, so it must not be mutated or yielded with `pc.parent`.
    group_env: Env = .{},

    fn open(self: *JoinOp, pc: *PipeCtx, outer: *Env) Error!void {
        self.clearInners(pc);
        self.inner_index = 0;
        self.outer_env = null;
        try opOpen(self.child, pc, outer);
    }

    fn clearInners(self: *JoinOp, pc: *PipeCtx) void {
        if (self.inners.len != 0) pc.parent.free(self.inners);
        self.inners = &.{};
        self.inners_cached = false;
    }

    fn ensureInners(self: *JoinOp, pc: *PipeCtx, outer: *Env) Error!void {
        if (self.inners_cached) return;
        self.clearInners(pc);
        const c: Ctx = .{ .allocator = pc.parent, .io = pc.io, .out = pc.out };
        self.inners = try expandSourceValues(c, self.join.kind, self.join.source, outer, pc.depth);
        self.inners_cached = exprJoinSourceStable(self.join.source, pc.script, self.child);
    }

    fn prepareOuter(self: *JoinOp, pc: *PipeCtx, outer: *Env) Error!void {
        try self.ensureInners(pc, outer);
        self.inner_index = 0;
        const c: Ctx = .{ .allocator = pc.parent, .io = pc.io, .out = pc.out };
        const raw = try evalExpr(c, self.join.outer_key, outer, pc.depth);
        self.outer_key_val = try unwrapScalar(self.join.outer_key, raw, false);
        self.row = try outer.clone(pc.parent);
    }

    fn nextEnv(self: *JoinOp, pc: *PipeCtx) Error!?*Env {
        while (true) {
            if (self.outer_env == null) {
                self.outer_env = (try opNextEnv(self.child, pc)) orelse return null;
                try self.prepareOuter(pc, self.outer_env.?);

                if (self.join.group_into) |gname| {
                    const c: Ctx = .{ .allocator = pc.parent, .io = pc.io, .out = pc.out };
                    var matches: std.ArrayListUnmanaged(Value) = .empty;
                    defer matches.deinit(pc.parent);
                    for (self.inners) |inner_val| {
                        try self.row.put(pc.parent, self.join.range, inner_val);
                        if (try keyEqualsCached(c, self.outer_key_val, self.join.outer_key, self.join.inner_key, &self.row, pc.depth)) {
                            try matches.append(pc.parent, inner_val);
                        }
                    }
                    const seq = try pc.parent.create(value.Seq);
                    seq.* = .{ .items = try pc.parent.dupe(Value, matches.items) };
                    self.group_env = try self.outer_env.?.clone(pc.parent);
                    try self.group_env.put(pc.parent, gname, .{ .seq = seq });
                    if (!self.inners_cached) self.clearInners(pc);
                    pc.row_alloc = pc.parent;
                    self.outer_env = null;
                    return &self.group_env;
                }
            }
            while (self.inner_index < self.inners.len) {
                const inner_val = self.inners[self.inner_index];
                self.inner_index += 1;
                pc.row_alloc = pc.parent;
                try self.row.put(pc.parent, self.join.range, inner_val);
                if (try keyEqualsCached(pc.ctx(), self.outer_key_val, self.join.outer_key, self.join.inner_key, &self.row, pc.depth)) {
                    return &self.row;
                }
            }
            if (!self.inners_cached) self.clearInners(pc);
            self.outer_env = null;
        }
    }

    fn close(self: *JoinOp, pc: *PipeCtx) void {
        self.clearInners(pc);
        self.outer_env = null;
        opClose(self.child, pc);
    }
};

const OrderByOp = struct {
    keys: []plan.OrderKey,
    child: *Op,
    rows: []Env = &.{},
    index: usize = 0,
    ready: bool = false,

    fn open(self: *OrderByOp, pc: *PipeCtx, outer: *Env) Error!void {
        self.rows = &.{};
        self.index = 0;
        self.ready = false;
        try opOpen(self.child, pc, outer);
    }

    fn materialize(self: *OrderByOp, pc: *PipeCtx) Error!void {
        const owned = try collectChildEnvs(pc, self.child);
        const c: Ctx = .{ .allocator = pc.parent, .io = pc.io, .out = pc.out };
        self.rows = try orderRows(c, owned, self.keys, pc.depth);
        pc.parent.free(owned);
    }

    fn nextEnv(self: *OrderByOp, pc: *PipeCtx) Error!?*Env {
        if (!self.ready) {
            try self.materialize(pc);
            self.ready = true;
        }
        if (self.index >= self.rows.len) return null;
        pc.row_alloc = pc.parent;
        const env = &self.rows[self.index];
        self.index += 1;
        return env;
    }

    fn close(self: *OrderByOp, pc: *PipeCtx) void {
        if (self.rows.len != 0) pc.parent.free(self.rows);
        self.rows = &.{};
        opClose(self.child, pc);
    }
};

const GroupIntoOp = struct {
    proj: *const Expr,
    key: *const Expr,
    into_name: []const u8,
    child: *Op,
    rows: []Env = &.{},
    index: usize = 0,
    ready: bool = false,

    fn open(self: *GroupIntoOp, pc: *PipeCtx, outer: *Env) Error!void {
        self.rows = &.{};
        self.index = 0;
        self.ready = false;
        try opOpen(self.child, pc, outer);
    }

    fn materialize(self: *GroupIntoOp, pc: *PipeCtx) Error!void {
        const collected = try collectChildEnvs(pc, self.child);
        defer pc.parent.free(collected);
        const c: Ctx = .{ .allocator = pc.parent, .io = pc.io, .out = pc.out };
        const groups = try buildGroups(c, collected, self.proj, self.key, pc.depth);
        const envs = try pc.parent.alloc(Env, groups.len);
        for (groups, 0..) |gv, i| {
            envs[i] = .{};
            try envs[i].put(pc.parent, self.into_name, gv);
        }
        pc.parent.free(groups);
        self.rows = envs;
    }

    fn nextEnv(self: *GroupIntoOp, pc: *PipeCtx) Error!?*Env {
        if (!self.ready) {
            try self.materialize(pc);
            self.ready = true;
        }
        if (self.index >= self.rows.len) return null;
        pc.row_alloc = pc.parent;
        const env = &self.rows[self.index];
        self.index += 1;
        return env;
    }

    fn close(self: *GroupIntoOp, pc: *PipeCtx) void {
        if (self.rows.len != 0) pc.parent.free(self.rows);
        self.rows = &.{};
        opClose(self.child, pc);
    }
};

const ProjectOp = struct {
    expr: *const Expr,
    child: *Op,

    fn open(self: *ProjectOp, pc: *PipeCtx, outer: *Env) Error!void {
        try opOpen(self.child, pc, outer);
    }

    fn nextValue(self: *ProjectOp, pc: *PipeCtx) Error!?Produced {
        const env = (try opNextEnv(self.child, pc)) orelse return null;
        return .{
            .value = try evalExpr(pc.ctx(), self.expr, env, pc.depth),
            .env = env,
        };
    }

    fn close(self: *ProjectOp, pc: *PipeCtx) void {
        opClose(self.child, pc);
    }
};

const GroupOutOp = struct {
    proj: *const Expr,
    key: *const Expr,
    child: *Op,
    groups: []Value = &.{},
    index: usize = 0,
    ready: bool = false,

    fn open(self: *GroupOutOp, pc: *PipeCtx, outer: *Env) Error!void {
        self.groups = &.{};
        self.index = 0;
        self.ready = false;
        try opOpen(self.child, pc, outer);
    }

    fn materialize(self: *GroupOutOp, pc: *PipeCtx) Error!void {
        const collected = try collectChildEnvs(pc, self.child);
        defer pc.parent.free(collected);
        const c: Ctx = .{ .allocator = pc.parent, .io = pc.io, .out = pc.out };
        self.groups = try buildGroups(c, collected, self.proj, self.key, pc.depth);
    }

    fn nextValue(self: *GroupOutOp, pc: *PipeCtx) Error!?Produced {
        if (!self.ready) {
            try self.materialize(pc);
            self.ready = true;
        }
        if (self.index >= self.groups.len) return null;
        const v = self.groups[self.index];
        self.index += 1;
        return .{ .value = v };
    }

    fn close(self: *GroupOutOp, pc: *PipeCtx) void {
        if (self.groups.len != 0) pc.parent.free(self.groups);
        self.groups = &.{};
        opClose(self.child, pc);
    }
};

/// Terminal `… into id;` — collect values from a producer (`project` / `group_out`) into the script env.
const ScriptBindOp = struct {
    name: []const u8,
    child: *Op,
    done: bool = false,

    fn open(self: *ScriptBindOp, pc: *PipeCtx, outer: *Env) Error!void {
        self.done = false;
        try opOpen(self.child, pc, outer);
    }

    fn nextValue(self: *ScriptBindOp, pc: *PipeCtx) Error!?Produced {
        if (self.done) return null;
        self.done = true;

        var list: std.ArrayListUnmanaged(Value) = .empty;
        defer list.deinit(pc.parent);
        while (try opNextValue(self.child, pc)) |row| {
            try list.append(pc.parent, try row.value.dupe(pc.parent));
        }
        try bindScriptValues(pc, self.name, list.items);
        return null;
    }

    fn close(self: *ScriptBindOp, pc: *PipeCtx) void {
        opClose(self.child, pc);
    }
};

fn bindScriptValues(pc: *PipeCtx, name: []const u8, items: []const Value) Error!void {
    const binding: Value = if (items.len == 1)
        items[0]
    else blk: {
        const seq = try pc.parent.create(value.Seq);
        seq.* = .{ .items = try pc.parent.dupe(Value, items) };
        break :blk .{ .seq = seq };
    };
    const gop = try pc.script.map.getOrPut(pc.script_alloc, name);
    if (!gop.found_existing) {
        gop.key_ptr.* = try pc.script_alloc.dupe(u8, name);
    }
    gop.value_ptr.* = try binding.dupe(pc.script_alloc);
}

fn collectChildEnvs(pc: *PipeCtx, child: *Op) Error![]Env {
    var list: std.ArrayListUnmanaged(Env) = .empty;
    errdefer list.deinit(pc.parent);
    while (try opNextEnv(child, pc)) |env| {
        try list.append(pc.parent, try env.dupe(pc.parent));
    }
    return try list.toOwnedSlice(pc.parent);
}

fn opOpen(op: *Op, pc: *PipeCtx, outer: *Env) Error!void {
    switch (op.*) {
        inline else => |*s| try s.open(pc, outer),
    }
}

fn opClose(op: *Op, pc: *PipeCtx) void {
    switch (op.*) {
        inline else => |*s| s.close(pc),
    }
}

fn opNextEnv(op: *Op, pc: *PipeCtx) Error!?*Env {
    return switch (op.*) {
        .project, .group_out, .script_bind => unreachable,
        inline else => |*s| s.nextEnv(pc),
    };
}

fn opNextValue(op: *Op, pc: *PipeCtx) Error!?Produced {
    return switch (op.*) {
        inline .project, .group_out, .script_bind => |*s| s.nextValue(pc),
        else => unreachable,
    };
}

fn createOp(allocator: std.mem.Allocator, op: Op) Error!*Op {
    const p = try allocator.create(Op);
    p.* = op;
    return p;
}

fn wrapClause(allocator: std.mem.Allocator, clause: *const plan.Clause, input: *Op) Error!*Op {
    switch (clause.*) {
        .where => |w| {
            const op = try createOp(allocator, .{ .where = .{ .pred = w.pred, .child = input } });
            return wrapClause(allocator, w.then, op);
        },
        .let => |l| {
            const op = try createOp(allocator, .{ .let = .{ .name = l.name, .expr = l.expr, .child = input } });
            return wrapClause(allocator, l.then, op);
        },
        .from => |f| {
            const op = try createOp(allocator, .{ .from = .{ .from = f, .child = input } });
            return wrapClause(allocator, f.then, op);
        },
        .join => |j| {
            const op = try createOp(allocator, .{ .join = .{ .join = j, .child = input } });
            return wrapClause(allocator, j.then, op);
        },
        .order_by => |o| {
            const op = try createOp(allocator, .{ .order_by = .{ .keys = o.keys, .child = input } });
            return wrapClause(allocator, o.then, op);
        },
        .group_by => |g| {
            if (g.into) |into| {
                if (into.body) |body| {
                    const op = try createOp(allocator, .{
                        .group_into = .{ .proj = g.proj, .key = g.key, .into_name = into.name, .child = input },
                    });
                    return wrapClause(allocator, body, op);
                }
                const out = try createOp(allocator, .{
                    .group_out = .{ .proj = g.proj, .key = g.key, .child = input },
                });
                return wrapScriptBind(allocator, into.name, out);
            }
            return createOp(allocator, .{ .group_out = .{ .proj = g.proj, .key = g.key, .child = input } });
        },
        .select => |sel| {
            if (sel.into) |into| {
                if (into.body) |body| {
                    const op = try createOp(allocator, .{
                        .select_into = .{ .name = into.name, .expr = sel.expr, .child = input },
                    });
                    return wrapClause(allocator, body, op);
                }
                const proj = try createOp(allocator, .{ .project = .{ .expr = sel.expr, .child = input } });
                return wrapScriptBind(allocator, into.name, proj);
            }
            return createOp(allocator, .{ .project = .{ .expr = sel.expr, .child = input } });
        },
    }
}

fn wrapScriptBind(allocator: std.mem.Allocator, name: []const u8, child: *Op) Error!*Op {
    return createOp(allocator, .{ .script_bind = .{ .name = name, .child = child } });
}

fn buildRoot(allocator: std.mem.Allocator, root: *const plan.From) Error!*Op {
    const scan = try createOp(allocator, .{ .from = .{ .from = root, .child = null } });
    return wrapClause(allocator, root.then, scan);
}

fn evalQueryValues(ctx: Ctx, query: *const plan.From, outer: *Env, depth: u32) Error![]Value {
    var out: std.ArrayListUnmanaged(Value) = .empty;
    errdefer out.deinit(ctx.allocator);
    // Nested queries do not script-bind; reuse ctx.allocator as a dummy script_alloc.
    try runPipeline(ctx, query, outer, depth, .{ .collect = &out }, ctx.allocator);
    return try out.toOwnedSlice(ctx.allocator);
}

/// Shared entry for sink (`run`) and nested collect (`evalQueryValues`).
fn runPipeline(
    ctx: Ctx,
    root: *const plan.From,
    outer: *Env,
    depth: u32,
    mode: DriveMode,
    script_alloc: std.mem.Allocator,
) Error!void {
    var op_arena = std.heap.ArenaAllocator.init(ctx.allocator);
    defer op_arena.deinit();
    var row_arena = std.heap.ArenaAllocator.init(ctx.allocator);
    defer row_arena.deinit();

    var pc: PipeCtx = .{
        .io = ctx.io,
        .out = ctx.out,
        .depth = depth,
        .parent = ctx.allocator,
        .row_arena = &row_arena,
        .row_alloc = ctx.allocator,
        .script = outer,
        .script_alloc = script_alloc,
    };

    const op = try buildRoot(op_arena.allocator(), root);
    try opOpen(op, &pc, outer);
    defer opClose(op, &pc);

    while (try opNextValue(op, &pc)) |row| {
        switch (mode) {
            .sink => switch (op.*) {
                .project => |*p| try sinkSelect(ctx, p.expr, row.env.?, row.value),
                .group_out => try sinkPrint(ctx, row.value),
                .script_bind => unreachable,
                else => unreachable,
            },
            .collect => |out| try out.append(ctx.allocator, try row.value.dupe(ctx.allocator)),
        }
    }
}

fn orderRows(ctx: Ctx, rows: []Env, order_keys: []plan.OrderKey, depth: u32) Error![]Env {
    const Indexed = struct {
        env: Env,
        keys: []Value,
        index: usize,
    };
    const indexed = try ctx.allocator.alloc(Indexed, rows.len);
    defer {
        for (indexed) |*ix| ctx.allocator.free(ix.keys);
        ctx.allocator.free(indexed);
    }
    for (rows, 0..) |*row, i| {
        const ks = try ctx.allocator.alloc(Value, order_keys.len);
        for (order_keys, 0..) |ok, j| {
            const raw = try evalExpr(ctx, ok.expr, row, depth);
            ks[j] = try unwrapScalar(ok.expr, raw, false);
        }
        indexed[i] = .{ .env = row.*, .keys = ks, .index = i };
    }

    // Reject mixed/incomparable key kinds before sort (std.mem.sort cannot bubble errors).
    if (indexed.len > 1) {
        for (order_keys, 0..) |ok, col| {
            const baseline = indexed[0].keys[col];
            for (indexed[1..]) |ix| {
                _ = baseline.compare(ix.keys[col]) catch |err| return failExpr(ok.expr, err);
            }
        }
    }

    const StableCtx = struct {
        order_keys: []plan.OrderKey,
        fn less(self: @This(), a: Indexed, b: Indexed) bool {
            for (self.order_keys, 0..) |ok, i| {
                const cmp = a.keys[i].compare(b.keys[i]) catch
                    @panic("invariant violated: order keys must be pre-validated comparable; report as bug");
                if (cmp == .eq) continue;
                if (ok.descending) return cmp == .gt;
                return cmp == .lt;
            }
            return a.index < b.index;
        }
    };
    std.mem.sort(Indexed, indexed, StableCtx{ .order_keys = order_keys }, StableCtx.less);

    const out = try ctx.allocator.alloc(Env, indexed.len);
    for (indexed, 0..) |ix, i| out[i] = ix.env;
    return out;
}

const GroupBucket = struct {
    key: Value,
    items: std.ArrayListUnmanaged(Value) = .empty,
};

fn buildGroups(
    ctx: Ctx,
    rows: []Env,
    proj: *const Expr,
    key_expr: *const Expr,
    depth: u32,
) Error![]Value {
    var buckets: std.ArrayListUnmanaged(GroupBucket) = .empty;
    defer {
        for (buckets.items) |*b| b.items.deinit(ctx.allocator);
        buckets.deinit(ctx.allocator);
    }

    for (rows) |*row| {
        const k = try unwrapScalar(key_expr, try evalExpr(ctx, key_expr, row, depth), false);
        const p = try evalExpr(ctx, proj, row, depth);
        var found: ?usize = null;
        for (buckets.items, 0..) |b, i| {
            const same = b.key.eql(k) catch |err| return failExpr(key_expr, err);
            if (same) {
                found = i;
                break;
            }
        }
        if (found) |i| {
            try buckets.items[i].items.append(ctx.allocator, p);
        } else {
            var bucket: GroupBucket = .{ .key = k };
            try bucket.items.append(ctx.allocator, p);
            try buckets.append(ctx.allocator, bucket);
        }
    }

    const out = try ctx.allocator.alloc(Value, buckets.items.len);
    for (buckets.items, 0..) |*b, i| {
        const seq = try ctx.allocator.create(value.Seq);
        seq.* = .{ .items = try ctx.allocator.dupe(Value, b.items.items) };
        const fields = try ctx.allocator.alloc(value.RecordField, 2);
        fields[0] = .{ .name = "key", .value = b.key };
        fields[1] = .{ .name = "items", .value = .{ .seq = seq } };
        const rec = try ctx.allocator.create(value.Record);
        rec.* = .{ .fields = fields };
        out[i] = .{ .record = rec };
    }
    return out;
}

fn keyEqualsCached(
    ctx: Ctx,
    outer_key_val: Value,
    outer_key: *const Expr,
    inner_key: *const Expr,
    inner: *Env,
    depth: u32,
) Error!bool {
    const r = try unwrapScalar(inner_key, try evalExpr(ctx, inner_key, inner, depth), false);
    return outer_key_val.eql(r) catch |err| return failExpr(outer_key, err);
}

fn sinkSelect(ctx: Ctx, e: *const Expr, env: *Env, v: Value) Error!void {
    // If projecting hash.algo, restore already printed via evalProp.
    if (e.kind == .prop and e.kind.prop.recv.kind == .name) {
        if (env.get(e.kind.prop.recv.kind.name)) |recv| {
            if (recv == .hash) return;
        }
    }
    try sinkPrint(ctx, v);
}

/// Execute a query plan. `script` is the shared multi-statement environment
/// (also used as the root outer env). `script_alloc` owns values stored by
/// terminal `into id;` binds and must outlive the query arena in `ctx.allocator`.
pub fn run(ctx: Ctx, query: *const plan.From, script: *Env, script_alloc: std.mem.Allocator) Error!void {
    try runPipeline(ctx, query, script, 0, .sink, script_alloc);
}

// --- tests ------------------------------------------------------------------

fn testCtx(allocator: std.mem.Allocator, out: *std.Io.Writer) Ctx {
    return .{
        .allocator = allocator,
        .io = std.testing.io,
        .out = out,
    };
}

test "eval string size and md5" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    var env: Env = .{};
    defer env.deinit(a);
    try env.put(a, "s", Value.plainStr("abc"));

    var recv: Expr = .{ .kind = .{ .name = "s" } };
    var size_e: Expr = .{ .kind = .{ .prop = .{ .recv = &recv, .prop = "size", .access = .size } } };
    // Act
    const size_v = try evalExpr(ctx, &size_e, &env, 0);
    // Assert
    try std.testing.expectEqual(@as(i64, 3), size_v.int);

    var md5_e: Expr = .{ .kind = .{ .prop = .{ .recv = &recv, .prop = "md5", .access = .hash_algo } } };
    const md5_v = try evalExpr(ctx, &md5_e, &env, 0);
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72", md5_v.string.bytes);
}

test "negative file window method is InvalidWindow" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    var env: Env = .{};
    defer env.deinit(a);
    try env.put(a, "f", Value.filePath("x"));

    var name_f: Expr = .{ .kind = .{ .name = "f" } };
    var neg: Expr = .{ .kind = .{ .int_lit = -1 } };
    var args = [_]*Expr{&neg};
    var call: Expr = .{ .kind = .{ .method = .{ .recv = &name_f, .name = "offset", .args = &args, .kind = .file_offset } } };

    try std.testing.expectError(error.InvalidWindow, evalExpr(ctx, &call, &env, 0));
}

test "negative tree depth is InvalidTreeDepth" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    var env: Env = .{};
    defer env.deinit(a);
    try env.put(a, "d", .{ .dir = .{ .path = "/tmp" } });

    var name_d: Expr = .{ .kind = .{ .name = "d" } };
    var neg: Expr = .{ .kind = .{ .int_lit = -1 } };
    var args = [_]*Expr{&neg};
    var call: Expr = .{ .kind = .{ .method = .{ .recv = &name_d, .name = "tree", .args = &args, .kind = .dir_tree } } };

    try std.testing.expectError(error.InvalidTreeDepth, evalExpr(ctx, &call, &env, 0));
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
    try sinkPrint(ctx, .{ .record = &rec });
    // Assert
    try std.testing.expectEqualStrings("aa\nbb\n", std.Io.Writer.buffered(&writer));
}

test "sink record rejects Seq field" {
    // Arrange — one line per field; nested Seq must not expand (§7)
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

    // Act / Assert
    try std.testing.expectError(error.TypeMismatch, sinkPrint(ctx, .{ .record = &rec }));
}

test "orderRows fails when key kinds differ across rows" {
    // Arrange — static compilation cannot see this mixed-key case.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [32]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    var row1: Env = .{};
    try row1.put(a, "k", .{ .int = 1 });
    var row2: Env = .{};
    try row2.put(a, "k", Value.plainStr("x"));
    var rows = [_]Env{ row1, row2 };

    var key_expr: Expr = .{
        .span = .{ .first_line = 1, .first_column = 8, .last_line = 1, .last_column = 9 },
        .kind = .{ .name = "k" },
    };
    var keys = [_]plan.OrderKey{.{ .expr = &key_expr }};

    // Act / Assert
    try std.testing.expectError(error.TypeMismatch, orderRows(ctx, &rows, &keys, 0));
}

test "from file in mixed sequence fails type check" {
    // Arrange — Seq(unknown) from mixed items is accepted statically; expectItem
    // rejects the wrong kind when expanding a file range over a named seq.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    const items = try a.alloc(Value, 2);
    items[0] = .{ .file = .{ .path = "a.txt" } };
    items[1] = Value.plainStr("not-a-file-value");
    const seq = try a.create(value.Seq);
    seq.* = .{ .items = items };

    var env: Env = .{};
    try env.put(a, "xs", .{ .seq = seq });

    var name_xs: Expr = .{ .kind = .{ .name = "xs" } };
    var name_f: Expr = .{ .kind = .{ .name = "f" } };
    const select = try a.create(plan.Select);
    select.* = .{ .expr = &name_f };
    const select_clause = try a.create(plan.Clause);
    select_clause.* = .{ .select = select };

    const from = try a.create(plan.From);
    from.* = .{
        .kind = .file,
        .range = "f",
        .source = &name_xs,
        .then = select_clause,
    };

    // Act / Assert
    try std.testing.expectError(
        error.TypeMismatch,
        runPipeline(ctx, from, &env, 0, .sink, a),
    );
}

test "group by rejects incomparable keys at runtime" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    var env1: Env = .{};
    try env1.put(a, "k", .{ .file = .{ .path = "/a" } });
    var env2: Env = .{};
    try env2.put(a, "k", Value.plainStr("b"));
    var rows = [_]Env{ env1, env2 };

    const key = try a.create(Expr);
    key.* = .{ .kind = .{ .name = "k" } };
    const proj = try a.create(Expr);
    proj.* = .{ .kind = .{ .name = "k" } };

    // Act / Assert
    try std.testing.expectError(error.TypeMismatch, buildGroups(ctx, rows[0..], proj, key, 0));
}

test "exprJoinSourceStable treats literals and unshadowed script names as stable" {
    // Arrange
    var script: Env = .{};
    defer script.deinit(std.testing.allocator);
    try script.put(std.testing.allocator, "files", Value.plainStr("x"));

    var lit: Expr = .{ .kind = .{ .string_lit = "a" } };
    var script_name: Expr = .{ .kind = .{ .name = "files" } };
    var row_name: Expr = .{ .kind = .{ .name = "id" } };
    var dummy_from: plan.From = undefined;
    var nested: Expr = .{ .kind = .{ .nested_query = &dummy_from } };

    var other_from: plan.From = .{
        .kind = .string,
        .range = "id",
        .source = &lit,
        .then = undefined,
    };
    var outer_no_shadow: Op = .{ .from = .{ .from = &other_from, .child = null } };

    var shadow_from: plan.From = .{
        .kind = .string,
        .range = "files",
        .source = &lit,
        .then = undefined,
    };
    var outer_shadows: Op = .{ .from = .{ .from = &shadow_from, .child = null } };

    // Act / Assert
    try std.testing.expect(exprJoinSourceStable(&lit, &script, &outer_no_shadow));
    try std.testing.expect(exprJoinSourceStable(&script_name, &script, &outer_no_shadow));
    try std.testing.expect(!exprJoinSourceStable(&script_name, &script, &outer_shadows));
    try std.testing.expect(!exprJoinSourceStable(&row_name, &script, &outer_no_shadow));
    try std.testing.expect(!exprJoinSourceStable(&nested, &script, &outer_no_shadow));
}
