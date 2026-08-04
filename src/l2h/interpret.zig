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
    InvalidProperty,
    UnknownHash,
    InvalidHashDigest,
    InvalidRecordField,
    DuplicateField,
    UnknownMethod,
    InvalidMethodArity,
    InvalidMethodReceiver,
    InvalidMethodFields,
    UnsupportedNode,
    InvalidAst,
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
    diag.noteSpan(e.span);
    return err;
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
    modes.str.hashFromString(bytes, def, digest[0..def.hash_length], ctx.allocator) catch return error.IoFailure;
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
pub fn evalProp(ctx: Ctx, recv: Value, prop: []const u8, sp: expr.Span) Error!Value {
    if (recv == .record) {
        return recv.record.get(prop) orelse failSpan(sp, error.UnknownProperty);
    }
    const kind = recv.sourceKind() orelse return failSpan(sp, error.UnknownProperty);
    const access = props.lookup(kind, prop) orelse return failSpan(sp, error.UnknownProperty);
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
                // Restore: side-effect to out (legacy calculateHash), value is the digest.
                const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = prop };
                var hctx: modes.HashCtx = .{ .builtin = &bctx, .hash = digest };
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
            return evalProp(ctx, recv, p.prop, e.span);
        },
        .method => |m| {
            const kind = method.lookup(m.name) orelse return failExpr(e, error.UnknownMethod);
            if (!method.arityOk(kind, m.args.len)) return failExpr(e, error.InvalidMethodArity);

            switch (kind) {
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
                    const f = if (kind == .file_offset)
                        recv.file.withOffset(arg_v.int)
                    else
                        recv.file.withLimit(arg_v.int);
                    return .{ .file = f };
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
            var out_fields = try ctx.allocator.alloc(value.RecordField, fields.len);
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

pub fn sinkPrint(ctx: Ctx, v: Value) Error!void {
    switch (v) {
        .string, .int, .bool => {
            try v.writeScalar(ctx.out);
            try ctx.out.writeAll("\n");
        },
        .record => |rec| {
            // Exactly one line per field (§7): no expanding nested Seq/Record.
            for (rec.fields) |f| try sinkFieldLine(ctx, f.value);
            return; // children already flushed
        },
        .file => |f| {
            try ctx.out.writeAll(f.path);
            try ctx.out.writeAll("\n");
        },
        .dir => |d| {
            try ctx.out.writeAll(d.path);
            try ctx.out.writeAll("\n");
        },
        .hash => |path| {
            try ctx.out.writeAll(path);
            try ctx.out.writeAll("\n");
        },
        .seq => |s| {
            for (s.items) |item| try sinkPrint(ctx, item);
            return; // children already flushed
        },
    }
    // Progressive output: main() uses a large stdout buffer; flush each sunk line.
    ctx.out.flush() catch return error.WriteFailed;
}

/// One sink line for a Record field value (scalars / path-like only).
fn sinkFieldLine(ctx: Ctx, v: Value) Error!void {
    switch (v) {
        .string, .int, .bool => {
            try v.writeScalar(ctx.out);
            try ctx.out.writeAll("\n");
        },
        .file => |f| {
            try ctx.out.writeAll(f.path);
            try ctx.out.writeAll("\n");
        },
        .dir => |d| {
            try ctx.out.writeAll(d.path);
            try ctx.out.writeAll("\n");
        },
        .hash => |path| {
            try ctx.out.writeAll(path);
            try ctx.out.writeAll("\n");
        },
        .record, .seq => return error.TypeMismatch,
    }
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
                    if (entry.kind != .file) continue;
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
                    if (entry.kind == .directory) {
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
                    if (entry.kind != .file) continue;
                    return try std.fs.path.join(path_allocator, &.{ self.dir.path, entry.path });
                }
            },
        }
    }
};

fn expectItem(kind: plan.SourceKind, item: Value) Error!Value {
    const got = item.sourceKind() orelse return error.TypeMismatch;
    if (got != kind) return error.TypeMismatch;
    return item;
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
            _ = expectItem(kind, item) catch |err| return failExpr(source, err);
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

// --- streaming pipeline -----------------------------------------------------

const StreamMode = union(enum) {
    sink,
    collect: *std.ArrayListUnmanaged(Value),
    to_barrier: struct {
        rows: *std.ArrayListUnmanaged(Env),
        barrier: *?*const plan.Clause,
    },
};

fn execStream(
    ctx: Ctx,
    clause: *const plan.Clause,
    env: *Env,
    depth: u32,
    mode: StreamMode,
    row_arena: *std.heap.ArenaAllocator,
    parent: std.mem.Allocator,
) Error!void {
    switch (clause.*) {
        .where => |w| {
            const pred = try evalExpr(ctx, w.pred, env, depth);
            if (!(try asBool(w.pred, pred))) return;
            return execStream(ctx, w.then, env, depth, mode, row_arena, parent);
        },
        .let => |l| {
            const v = try evalExpr(ctx, l.expr, env, depth);
            try env.put(ctx.allocator, l.name, v);
            return execStream(ctx, l.then, env, depth, mode, row_arena, parent);
        },
        .from => |f| {
            return streamExpand(ctx, f, env, depth, mode, row_arena, parent);
        },
        .join => |j| {
            return streamJoin(ctx, j, env, depth, mode, row_arena, parent);
        },
        .order_by, .group_by => {
            switch (mode) {
                .to_barrier => |tb| {
                    tb.barrier.* = clause;
                    try tb.rows.append(parent, try env.dupe(parent));
                },
                .sink, .collect => unreachable, // routed via streamRows / execBarrier
            }
        },
        .select => |sel| {
            if (sel.into) |into| {
                const v = try evalExpr(ctx, sel.expr, env, depth);
                var cont: Env = .{};
                try cont.put(ctx.allocator, into.name, v);
                return execStream(ctx, into.body, &cont, depth, mode, row_arena, parent);
            }
            const v = try evalExpr(ctx, sel.expr, env, depth);
            switch (mode) {
                .sink => try sinkSelect(ctx, sel.expr, env, v),
                .collect => |out| try out.append(parent, try v.dupe(parent)),
                .to_barrier => unreachable,
            }
        },
    }
}

fn streamExpand(
    ctx: Ctx,
    from: *const plan.From,
    outer: *Env,
    depth: u32,
    mode: StreamMode,
    row_arena: *std.heap.ArenaAllocator,
    parent: std.mem.Allocator,
) Error!void {
    switch (try bindSource(ctx, from.kind, from.source, outer, depth)) {
        .dir_files => |dir| {
            // Freeze outer into parent so per-file row_arena.reset cannot invalidate bindings.
            const stable_outer = try outer.dupe(parent);
            var iter = try DirFileIter.init(parent, ctx.io, dir);
            defer iter.deinit();
            while (true) {
                _ = row_arena.reset(.retain_capacity);
                const ralloc = row_arena.allocator();
                const path = (try iter.next(ralloc)) orelse break;
                var env = try stable_outer.clone(ralloc);
                try env.put(ralloc, from.range, .{ .file = .{ .path = path } });
                const row_ctx: Ctx = .{ .allocator = ralloc, .io = ctx.io, .out = ctx.out };
                try execStream(row_ctx, from.then, &env, depth, mode, row_arena, parent);
            }
        },
        .seq => |items| {
            const stable_outer = try outer.dupe(parent);
            for (items) |item| {
                _ = row_arena.reset(.retain_capacity);
                const ralloc = row_arena.allocator();
                var env = try stable_outer.clone(ralloc);
                try env.put(ralloc, from.range, try item.dupe(ralloc));
                const row_ctx: Ctx = .{ .allocator = ralloc, .io = ctx.io, .out = ctx.out };
                try execStream(row_ctx, from.then, &env, depth, mode, row_arena, parent);
            }
        },
        .one => |bound| {
            // Outer bindings (e.g. `dir d`) must outlive per-file row-arena resets in nested from.
            var env = try outer.dupe(parent);
            try env.put(parent, from.range, try bound.dupe(parent));
            try execStream(ctx, from.then, &env, depth, mode, row_arena, parent);
        },
    }
}

fn streamJoin(
    ctx: Ctx,
    j: *const plan.Join,
    outer: *Env,
    depth: u32,
    mode: StreamMode,
    row_arena: *std.heap.ArenaAllocator,
    parent: std.mem.Allocator,
) Error!void {
    // Materialize join inners (often small / seq); Dir sources still avoid a second full Env batch.
    const inners = try expandSourceValues(ctx, j.kind, j.source, outer, depth);
    defer ctx.allocator.free(inners);

    if (j.group_into) |gname| {
        var matches: std.ArrayListUnmanaged(Value) = .empty;
        defer matches.deinit(ctx.allocator);
        for (inners) |inner_val| {
            var inner_env = try outer.clone(ctx.allocator);
            try inner_env.put(ctx.allocator, j.range, inner_val);
            const ok = try keysEqual(ctx, j.outer_key, j.inner_key, outer, &inner_env, depth);
            if (ok) try matches.append(ctx.allocator, inner_val);
        }
        const seq = try ctx.allocator.create(value.Seq);
        seq.* = .{ .items = try ctx.allocator.dupe(Value, matches.items) };
        try outer.put(ctx.allocator, gname, .{ .seq = seq });
        return execStream(ctx, j.then, outer, depth, mode, row_arena, parent);
    }
    for (inners) |inner_val| {
        var inner_env = try outer.clone(ctx.allocator);
        try inner_env.put(ctx.allocator, j.range, inner_val);
        const ok = try keysEqual(ctx, j.outer_key, j.inner_key, outer, &inner_env, depth);
        if (!ok) continue;
        try execStream(ctx, j.then, &inner_env, depth, mode, row_arena, parent);
    }
}

/// Push already-materialized rows through `clause` (stream, or collect to the next barrier).
fn streamRows(
    ctx: Ctx,
    clause: *const plan.Clause,
    rows: []Env,
    depth: u32,
    mode: StreamMode,
    row_arena: *std.heap.ArenaAllocator,
    parent: std.mem.Allocator,
) Error!void {
    if (clause.hasBarrier()) {
        var next_rows: std.ArrayListUnmanaged(Env) = .empty;
        defer next_rows.deinit(parent);
        var barrier: ?*const plan.Clause = null;
        const tb: StreamMode = .{ .to_barrier = .{ .rows = &next_rows, .barrier = &barrier } };
        for (rows) |*row| {
            try execStream(ctx, clause, row, depth, tb, row_arena, parent);
        }
        if (barrier) |b| try execBarrier(ctx, b, next_rows.items, depth, mode, row_arena, parent);
        return;
    }
    for (rows) |*row| {
        try execStream(ctx, clause, row, depth, mode, row_arena, parent);
    }
}

/// `clause` is an `order_by` or `group_by` reached via to_barrier collection.
fn execBarrier(
    ctx: Ctx,
    clause: *const plan.Clause,
    rows: []Env,
    depth: u32,
    mode: StreamMode,
    row_arena: *std.heap.ArenaAllocator,
    parent: std.mem.Allocator,
) Error!void {
    switch (clause.*) {
        .order_by => |o| {
            const sorted = try orderRows(ctx, rows, o.keys, depth);
            defer ctx.allocator.free(sorted);
            try streamRows(ctx, o.then, sorted, depth, mode, row_arena, parent);
        },
        .group_by => |g| {
            const groups = try buildGroups(ctx, rows, g.proj, g.key, depth);
            if (g.into) |into| {
                defer ctx.allocator.free(groups);
                var cont: std.ArrayListUnmanaged(Env) = .empty;
                defer cont.deinit(ctx.allocator);
                for (groups) |gv| {
                    var env: Env = .{};
                    try env.put(ctx.allocator, into.name, gv);
                    try cont.append(ctx.allocator, env);
                }
                try streamRows(ctx, into.body, cont.items, depth, mode, row_arena, parent);
                return;
            }
            switch (mode) {
                .sink => {
                    defer ctx.allocator.free(groups);
                    for (groups) |gv| try sinkPrint(ctx, gv);
                },
                .collect => |out| {
                    defer ctx.allocator.free(groups);
                    for (groups) |gv| try out.append(parent, gv);
                },
                .to_barrier => unreachable,
            }
        },
        else => unreachable,
    }
}

fn evalQueryValues(ctx: Ctx, query: *const plan.From, outer: *Env, depth: u32) Error![]Value {
    var out: std.ArrayListUnmanaged(Value) = .empty;
    errdefer out.deinit(ctx.allocator);
    try runPipeline(ctx, query, outer, depth, .{ .collect = &out });
    return try out.toOwnedSlice(ctx.allocator);
}

/// Shared entry for sink (`run`) and nested collect (`evalQueryValues`).
fn runPipeline(
    ctx: Ctx,
    root: *const plan.From,
    outer: *Env,
    depth: u32,
    mode: StreamMode,
) Error!void {
    var row_arena = std.heap.ArenaAllocator.init(ctx.allocator);
    defer row_arena.deinit();
    const parent = ctx.allocator;

    if (root.then.hasBarrier()) {
        var rows: std.ArrayListUnmanaged(Env) = .empty;
        defer rows.deinit(parent);
        var barrier: ?*const plan.Clause = null;
        const tb: StreamMode = .{ .to_barrier = .{ .rows = &rows, .barrier = &barrier } };
        try streamExpand(ctx, root, outer, depth, tb, &row_arena, parent);
        if (barrier) |b| try execBarrier(ctx, b, rows.items, depth, mode, &row_arena, parent);
        return;
    }
    try streamExpand(ctx, root, outer, depth, mode, &row_arena, parent);
}

fn orderRows(ctx: Ctx, rows: []Env, order_keys: []plan.OrderKey, depth: u32) Error![]Env {
    const Indexed = struct {
        env: Env,
        keys: []Value,
        index: usize,
    };
    var indexed = try ctx.allocator.alloc(Indexed, rows.len);
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

fn keysEqual(
    ctx: Ctx,
    outer_key: *const Expr,
    inner_key: *const Expr,
    outer: *Env,
    inner: *Env,
    depth: u32,
) Error!bool {
    const l = try unwrapScalar(outer_key, try evalExpr(ctx, outer_key, outer, depth), false);
    const r = try unwrapScalar(inner_key, try evalExpr(ctx, inner_key, inner, depth), false);
    return l.eql(r) catch |err| return failExpr(outer_key, err);
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

/// Execute a query plan starting from an empty environment.
/// Plan/AST should live in `ctx.allocator`; per-file paths use a child row arena.
pub fn run(ctx: Ctx, query: *const plan.From) Error!void {
    var empty: Env = .{};
    try runPipeline(ctx, query, &empty, 0, .sink);
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
    var size_e: Expr = .{ .kind = .{ .prop = .{ .recv = &recv, .prop = "size" } } };
    // Act
    const size_v = try evalExpr(ctx, &size_e, &env, 0);
    // Assert
    try std.testing.expectEqual(@as(i64, 3), size_v.int);

    var md5_e: Expr = .{ .kind = .{ .prop = .{ .recv = &recv, .prop = "md5" } } };
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
    var call: Expr = .{ .kind = .{ .method = .{ .recv = &name_f, .name = "offset", .args = &args } } };

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
    var call: Expr = .{ .kind = .{ .method = .{ .recv = &name_d, .name = "tree", .args = &args } } };

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

    var row_arena = std.heap.ArenaAllocator.init(a);
    defer row_arena.deinit();

    // Act / Assert
    try std.testing.expectError(
        error.TypeMismatch,
        streamExpand(ctx, from, &env, 0, .sink, &row_arena, a),
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
