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
    UnsupportedMethodCall,
    UnsupportedNode,
    InvalidAst,
    IoFailure,
    WriteFailed,
    Overflow,
    QueryTooDeep,
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

/// Map errors from `modes.hashRun` / `builtinRun` without collapsing digest
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

fn hashHexOfFile(ctx: Ctx, algo: []const u8, path: []const u8) Error![]const u8 {
    const def = hashes.getHash(algo) orelse return error.UnknownHash;
    const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = algo };
    var fctx: modes.FileCtx = .{ .builtin = &bctx, .file_path = path };
    const result = modes.file.calculateFile(path, &fctx, runEnv(ctx), def) catch return error.IoFailure;
    if (result.open_error != null or result.info_error != null or result.hash_error != null)
        return error.IoFailure;
    var hex_buf: [modes.types.MAX_DIGEST_SIZE * 2]u8 = undefined;
    const hex = modes.types.hashToHex(result.digest[0..result.digest_len], true, &hex_buf);
    return try ctx.allocator.dupe(u8, hex);
}

fn fileSize(ctx: Ctx, path: []const u8) Error!i64 {
    var file = std.Io.Dir.cwd().openFile(ctx.io, path, .{}) catch return error.IoFailure;
    defer file.close(ctx.io);
    const st = file.stat(ctx.io) catch return error.IoFailure;
    // usize (u64 on 64-bit) -> i64: a >2^63-byte file would overflow. Surface a
    // clean error instead of trapping (Debug/ReleaseSafe) or UB (ReleaseFast).
    return std.math.cast(i64, st.size) orelse return error.Overflow;
}

/// Demand-driven property access (semantics §4).
pub fn evalProp(ctx: Ctx, recv: Value, prop: []const u8, sp: expr.Span) Error!Value {
    if (recv == .record) {
        return recv.record.get(prop) orelse failSpan(sp, error.UnknownProperty);
    }
    const kind = props.ofValue(recv) orelse return failSpan(sp, error.UnknownProperty);
    const access = props.lookup(kind, prop) orelse return failSpan(sp, error.UnknownProperty);
    return switch (access) {
        .path => switch (recv) {
            .file, .dir => |path| Value.plainStr(path),
            else => unreachable,
        },
        .size => switch (recv) {
            .file => |path| .{ .int = fileSize(ctx, path) catch |err| return failSpan(sp, err) },
            .string => |s| .{ .int = std.math.cast(i64, s.bytes.len) orelse return failSpan(sp, error.Overflow) },
            else => unreachable,
        },
        .hash_algo => switch (recv) {
            .file => |path| Value.digestStr(hashHexOfFile(ctx, prop, path) catch |err| return failSpan(sp, err)),
            .string => |s| Value.digestStr(hashHexOfBytes(ctx, prop, s.bytes) catch |err| return failSpan(sp, err)),
            .hash => |digest| blk: {
                // Restore: side-effect to out (legacy calculateHash), value is the digest.
                const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = prop };
                var hctx: modes.HashCtx = .{ .builtin = &bctx, .hash = digest };
                modes.builtinRun(modes.HashCtx, &bctx, &hctx, modes.hashRun, runEnv(ctx)) catch |err| {
                    return failSpan(sp, mapHashRestoreError(err));
                };
                break :blk Value.digestStr(digest);
            },
            else => unreachable,
        },
    };
}

// --- expression evaluation --------------------------------------------------

fn valuesEqual(a: Value, b: Value) Error!bool {
    return switch (a) {
        .int => |x| b == .int and x == b.int,
        .bool => |x| b == .bool and x == b.bool,
        .string => |x| blk: {
            if (b != .string) return error.TypeMismatch;
            // §5: case-insensitive when either side is a hash-property digest.
            if (x.is_digest or b.string.is_digest)
                break :blk std.ascii.eqlIgnoreCase(x.bytes, b.string.bytes);
            break :blk std.mem.eql(u8, x.bytes, b.string.bytes);
        },
        else => error.TypeMismatch,
    };
}

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

fn unwrapForCompare(e: *const Expr, v: Value) Error!Value {
    if (e.kind == .nested_query) {
        if (v != .seq) return v;
        if (v.seq.items.len != 1) return failExpr(e, error.TypeMismatch);
        return v.seq.items[0];
    }
    if (v == .seq) return failExpr(e, error.TypeMismatch);
    return v;
}

fn asBool(e: *const Expr, v: Value) Error!bool {
    return switch (v) {
        .bool => |b| b,
        .seq => |s| s.items.len > 0,
        else => failExpr(e, error.TypeMismatch),
    };
}

pub fn evalExpr(ctx: Ctx, e: *const Expr, env: *const Env, depth: u32) Error!Value {
    return switch (e.kind) {
        .string_lit => |s| Value.plainStr(s),
        .int_lit => |n| .{ .int = n },
        .nested_query => |q| {
            // Nested plan was compiled during typecheck; bound the runtime stack
            // in step with the analysis-time limit.
            if (depth >= compile.MAX_QUERY_DEPTH) return failExpr(e, error.QueryTooDeep);
            const items = try evalQueryValues(ctx, q, env, depth + 1);
            const seq = try ctx.allocator.create(value.Seq);
            seq.* = .{ .items = items };
            return .{ .seq = seq };
        },
        .query_ast => failExpr(e, error.InvalidAst),
        .name => |n| env.get(n) orelse failExpr(e, error.UndefinedName),
        .prop => |p| {
            const recv = try evalExpr(ctx, p.recv, env, depth);
            return evalProp(ctx, recv, p.prop, e.span);
        },
        .unary => |u| switch (u.op) {
            .not_ => {
                const v = try evalExpr(ctx, u.arg, env, depth);
                return .{ .bool = !(try asBool(u.arg, v)) };
            },
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
                    const l = try unwrapForCompare(b.left, try evalExpr(ctx, b.left, env, depth));
                    const r = try unwrapForCompare(b.right, try evalExpr(ctx, b.right, env, depth));
                    if (l != .string or r != .string) return failExpr(e, error.TypeMismatch);
                    const matched = re_match.matchRe(r.string.bytes, l.string.bytes);
                    return .{ .bool = if (b.op == .match) matched else !matched };
                },
                .eq, .neq => {
                    const l = try unwrapForCompare(b.left, try evalExpr(ctx, b.left, env, depth));
                    const r = try unwrapForCompare(b.right, try evalExpr(ctx, b.right, env, depth));
                    const eq = valuesEqual(l, r) catch |err| return failExpr(e, err);
                    return .{ .bool = if (b.op == .eq) eq else !eq };
                },
                .gt, .ge, .lt, .le => {
                    const l = try unwrapForCompare(b.left, try evalExpr(ctx, b.left, env, depth));
                    const r = try unwrapForCompare(b.right, try evalExpr(ctx, b.right, env, depth));
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
                out_fields[i] = .{ .name = f.name, .value = try evalExpr(ctx, f.expr, env, depth), };
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
        .string => |s| {
            try ctx.out.writeAll(s.bytes);
            try ctx.out.writeAll("\n");
        },
        .int => |n| {
            try ctx.out.print("{d}\n", .{n});
        },
        .bool => |b| {
            try ctx.out.print("{s}\n", .{if (b) "true" else "false"});
        },
        .record => |rec| {
            for (rec.fields) |f| try sinkPrint(ctx, f.value);
        },
        .file, .dir, .hash => |path| {
            try ctx.out.writeAll(path);
            try ctx.out.writeAll("\n");
        },
        .seq => |s| {
            for (s.items) |item| try sinkPrint(ctx, item);
        },
    }
}

// --- sources ----------------------------------------------------------------

fn openAs(ctx: Ctx, kind: plan.SourceKind, path_or_payload: []const u8) Error!Value {
    switch (kind) {
        .string => return Value.plainStr(path_or_payload),
        .hash => return .{ .hash = path_or_payload },
        .file => {
            var f = std.Io.Dir.cwd().openFile(ctx.io, path_or_payload, .{}) catch return error.IoFailure;
            f.close(ctx.io);
            return .{ .file = path_or_payload };
        },
        .dir => {
            var d = std.Io.Dir.cwd().openDir(ctx.io, path_or_payload, .{}) catch return error.IoFailure;
            d.close(ctx.io);
            return .{ .dir = path_or_payload };
        },
    }
}

fn listFilesInDir(ctx: Ctx, dir_path: []const u8) Error![]Value {
    var root = std.Io.Dir.cwd().openDir(ctx.io, dir_path, .{ .iterate = true }) catch return error.IoFailure;
    defer root.close(ctx.io);

    var list: std.ArrayListUnmanaged(Value) = .empty;
    errdefer list.deinit(ctx.allocator);

    var names: std.ArrayListUnmanaged([]const u8) = .empty;
    defer names.deinit(ctx.allocator);

    var it = root.iterate();
    while (true) {
        const maybe = it.next(ctx.io) catch return error.IoFailure;
        const entry = maybe orelse break;
        // Skip symlinks and non-files (directories, etc.).
        if (entry.kind != .file) continue;
        const name = try ctx.allocator.dupe(u8, entry.name);
        try names.append(ctx.allocator, name);
    }

    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.less);

    for (names.items) |name| {
        const full = try std.fs.path.join(ctx.allocator, &.{ dir_path, name });
        try list.append(ctx.allocator, .{ .file = full });
    }
    return try list.toOwnedSlice(ctx.allocator);
}

fn expectItem(kind: plan.SourceKind, item: Value) Error!Value {
    return switch (kind) {
        .string => switch (item) {
            .string => item,
            else => error.TypeMismatch,
        },
        .file => switch (item) {
            .file => item,
            else => error.TypeMismatch,
        },
        .dir => switch (item) {
            .dir => item,
            else => error.TypeMismatch,
        },
        .hash => switch (item) {
            .hash => item,
            else => error.TypeMismatch,
        },
    };
}

fn expandFrom(
    ctx: Ctx,
    from: *const plan.From,
    outer: *const Env,
    depth: u32,
) Error![]Env {
    const values = try expandSourceValues(ctx, from.kind, from.source, outer, depth);
    defer ctx.allocator.free(values);

    var out: std.ArrayListUnmanaged(Env) = .empty;
    errdefer out.deinit(ctx.allocator);
    for (values) |v| {
        var env = try outer.clone(ctx.allocator);
        try env.put(ctx.allocator, from.range, v);
        try out.append(ctx.allocator, env);
    }
    return try out.toOwnedSlice(ctx.allocator);
}

// --- clause interpreter -----------------------------------------------------

const ClauseMode = enum { sink, collect };

fn execClause(
    ctx: Ctx,
    clause: *const plan.Clause,
    rows: []Env,
    depth: u32,
    comptime mode: ClauseMode,
) if (mode == .collect) Error![]Value else Error!void {
    switch (clause.*) {
        .where => |w| {
            var filtered: std.ArrayListUnmanaged(Env) = .empty;
            defer filtered.deinit(ctx.allocator);
            for (rows) |row| {
                const pred = try evalExpr(ctx, w.pred, &row, depth);
                if (!(try asBool(w.pred, pred))) continue;
                try filtered.append(ctx.allocator, row);
            }
            return execClause(ctx, w.then, filtered.items, depth, mode);
        },
        .from => |f| {
            var next: std.ArrayListUnmanaged(Env) = .empty;
            defer next.deinit(ctx.allocator);
            for (rows) |row| {
                const expanded = try expandFrom(ctx, f, &row, depth);
                defer ctx.allocator.free(expanded);
                try next.appendSlice(ctx.allocator, expanded);
            }
            return execClause(ctx, f.then, next.items, depth, mode);
        },
        .let => |l| {
            var next: std.ArrayListUnmanaged(Env) = .empty;
            defer next.deinit(ctx.allocator);
            for (rows) |row| {
                const v = try evalExpr(ctx, l.expr, &row, depth);
                var env = try row.clone(ctx.allocator);
                try env.put(ctx.allocator, l.name, v);
                try next.append(ctx.allocator, env);
            }
            return execClause(ctx, l.then, next.items, depth, mode);
        },
        .join => |j| {
            var next: std.ArrayListUnmanaged(Env) = .empty;
            defer next.deinit(ctx.allocator);
            for (rows) |outer| {
                const inners = try expandSourceValues(ctx, j.kind, j.source, &outer, depth);
                defer ctx.allocator.free(inners);

                if (j.group_into) |gname| {
                    var matches: std.ArrayListUnmanaged(Value) = .empty;
                    defer matches.deinit(ctx.allocator);
                    for (inners) |inner_val| {
                        var inner_env = try outer.clone(ctx.allocator);
                        try inner_env.put(ctx.allocator, j.range, inner_val);
                        const ok = try keysEqual(ctx, j.outer_key, j.inner_key, &outer, &inner_env, depth);
                        if (ok) try matches.append(ctx.allocator, inner_val);
                    }
                    const seq = try ctx.allocator.create(value.Seq);
                    seq.* = .{ .items = try ctx.allocator.dupe(Value, matches.items) };
                    var env = try outer.clone(ctx.allocator);
                    try env.put(ctx.allocator, gname, .{ .seq = seq });
                    try next.append(ctx.allocator, env);
                } else {
                    for (inners) |inner_val| {
                        var inner_env = try outer.clone(ctx.allocator);
                        try inner_env.put(ctx.allocator, j.range, inner_val);
                        const ok = try keysEqual(ctx, j.outer_key, j.inner_key, &outer, &inner_env, depth);
                        if (!ok) continue;
                        try next.append(ctx.allocator, inner_env);
                    }
                }
            }
            return execClause(ctx, j.then, next.items, depth, mode);
        },
        .order_by => |o| {
            const sorted = try orderRows(ctx, rows, o.keys, depth);
            defer ctx.allocator.free(sorted);
            return execClause(ctx, o.then, sorted, depth, mode);
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
                return execClause(ctx, into.body, cont.items, depth, mode);
            }
            if (comptime mode == .collect) return groups;
            defer ctx.allocator.free(groups);
            for (groups) |gv| try sinkPrint(ctx, gv);
        },
        .select => |sel| {
            if (sel.into) |into| {
                var cont: std.ArrayListUnmanaged(Env) = .empty;
                defer cont.deinit(ctx.allocator);
                for (rows) |row| {
                    const v = try evalExpr(ctx, sel.expr, &row, depth);
                    var env: Env = .{};
                    try env.put(ctx.allocator, into.name, v);
                    try cont.append(ctx.allocator, env);
                }
                return execClause(ctx, into.body, cont.items, depth, mode);
            }
            if (comptime mode == .collect) {
                const out = try ctx.allocator.alloc(Value, rows.len);
                for (rows, 0..) |row, i| out[i] = try evalExpr(ctx, sel.expr, &row, depth);
                return out;
            }
            for (rows) |row| {
                const v = try evalExpr(ctx, sel.expr, &row, depth);
                try sinkSelect(ctx, sel.expr, &row, v);
            }
        },
    }
}

fn evalQueryValues(ctx: Ctx, query: *const plan.QueryPlan, outer: *const Env, depth: u32) Error![]Value {
    const rows = try expandFrom(ctx, query.root, outer, depth);
    defer ctx.allocator.free(rows);
    return execClause(ctx, query.root.then, rows, depth, .collect);
}

const RowKeys = struct {
    env: Env,
    keys: []Value,
};

fn orderRows(ctx: Ctx, rows: []Env, order_keys: []plan.OrderKey, depth: u32) Error![]Env {
    var decorated = try ctx.allocator.alloc(RowKeys, rows.len);
    defer {
        for (decorated) |*d| ctx.allocator.free(d.keys);
        ctx.allocator.free(decorated);
    }
    for (rows, 0..) |row, i| {
        const ks = try ctx.allocator.alloc(Value, order_keys.len);
        for (order_keys, 0..) |ok, j| {
            const raw = try evalExpr(ctx, ok.expr, &row, depth);
            ks[j] = try unwrapForCompare(ok.expr, raw);
        }
        decorated[i] = .{ .env = row, .keys = ks };
    }

    // Reject mixed/incomparable key kinds before sort (std.mem.sort cannot bubble errors).
    if (decorated.len > 1) {
        for (order_keys, 0..) |ok, col| {
            const baseline = decorated[0].keys[col];
            for (decorated[1..]) |d| {
                _ = compareValues(baseline, d.keys[col]) catch |err| return failExpr(ok.expr, err);
            }
        }
    }

    // Decorate with index for stable ordering.
    const Indexed = struct { row: RowKeys, index: usize };
    var indexed = try ctx.allocator.alloc(Indexed, decorated.len);
    defer ctx.allocator.free(indexed);
    for (decorated, 0..) |d, i| indexed[i] = .{ .row = d, .index = i };

    const StableCtx = struct {
        order_keys: []plan.OrderKey,
        fn less(self: @This(), a: Indexed, b: Indexed) bool {
            for (self.order_keys, 0..) |ok, i| {
                const cmp = compareValues(a.row.keys[i], b.row.keys[i]) catch
                    @panic("invariant violated: order keys must be pre-validated comparable; report as bug");
                if (cmp == 0) continue;
                if (ok.descending) return cmp > 0;
                return cmp < 0;
            }
            return a.index < b.index;
        }
    };
    std.mem.sort(Indexed, indexed, StableCtx{ .order_keys = order_keys }, StableCtx.less);

    const out = try ctx.allocator.alloc(Env, indexed.len);
    for (indexed, 0..) |ix, i| out[i] = ix.row.env;
    return out;
}

fn compareValues(a: Value, b: Value) Error!i8 {
    if (a == .int and b == .int) {
        if (a.int < b.int) return -1;
        if (a.int > b.int) return 1;
        return 0;
    }
    if (a == .string and b == .string) {
        const as = a.string;
        const bs = b.string;
        if (as.is_digest or bs.is_digest) {
            var i: usize = 0;
            while (i < as.bytes.len and i < bs.bytes.len) : (i += 1) {
                const ca = std.ascii.toLower(as.bytes[i]);
                const cb = std.ascii.toLower(bs.bytes[i]);
                if (ca < cb) return -1;
                if (ca > cb) return 1;
            }
            if (as.bytes.len < bs.bytes.len) return -1;
            if (as.bytes.len > bs.bytes.len) return 1;
            return 0;
        }
        const ord = std.mem.order(u8, as.bytes, bs.bytes);
        return switch (ord) {
            .lt => @as(i8, -1),
            .gt => @as(i8, 1),
            .eq => @as(i8, 0),
        };
    }
    if (a == .bool and b == .bool) {
        if (a.bool == b.bool) return 0;
        if (!a.bool and b.bool) return -1;
        return 1;
    }
    return error.TypeMismatch;
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

    for (rows) |row| {
        const k = try unwrapForCompare(key_expr, try evalExpr(ctx, key_expr, &row, depth));
        const p = try evalExpr(ctx, proj, &row, depth);
        var found: ?usize = null;
        for (buckets.items, 0..) |b, i| {
            const same = valuesEqual(b.key, k) catch |err| return failExpr(key_expr, err);
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
    outer: *const Env,
    inner: *const Env,
    depth: u32,
) Error!bool {
    const l = try unwrapForCompare(outer_key, try evalExpr(ctx, outer_key, outer, depth));
    const r = try unwrapForCompare(inner_key, try evalExpr(ctx, inner_key, inner, depth));
    return valuesEqual(l, r) catch |err| return failExpr(outer_key, err);
}

fn expandSourceValues(
    ctx: Ctx,
    kind: plan.SourceKind,
    source: plan.SourceExpr,
    env: *const Env,
    depth: u32,
) Error![]Value {
    switch (source) {
        .expr => |e| {
            const src_val = try evalExpr(ctx, e, env, depth);
            if (src_val == .seq) {
                const out = try ctx.allocator.alloc(Value, src_val.seq.items.len);
                for (src_val.seq.items, 0..) |item, i| {
                    out[i] = expectItem(kind, item) catch |err| return failExpr(e, err);
                }
                return out;
            }
            const payload = switch (src_val) {
                .string => |s| s.bytes,
                .file, .dir, .hash => |p| p,
                else => return failExpr(e, error.TypeMismatch),
            };
            const bound = openAs(ctx, kind, payload) catch |err| return failExpr(e, err);
            const slice = try ctx.allocator.alloc(Value, 1);
            slice[0] = bound;
            return slice;
        },
        .files_in_dir => |dir_name| {
            const dval = env.get(dir_name) orelse return error.UndefinedName;
            if (dval != .dir) return error.TypeMismatch;
            return listFilesInDir(ctx, dval.dir);
        },
    }
}

fn sinkSelect(ctx: Ctx, e: *const Expr, env: *const Env, v: Value) Error!void {
    // If projecting hash.algo, restore already printed via evalProp.
    if (e.kind == .prop and e.kind.prop.recv.kind == .name) {
        if (env.get(e.kind.prop.recv.kind.name)) |recv| {
            if (recv == .hash) return;
        }
    }
    try sinkPrint(ctx, v);
}

/// Execute a query plan starting from an empty environment.
/// Caller should use an arena allocator for `ctx.allocator` (envs are not deeply freed).
pub fn run(ctx: Ctx, query: *const plan.QueryPlan) Error!void {
    var empty: Env = .{};
    const rows = try expandFrom(ctx, query.root, &empty, 0);
    defer ctx.allocator.free(rows);
    try execClause(ctx, query.root.then, rows, 0, .sink);
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

test "from string where size select md5" {
    // Arrange
    // from string s in 'abc' where s.size > 0 select s.md5;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    const lit = try a.create(Expr);
    lit.* = .{ .kind = .{ .string_lit = "abc" } };
    const name_s = try a.create(Expr);
    name_s.* = .{ .kind = .{ .name = "s" } };
    const size_p = try a.create(Expr);
    size_p.* = .{ .kind = .{ .prop = .{ .recv = name_s, .prop = "size" } } };
    const zero = try a.create(Expr);
    zero.* = .{ .kind = .{ .int_lit = 0 } };
    const pred = try a.create(Expr);
    pred.* = .{ .kind = .{ .binary = .{ .op = .gt, .left = size_p, .right = zero } } };
    const md5_p = try a.create(Expr);
    md5_p.* = .{ .kind = .{ .prop = .{ .recv = name_s, .prop = "md5" } } };

    const select = try a.create(plan.Select);
    select.* = .{ .expr = md5_p };
    const where_clause = try a.create(plan.Clause);
    where_clause.* = .{ .where = .{ .pred = pred, .then = try a.create(plan.Clause) } };
    where_clause.where.then.* = .{ .select = select };

    const root = try a.create(plan.From);
    root.* = .{
        .kind = .string,
        .range = "s",
        .source = .{ .expr = lit },
        .then = where_clause,
    };
    const q = plan.QueryPlan{ .root = root };
    // Act
    try run(ctx, &q);

    const got = std.Io.Writer.buffered(&writer);
    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got);
}

test "where filters out by size" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    const lit = try a.create(Expr);
    lit.* = .{ .kind = .{ .string_lit = "" } };
    const name_s = try a.create(Expr);
    name_s.* = .{ .kind = .{ .name = "s" } };
    const size_p = try a.create(Expr);
    size_p.* = .{ .kind = .{ .prop = .{ .recv = name_s, .prop = "size" } } };
    const zero = try a.create(Expr);
    zero.* = .{ .kind = .{ .int_lit = 0 } };
    const pred = try a.create(Expr);
    pred.* = .{ .kind = .{ .binary = .{ .op = .gt, .left = size_p, .right = zero } } };
    const md5_p = try a.create(Expr);
    md5_p.* = .{ .kind = .{ .prop = .{ .recv = name_s, .prop = "md5" } } };

    const select = try a.create(plan.Select);
    select.* = .{ .expr = md5_p };
    const where_clause = try a.create(plan.Clause);
    where_clause.* = .{ .where = .{ .pred = pred, .then = try a.create(plan.Clause) } };
    where_clause.where.then.* = .{ .select = select };

    const root = try a.create(plan.From);
    root.* = .{
        .kind = .string,
        .range = "s",
        .source = .{ .expr = lit },
        .then = where_clause,
    };
    try run(ctx, &.{ .root = root });
    // Act
    try std.testing.expectEqualStrings("", std.Io.Writer.buffered(&writer));
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

test "let binds intermediate then select" {
    // Arrange
    // from string s in 'abc' let d = s.md5 select d;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    const lit = try a.create(Expr);
    lit.* = .{ .kind = .{ .string_lit = "abc" } };
    const name_s = try a.create(Expr);
    name_s.* = .{ .kind = .{ .name = "s" } };
    const md5_p = try a.create(Expr);
    md5_p.* = .{ .kind = .{ .prop = .{ .recv = name_s, .prop = "md5" } } };
    const name_d = try a.create(Expr);
    name_d.* = .{ .kind = .{ .name = "d" } };

    const select = try a.create(plan.Select);
    select.* = .{ .expr = name_d };
    const let_clause = try a.create(plan.Clause);
    let_clause.* = .{ .let = .{ .name = "d", .expr = md5_p, .then = try a.create(plan.Clause) } };
    let_clause.let.then.* = .{ .select = select };

    const root = try a.create(plan.From);
    root.* = .{ .kind = .string, .range = "s", .source = .{ .expr = lit }, .then = let_clause };
    // Act
    try run(ctx, &.{ .root = root });
    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", std.Io.Writer.buffered(&writer));
}

test "select into then select continuation" {
    // Arrange
    // from string s in 'abc' select s.md5 into h select h;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    const lit = try a.create(Expr);
    lit.* = .{ .kind = .{ .string_lit = "abc" } };
    const name_s = try a.create(Expr);
    name_s.* = .{ .kind = .{ .name = "s" } };
    const md5_p = try a.create(Expr);
    md5_p.* = .{ .kind = .{ .prop = .{ .recv = name_s, .prop = "md5" } } };
    const name_h = try a.create(Expr);
    name_h.* = .{ .kind = .{ .name = "h" } };

    const select2 = try a.create(plan.Select);
    select2.* = .{ .expr = name_h };
    const body = try a.create(plan.Clause);
    body.* = .{ .select = select2 };

    const select1 = try a.create(plan.Select);
    select1.* = .{ .expr = md5_p, .into = .{ .name = "h", .body = body } };

    const root_clause = try a.create(plan.Clause);
    root_clause.* = .{ .select = select1 };
    const root = try a.create(plan.From);
    root.* = .{ .kind = .string, .range = "s", .source = .{ .expr = lit }, .then = root_clause };
    // Act
    try run(ctx, &.{ .root = root });
    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", std.Io.Writer.buffered(&writer));
}

test "inner join on md5" {
    // Arrange
    // from string a in 'abc' join string b in 'abc' on a.md5 equals b.md5 select a.md5;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    const lit_a = try a.create(Expr);
    lit_a.* = .{ .kind = .{ .string_lit = "abc" } };
    const lit_b = try a.create(Expr);
    lit_b.* = .{ .kind = .{ .string_lit = "abc" } };
    const name_a = try a.create(Expr);
    name_a.* = .{ .kind = .{ .name = "a" } };
    const name_b = try a.create(Expr);
    name_b.* = .{ .kind = .{ .name = "b" } };
    const a_md5 = try a.create(Expr);
    a_md5.* = .{ .kind = .{ .prop = .{ .recv = name_a, .prop = "md5" } } };
    const b_md5 = try a.create(Expr);
    b_md5.* = .{ .kind = .{ .prop = .{ .recv = name_b, .prop = "md5" } } };

    const select = try a.create(plan.Select);
    select.* = .{ .expr = a_md5 };
    const sel_cl = try a.create(plan.Clause);
    sel_cl.* = .{ .select = select };

    const join = try a.create(plan.Join);
    join.* = .{
        .kind = .string,
        .range = "b",
        .source = .{ .expr = lit_b },
        .outer_key = a_md5,
        .inner_key = b_md5,
        .then = sel_cl,
    };
    const join_cl = try a.create(plan.Clause);
    join_cl.* = .{ .join = join };

    const root = try a.create(plan.From);
    root.* = .{ .kind = .string, .range = "a", .source = .{ .expr = lit_a }, .then = join_cl };
    // Act
    try run(ctx, &.{ .root = root });
    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", std.Io.Writer.buffered(&writer));
}

test "join into group then from seq select" {
    // Arrange
    // from string a in 'abc'
    // join string b in 'abc' on a.md5 equals b.md5 into g
    // from string x in g
    // select x.md5;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    const lit_a = try a.create(Expr);
    lit_a.* = .{ .kind = .{ .string_lit = "abc" } };
    const lit_b = try a.create(Expr);
    lit_b.* = .{ .kind = .{ .string_lit = "abc" } };
    const name_a = try a.create(Expr);
    name_a.* = .{ .kind = .{ .name = "a" } };
    const name_b = try a.create(Expr);
    name_b.* = .{ .kind = .{ .name = "b" } };
    const name_g = try a.create(Expr);
    name_g.* = .{ .kind = .{ .name = "g" } };
    const name_x = try a.create(Expr);
    name_x.* = .{ .kind = .{ .name = "x" } };
    const a_md5 = try a.create(Expr);
    a_md5.* = .{ .kind = .{ .prop = .{ .recv = name_a, .prop = "md5" } } };
    const b_md5 = try a.create(Expr);
    b_md5.* = .{ .kind = .{ .prop = .{ .recv = name_b, .prop = "md5" } } };
    const x_md5 = try a.create(Expr);
    x_md5.* = .{ .kind = .{ .prop = .{ .recv = name_x, .prop = "md5" } } };

    const select = try a.create(plan.Select);
    select.* = .{ .expr = x_md5 };
    const sel_cl = try a.create(plan.Clause);
    sel_cl.* = .{ .select = select };

    const from_x = try a.create(plan.From);
    from_x.* = .{
        .kind = .string,
        .range = "x",
        .source = .{ .expr = name_g },
        .then = sel_cl,
    };
    const from_cl = try a.create(plan.Clause);
    from_cl.* = .{ .from = from_x };

    const join = try a.create(plan.Join);
    join.* = .{
        .kind = .string,
        .range = "b",
        .source = .{ .expr = lit_b },
        .outer_key = a_md5,
        .inner_key = b_md5,
        .group_into = "g",
        .then = from_cl,
    };
    const join_cl = try a.create(plan.Clause);
    join_cl.* = .{ .join = join };

    const root = try a.create(plan.From);
    root.* = .{ .kind = .string, .range = "a", .source = .{ .expr = lit_a }, .then = join_cl };
    // Act
    try run(ctx, &.{ .root = root });
    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", std.Io.Writer.buffered(&writer));
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
    items[0] = .{ .file = "a.txt" };
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
        .source = .{ .expr = &name_xs },
        .then = select_clause,
    };

    // Act / Assert
    try std.testing.expectError(error.TypeMismatch, expandFrom(ctx, from, &env, 0));
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
    try env1.put(a, "k", .{ .file = "/a" });
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
