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
const builtins = @import("builtins.zig");
const re_match = @import("match_re.zig");

const Value = value.Value;
const Env = value.Env;
const Expr = expr.Expr;
const BinaryOp = expr.BinaryOp;

pub const Error = builtins.Error || error{
    UndefinedName,
    DuplicateField,
    QueryTooDeep,
    CallChild,
    /// Invalid regex pattern for `~` / `!~` (§5.3 / §8).
    BadRegex,
    /// PCRE2 match/depth limit hit: the result is unknown (§5.3).
    RegexLimit,
    /// PCRE2 match failed for another reason; never read as a non-match.
    RegexFailed,
};

pub const Ctx = builtins.Ctx;

fn failExpr(e: *const Expr, err: Error) Error {
    return diag.failSpan(e.span, err);
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
            return builtins.evalProp(ctx, recv, p.prop, p.access, e.span);
        },
        .method => |m| {
            if (!builtins.spec(m.kind).arityOk(m.args.len)) return failExpr(e, error.InvalidMethodArity);
            const recv = try evalExpr(ctx, m.recv, env, depth);
            const args = try ctx.allocator.alloc(Value, m.args.len);
            for (m.args, 0..) |arg, i| {
                args[i] = try unwrapScalar(arg, try evalExpr(ctx, arg, env, depth), true);
            }
            return builtins.evalMethod(ctx, m.kind, m.name, recv, args, e.span);
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
                    const matched = re_match.matchRe(r.string.bytes, l.string.bytes) catch |err| switch (err) {
                        error.BadRegex => return failExpr(e, error.BadRegex),
                        error.RegexLimit => return failExpr(e, error.RegexLimit),
                        error.RegexFailed => return failExpr(e, error.RegexFailed),
                        error.OutOfMemory => return error.OutOfMemory,
                    };
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
        .hash => |h| try ctx.out.writeAll(h.digest),
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
        .hash => return Value.hashDigest(path_or_payload),
        .file => {
            // §3.3: regular file only — openFile succeeds on directories on Linux.
            const entry_kind = modes.file.pathKind(ctx.io, path_or_payload) catch return diag.ioFail(path_or_payload);
            if (entry_kind != .file) return diag.ioFail(path_or_payload);
            return .{ .file = .{ .path = path_or_payload } };
        },
        .dir => {
            if (!modes.dir.dirExists(ctx.io, path_or_payload)) return diag.ioFail(path_or_payload);
            return .{ .dir = .{ .path = path_or_payload } };
        },
    }
}

// --- directory file iterator (§3.4 / §4.6) ----------------------------------

/// Yields regular-file paths under a Dir one at a time (walk order, no sort).
const DirFileIter = struct {
    skip_errors: bool,
    /// `null` when the root could not be opened for listing under
    /// `skipErrors()`: the walk yields no files.
    walk: ?modes.dir.FileWalk,

    fn init(gpa: std.mem.Allocator, io: std.Io, dir: value.DirVal) Error!DirFileIter {
        // `from dir` already rejected a missing path or a non-directory; a root
        // that exists but cannot be listed is a walk failure like any other.
        const walk = modes.dir.FileWalk.init(gpa, io, dir.path, dir.max_depth) catch |err| switch (err) {
            error.OpenFailed => if (dir.skip_errors) null else return diag.ioFail(dir.path),
            error.OutOfMemory => return error.OutOfMemory,
        };
        return .{ .skip_errors = dir.skip_errors, .walk = walk };
    }

    fn deinit(self: *DirFileIter) void {
        if (self.walk) |*w| w.deinit();
    }

    /// Owned path allocated with `path_gpa`, or `null` when exhausted.
    fn next(self: *DirFileIter, path_gpa: std.mem.Allocator) Error!?[]const u8 {
        const walk = if (self.walk) |*w| w else return null;
        while (true) {
            const step = (try walk.next(path_gpa)) orelse return null;
            switch (step) {
                .file => |p| return p,
                .failed => |fail| {
                    defer path_gpa.free(fail.path);
                    if (self.skip_errors) continue;
                    return diag.ioFail(fail.path);
                },
            }
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
    var list: std.ArrayList(Value) = .empty;
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
    collect: *std.ArrayList(Value),
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
    /// Allocator for the current row (the streaming `FromOp`'s row arena, else parent).
    row_alloc: std.mem.Allocator,
    /// Script env for terminal `into id;` binds (same pointer as drive outer).
    script: *Env,
    /// Allocator for values stored in `script` (survives query arenas).
    script_alloc: std.mem.Allocator,
    env_result: union(enum) { none, value: ?*Env } = .none,
    env_calls: std.ArrayList(*Op) = .empty,

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
    /// Per-outer-row state: the bound source, `stable_outer`, Seq items and the
    /// singleton row. Reset when the next outer row is pulled, so a nested
    /// `from` over a streaming source does not grow with the row count.
    outer_arena: ?std.heap.ArenaAllocator = null,
    /// Rows streamed from a Dir walk or Seq. Owned per operator: a reset here
    /// must not free rows of an upstream `from` that a `join` still yields.
    row_arena: ?std.heap.ArenaAllocator = null,

    fn open(self: *FromOp, pc: *PipeCtx, outer: *Env) Error!void {
        self.phase = .need_outer;
        if (self.outer_arena == null) self.outer_arena = .init(pc.parent);
        if (self.row_arena == null) self.row_arena = .init(pc.parent);
        if (self.child == null) self.script_outer = outer;
    }

    fn nextEnv(self: *FromOp, pc: *PipeCtx) Error!?*Env {
        while (true) {
            switch (self.phase) {
                .need_outer => {
                    // The previous outer row is finished: downstream pulled again.
                    const oalloc = resetArena(&self.outer_arena.?);
                    const outer: *Env = if (self.child) |c|
                        (try nextEnvChild(c, pc)) orelse return null
                    else blk: {
                        const o = self.script_outer orelse return null;
                        self.script_outer = null;
                        break :blk o;
                    };
                    // Bind in the outer arena, then persist Dir/Seq payloads there: they
                    // must outlive this op's row arena reset in `.in_dir` / `.in_seq`. Env
                    // lookups (Dir via Seq/nested query/`into`) may still alias the
                    // upstream row arena, which the next pull recycles.
                    const bind_ctx: Ctx = .{ .allocator = oalloc, .io = pc.io, .out = pc.out };
                    const bound = try bindSource(bind_ctx, self.from.kind, self.from.source, outer, pc.depth);
                    switch (bound) {
                        .dir_files => |dir| {
                            self.stable_outer = try outer.dupe(oalloc);
                            const stable_dir: value.DirVal = .{
                                .path = try oalloc.dupe(u8, dir.path),
                                .max_depth = dir.max_depth,
                                .skip_errors = dir.skip_errors,
                            };
                            self.dir_iter = try DirFileIter.init(oalloc, pc.io, stable_dir);
                            self.phase = .in_dir;
                        },
                        .seq => |items| {
                            self.stable_outer = try outer.dupe(oalloc);
                            const owned = try oalloc.alloc(Value, items.len);
                            for (items, 0..) |item, i| {
                                owned[i] = try item.dupe(oalloc);
                            }
                            self.seq_items = owned;
                            self.seq_index = 0;
                            self.phase = .in_seq;
                        },
                        .one => |v| {
                            pc.row_alloc = oalloc;
                            self.row = try outer.dupe(oalloc);
                            try self.row.put(oalloc, self.from.range, try v.dupe(oalloc));
                            self.phase = .in_one;
                        },
                    }
                },
                .in_dir => {
                    const iter = &self.dir_iter.?;
                    const ralloc = resetArena(&self.row_arena.?);
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
                    const ralloc = resetArena(&self.row_arena.?);
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

    fn close(self: *FromOp, _: *PipeCtx) void {
        // The walker's stack lives in the outer arena: deinit it first.
        if (self.dir_iter) |*it| {
            it.deinit();
            self.dir_iter = null;
        }
        if (self.row_arena) |*arena| {
            arena.deinit();
            self.row_arena = null;
        }
        if (self.outer_arena) |*arena| {
            arena.deinit();
            self.outer_arena = null;
        }
    }
};

fn resetArena(arena: *std.heap.ArenaAllocator) std.mem.Allocator {
    _ = arena.reset(.retain_capacity);
    return arena.allocator();
}

const WhereOp = struct {
    pred: *const Expr,
    child: *Op,

    fn open(self: *WhereOp, _: *PipeCtx, _: *Env) Error!void {
        _ = self;
    }

    fn nextEnv(self: *WhereOp, pc: *PipeCtx) Error!?*Env {
        while (true) {
            const env = (try nextEnvChild(self.child, pc)) orelse return null;
            const pred = try evalExpr(pc.ctx(), self.pred, env, pc.depth);
            if (try asBool(self.pred, pred)) return env;
        }
    }

    fn close(self: *WhereOp, _: *PipeCtx) void {
        _ = self;
    }
};

const LetOp = struct {
    name: []const u8,
    expr: *const Expr,
    child: *Op,

    fn open(self: *LetOp, _: *PipeCtx, _: *Env) Error!void {
        _ = self;
    }

    fn nextEnv(self: *LetOp, pc: *PipeCtx) Error!?*Env {
        const env = (try nextEnvChild(self.child, pc)) orelse return null;
        const v = try evalExpr(pc.ctx(), self.expr, env, pc.depth);
        try env.put(pc.row_alloc, self.name, v);
        return env;
    }

    fn close(self: *LetOp, _: *PipeCtx) void {
        _ = self;
    }
};

const SelectIntoOp = struct {
    name: []const u8,
    expr: *const Expr,
    child: *Op,
    cont: Env = .{},

    fn open(self: *SelectIntoOp, _: *PipeCtx, _: *Env) Error!void {
        _ = self;
    }

    fn nextEnv(self: *SelectIntoOp, pc: *PipeCtx) Error!?*Env {
        const env = (try nextEnvChild(self.child, pc)) orelse return null;
        const v = try evalExpr(pc.ctx(), self.expr, env, pc.depth);
        self.cont = .{};
        try self.cont.put(pc.row_alloc, self.name, v);
        return &self.cont;
    }

    fn close(self: *SelectIntoOp, _: *PipeCtx) void {
        _ = self;
    }
};

/// True when `op` or its descendants bind `name` into the row env (range / let / into).
fn opBindsName(op: *const Op, name: []const u8) bool {
    var current: ?*const Op = op;
    while (current) |c| {
        switch (c.*) {
            .from => |*f| if (std.mem.eql(u8, f.from.range, name)) return true,
            .let => |*l| if (std.mem.eql(u8, l.name, name)) return true,
            .join => |*j| {
                if (std.mem.eql(u8, j.join.range, name)) return true;
                if (j.join.group_into) |g| {
                    if (std.mem.eql(u8, g, name)) return true;
                }
            },
            .group_into => |*g| if (std.mem.eql(u8, g.into_name, name)) return true,
            .select_into => |*s| if (std.mem.eql(u8, s.name, name)) return true,
            else => {},
        }
        current = opChild(c);
    }
    return false;
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
    /// Dir listings (and other expand scratch) live here so rematerialize can
    /// `reset` instead of abandoning path bytes in the query arena.
    inners_arena: std.heap.ArenaAllocator,
    inner_index: usize = 0,
    outer_env: ?*Env = null,
    /// Cached `normalize(outer_key)` for the current outer row.
    outer_key_val: Value = .{ .bool = false },
    /// Scratch / yield env: one clone of the outer, range binding overwritten per candidate.
    row: Env = .{},
    /// Group-join yield env: outer-arena clone of the outer row plus the `into`
    /// binding. The child row env's map lives in the child's row arena, which
    /// the next pull recycles, so it must not be mutated or yielded directly.
    group_env: Env = .{},
    /// Per-outer-row state: `row`, `group_env`, the outer key and the inner
    /// keys evaluated against it. Reset when the next outer row is pulled.
    outer_arena: ?std.heap.ArenaAllocator = null,

    fn open(self: *JoinOp, pc: *PipeCtx, _: *Env) Error!void {
        if (self.outer_arena == null) self.outer_arena = .init(pc.parent);
        self.clearInners();
        self.inner_index = 0;
        self.outer_env = null;
    }

    fn clearInners(self: *JoinOp) void {
        self.inners = &.{};
        self.inners_cached = false;
        _ = self.inners_arena.reset(.retain_capacity);
    }

    fn ensureInners(self: *JoinOp, pc: *PipeCtx, outer: *Env) Error!void {
        if (self.inners_cached) return;
        self.clearInners();
        const c: Ctx = .{ .allocator = self.inners_arena.allocator(), .io = pc.io, .out = pc.out };
        self.inners = try expandSourceValues(c, self.join.kind, self.join.source, outer, pc.depth);
        self.inners_cached = exprJoinSourceStable(self.join.source, pc.script, self.child);
    }

    fn prepareOuter(self: *JoinOp, pc: *PipeCtx, oalloc: std.mem.Allocator, outer: *Env) Error!void {
        try self.ensureInners(pc, outer);
        self.inner_index = 0;
        const c: Ctx = .{ .allocator = oalloc, .io = pc.io, .out = pc.out };
        const raw = try evalExpr(c, self.join.outer_key, outer, pc.depth);
        self.outer_key_val = try unwrapScalar(self.join.outer_key, raw, false);
        self.row = try outer.clone(oalloc);
    }

    fn nextEnv(self: *JoinOp, pc: *PipeCtx) Error!?*Env {
        const oalloc = self.outer_arena.?.allocator();
        while (true) {
            if (self.outer_env == null) {
                // The previous outer row is finished: downstream pulled again.
                _ = resetArena(&self.outer_arena.?);
                self.outer_env = (try nextEnvChild(self.child, pc)) orelse return null;
                try self.prepareOuter(pc, oalloc, self.outer_env.?);

                if (self.join.group_into) |gname| {
                    const c: Ctx = .{ .allocator = oalloc, .io = pc.io, .out = pc.out };
                    var matches: std.ArrayList(Value) = .empty;
                    for (self.inners) |inner_val| {
                        try self.row.put(oalloc, self.join.range, inner_val);
                        if (try keyEqualsCached(c, self.outer_key_val, self.join.outer_key, self.join.inner_key, &self.row, pc.depth)) {
                            try matches.append(oalloc, inner_val);
                        }
                    }
                    // Deep-dupe before clearInners: matches alias inners_arena paths.
                    for (matches.items) |*item| item.* = try item.dupe(oalloc);
                    const seq = try oalloc.create(value.Seq);
                    seq.* = .{ .items = matches.items };
                    self.group_env = try self.outer_env.?.clone(oalloc);
                    try self.group_env.put(oalloc, gname, .{ .seq = seq });
                    if (!self.inners_cached) self.clearInners();
                    pc.row_alloc = oalloc;
                    self.outer_env = null;
                    return &self.group_env;
                }
            }
            while (self.inner_index < self.inners.len) {
                const inner_val = self.inners[self.inner_index];
                self.inner_index += 1;
                pc.row_alloc = oalloc;
                try self.row.put(oalloc, self.join.range, inner_val);
                if (try keyEqualsCached(pc.ctx(), self.outer_key_val, self.join.outer_key, self.join.inner_key, &self.row, pc.depth)) {
                    return &self.row;
                }
            }
            if (!self.inners_cached) self.clearInners();
            self.outer_env = null;
        }
    }

    fn close(self: *JoinOp, _: *PipeCtx) void {
        self.inners = &.{};
        self.inners_cached = false;
        self.inners_arena.deinit();
        if (self.outer_arena) |*arena| {
            arena.deinit();
            self.outer_arena = null;
        }
        self.outer_env = null;
    }
};

const OrderByOp = struct {
    keys: []plan.OrderKey,
    child: *Op,
    rows: []Env = &.{},
    index: usize = 0,
    ready: bool = false,

    fn open(self: *OrderByOp, _: *PipeCtx, _: *Env) Error!void {
        self.rows = &.{};
        self.index = 0;
        self.ready = false;
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

    fn open(self: *GroupIntoOp, _: *PipeCtx, _: *Env) Error!void {
        self.rows = &.{};
        self.index = 0;
        self.ready = false;
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
    }
};

const ProjectOp = struct {
    expr: *const Expr,
    child: *Op,

    fn open(self: *ProjectOp, _: *PipeCtx, _: *Env) Error!void {
        _ = self;
    }

    fn nextValue(self: *ProjectOp, pc: *PipeCtx) Error!?Produced {
        const env = (try opNextEnv(self.child, pc)) orelse return null;
        return .{
            .value = try evalExpr(pc.ctx(), self.expr, env, pc.depth),
            .env = env,
        };
    }

    fn close(self: *ProjectOp, _: *PipeCtx) void {
        _ = self;
    }
};

const GroupOutOp = struct {
    proj: *const Expr,
    key: *const Expr,
    child: *Op,
    groups: []Value = &.{},
    index: usize = 0,
    ready: bool = false,

    fn open(self: *GroupOutOp, _: *PipeCtx, _: *Env) Error!void {
        self.groups = &.{};
        self.index = 0;
        self.ready = false;
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
    }
};

/// Terminal `… into id;` — collect values from a producer (`project` / `group_out`) into the script env.
const ScriptBindOp = struct {
    name: []const u8,
    child: *Op,
    done: bool = false,

    fn open(self: *ScriptBindOp, _: *PipeCtx, _: *Env) Error!void {
        self.done = false;
    }

    fn nextValue(self: *ScriptBindOp, pc: *PipeCtx) Error!?Produced {
        if (self.done) return null;
        self.done = true;

        var list: std.ArrayList(Value) = .empty;
        defer list.deinit(pc.parent);
        while (try opNextValue(self.child, pc)) |row| {
            try list.append(pc.parent, try row.value.dupe(pc.parent));
        }
        try bindScriptValues(pc, self.name, list.items);
        return null;
    }

    fn close(self: *ScriptBindOp, _: *PipeCtx) void {
        _ = self;
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
    var list: std.ArrayList(Env) = .empty;
    errdefer list.deinit(pc.parent);
    while (try opNextEnv(child, pc)) |env| {
        try list.append(pc.parent, try env.dupe(pc.parent));
    }
    return try list.toOwnedSlice(pc.parent);
}

fn opChild(op: *const Op) ?*Op {
    return switch (op.*) {
        .from => |*f| f.child,
        .where => |*w| w.child,
        .let => |*l| l.child,
        .join => |*j| j.child,
        .order_by => |*o| o.child,
        .group_into => |*g| g.child,
        .select_into => |*s| s.child,
        .project => |*p| p.child,
        .group_out => |*g| g.child,
        .script_bind => |*s| s.child,
    };
}

fn opOpen(op: *Op, pc: *PipeCtx, outer: *Env) Error!void {
    var current: ?*Op = op;
    while (current) |c| {
        switch (c.*) {
            inline else => |*s| try s.open(pc, outer),
        }
        current = opChild(c);
    }
}

fn opClose(op: *Op, pc: *PipeCtx) void {
    var current: ?*Op = op;
    while (current) |c| {
        switch (c.*) {
            inline else => |*s| s.close(pc),
        }
        current = opChild(c);
    }
}

fn nextEnvChild(_: *Op, pc: *PipeCtx) Error!?*Env {
    return switch (pc.env_result) {
        .none => error.CallChild,
        .value => |v| blk: {
            pc.env_result = .none;
            break :blk v;
        },
    };
}

fn opNextEnv(op: *Op, pc: *PipeCtx) Error!?*Env {
    const base = pc.env_calls.items.len;
    defer pc.env_calls.shrinkRetainingCapacity(base);
    try pc.env_calls.append(pc.parent, op);

    while (pc.env_calls.items.len > base) {
        const current = pc.env_calls.last().?;
        const result = switch (current.*) {
            .project, .group_out, .script_bind => unreachable,
            inline else => |*s| s.nextEnv(pc),
        } catch |err| {
            if (err != error.CallChild) return err;
            try pc.env_calls.append(pc.parent, opChild(current).?);
            continue;
        };
        _ = pc.env_calls.pop();
        pc.env_result = .{ .value = result };
    }

    const result = pc.env_result.value;
    pc.env_result = .none;
    return result;
}

fn opNextValue(op: *Op, pc: *PipeCtx) Error!?Produced {
    return switch (op.*) {
        inline .project, .group_out, .script_bind => |*s| s.nextValue(pc),
        else => unreachable,
    };
}

fn createOp(gpa: std.mem.Allocator, op: Op) Error!*Op {
    const p = try gpa.create(Op);
    p.* = op;
    return p;
}

fn wrapClause(gpa: std.mem.Allocator, clause: *const plan.Clause, input: *Op) Error!*Op {
    var current = clause;
    var output = input;
    while (true) {
        switch (current.*) {
            .where => |w| {
                output = try createOp(gpa, .{ .where = .{ .pred = w.pred, .child = output } });
                current = w.then;
            },
            .let => |l| {
                output = try createOp(gpa, .{ .let = .{ .name = l.name, .expr = l.expr, .child = output } });
                current = l.then;
            },
            .from => |f| {
                output = try createOp(gpa, .{ .from = .{ .from = f, .child = output } });
                current = f.then;
            },
            .join => |j| {
                output = try createOp(gpa, .{ .join = .{
                    .join = j,
                    .child = output,
                    .inners_arena = .init(gpa),
                } });
                current = j.then;
            },
            .order_by => |o| {
                output = try createOp(gpa, .{ .order_by = .{ .keys = o.keys, .child = output } });
                current = o.then;
            },
            .group_by => |g| {
                if (g.into) |into| {
                    if (into.body) |body| {
                        output = try createOp(gpa, .{
                            .group_into = .{ .proj = g.proj, .key = g.key, .into_name = into.name, .child = output },
                        });
                        current = body;
                        continue;
                    }
                    const out = try createOp(gpa, .{
                        .group_out = .{ .proj = g.proj, .key = g.key, .child = output },
                    });
                    return wrapScriptBind(gpa, into.name, out);
                }
                return createOp(gpa, .{ .group_out = .{ .proj = g.proj, .key = g.key, .child = output } });
            },
            .select => |sel| {
                if (sel.into) |into| {
                    if (into.body) |body| {
                        output = try createOp(gpa, .{
                            .select_into = .{ .name = into.name, .expr = sel.expr, .child = output },
                        });
                        current = body;
                        continue;
                    }
                    const proj = try createOp(gpa, .{ .project = .{ .expr = sel.expr, .child = output } });
                    return wrapScriptBind(gpa, into.name, proj);
                }
                return createOp(gpa, .{ .project = .{ .expr = sel.expr, .child = output } });
            },
        }
    }
}

fn wrapScriptBind(gpa: std.mem.Allocator, name: []const u8, child: *Op) Error!*Op {
    return createOp(gpa, .{ .script_bind = .{ .name = name, .child = child } });
}

fn buildRoot(gpa: std.mem.Allocator, root: *const plan.From) Error!*Op {
    const scan = try createOp(gpa, .{ .from = .{ .from = root, .child = null } });
    return wrapClause(gpa, root.then, scan);
}

fn evalQueryValues(ctx: Ctx, query: *const plan.From, outer: *Env, depth: u32) Error![]Value {
    var out: std.ArrayList(Value) = .empty;
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

    var pc: PipeCtx = .{
        .io = ctx.io,
        .out = ctx.out,
        .depth = depth,
        .parent = ctx.allocator,
        .row_alloc = ctx.allocator,
        .script = outer,
        .script_alloc = script_alloc,
    };
    defer pc.env_calls.deinit(pc.parent);

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

    // Reject incomparable / mixed key kinds before sort (std.mem.sort cannot
    // bubble errors). Singleton rows still need a comparable key (§6.5).
    for (order_keys, 0..) |ok, col| {
        if (indexed.len == 0) break;
        const baseline = indexed[0].keys[col];
        _ = baseline.compare(baseline) catch |err| return failExpr(ok.expr, err);
        for (indexed[1..]) |ix| {
            _ = baseline.compare(ix.keys[col]) catch |err| return failExpr(ok.expr, err);
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
    items: std.ArrayList(Value) = .empty,
};

fn buildGroups(
    ctx: Ctx,
    rows: []Env,
    proj: *const Expr,
    key_expr: *const Expr,
    depth: u32,
) Error![]Value {
    var buckets: std.ArrayList(GroupBucket) = .empty;
    defer {
        for (buckets.items) |*b| b.items.deinit(ctx.allocator);
        buckets.deinit(ctx.allocator);
    }

    for (rows) |*row| {
        const k = try unwrapScalar(key_expr, try evalExpr(ctx, key_expr, row, depth), false);
        _ = k.eql(k) catch |err| return failExpr(key_expr, err);
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

fn sinkSelect(ctx: Ctx, e: *const Expr, env: *Env, v: Value) Error!void {
    // Hash.<algo> restore already printed via evalProp (§4.4).
    if (isHashRestoreSelect(e, env)) return;
    try sinkPrint(ctx, v);
}

/// Execute a query plan. `script` is the shared multi-statement environment
/// (also used as the root outer env). `script_alloc` owns values stored by
/// terminal `into id;` binds and must outlive the query arena in `ctx.allocator`.
pub fn run(ctx: Ctx, query: *const plan.From, script: *Env, script_alloc: std.mem.Allocator) Error!void {
    try runPipeline(ctx, query, script, 0, .sink, script_alloc);
}

// --- tests ------------------------------------------------------------------

fn testCtx(gpa: std.mem.Allocator, out: *std.Io.Writer) Ctx {
    return .{
        .allocator = gpa,
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
    // Arrange
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

    // Act

    // Assert
    try std.testing.expectError(error.InvalidWindow, evalExpr(ctx, &call, &env, 0));
}

test "negative tree depth is InvalidTreeDepth" {
    // Arrange
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

    // Act

    // Assert
    try std.testing.expectError(error.InvalidTreeDepth, evalExpr(ctx, &call, &env, 0));
}

test "hash min/max reject zero negative overflow; restore rejects inverted range and oversized max" {
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    const ctx = testCtx(a, &writer);

    var env: Env = .{};
    defer env.deinit(a);
    try env.put(a, "h", Value.hashDigest("00"));

    var name_h: Expr = .{ .kind = .{ .name = "h" } };

    // Act

    // Assert
    var zero: Expr = .{ .kind = .{ .int_lit = 0 } };
    var zero_args = [_]*Expr{&zero};
    var zero_call: Expr = .{ .kind = .{ .method = .{ .recv = &name_h, .name = "min", .args = &zero_args, .kind = .hash_min } } };
    try std.testing.expectError(error.InvalidRestoreBound, evalExpr(ctx, &zero_call, &env, 0));

    var neg: Expr = .{ .kind = .{ .int_lit = -1 } };
    var neg_args = [_]*Expr{&neg};
    var neg_call: Expr = .{ .kind = .{ .method = .{ .recv = &name_h, .name = "max", .args = &neg_args, .kind = .hash_max } } };
    try std.testing.expectError(error.InvalidRestoreBound, evalExpr(ctx, &neg_call, &env, 0));

    var big: Expr = .{ .kind = .{ .int_lit = @as(i64, std.math.maxInt(i32)) + 1 } };
    var big_args = [_]*Expr{&big};
    var big_call: Expr = .{ .kind = .{ .method = .{ .recv = &name_h, .name = "min", .args = &big_args, .kind = .hash_min } } };
    try std.testing.expectError(error.Overflow, evalExpr(ctx, &big_call, &env, 0));

    var five: Expr = .{ .kind = .{ .int_lit = 5 } };
    var two: Expr = .{ .kind = .{ .int_lit = 2 } };
    var min_args = [_]*Expr{&five};
    var min_call: Expr = .{ .kind = .{ .method = .{ .recv = &name_h, .name = "min", .args = &min_args, .kind = .hash_min } } };
    const raised = try evalExpr(ctx, &min_call, &env, 0);
    try env.put(a, "h5", raised);
    var name_h5: Expr = .{ .kind = .{ .name = "h5" } };
    var max_args = [_]*Expr{&two};
    var max_call: Expr = .{ .kind = .{ .method = .{ .recv = &name_h5, .name = "max", .args = &max_args, .kind = .hash_max } } };
    const inverted = try evalExpr(ctx, &max_call, &env, 0);
    try std.testing.expectEqual(@as(i32, 5), inverted.hash.min);
    try std.testing.expectEqual(@as(i32, 2), inverted.hash.max);

    try env.put(a, "inv", .{ .hash = .{
        .digest = "D41D8CD98F00B204E9800998ECF8427E",
        .min = 5,
        .max = 2,
    } });
    var name_inv: Expr = .{ .kind = .{ .name = "inv" } };
    var md5_e: Expr = .{ .kind = .{ .prop = .{ .recv = &name_inv, .prop = "md5", .access = .hash_algo } } };
    try std.testing.expectError(error.InvalidRestoreRange, evalExpr(ctx, &md5_e, &env, 0));

    try env.put(a, "huge", .{ .hash = .{
        .digest = "202CB962AC59075B964B07152D234B70",
        .max = 999999999,
        .no_probe = true,
    } });
    var name_huge: Expr = .{ .kind = .{ .name = "huge" } };
    var md5_huge: Expr = .{ .kind = .{ .prop = .{ .recv = &name_huge, .prop = "md5", .access = .hash_algo } } };
    try std.testing.expectError(error.InvalidRestoreLength, evalExpr(ctx, &md5_huge, &env, 0));
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
    try std.testing.expectError(error.TypeMismatch, sinkPrint(ctx, .{ .record = &rec }));
}

test "orderRows fails when key kinds differ across rows" {
    // Arrange
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

    // Act

    // Assert
    try std.testing.expectError(error.TypeMismatch, orderRows(ctx, &rows, &keys, 0));
}

test "from file in mixed sequence fails type check" {
    // Arrange
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

    // Act

    // Assert
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

    // Act

    // Assert
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

    // Act

    // Assert
    try std.testing.expect(exprJoinSourceStable(&lit, &script, &outer_no_shadow));
    try std.testing.expect(exprJoinSourceStable(&script_name, &script, &outer_no_shadow));
    try std.testing.expect(!exprJoinSourceStable(&script_name, &script, &outer_shadows));
    try std.testing.expect(!exprJoinSourceStable(&row_name, &script, &outer_no_shadow));
    try std.testing.expect(!exprJoinSourceStable(&nested, &script, &outer_no_shadow));
}

test "join Dir rematerialize reclaims listings via inners arena reset" {
    // Non-cached join rebuilds Dir listings per outer. Freeing only the []Value
    // slice leaves path bytes behind; a resettable arena (as JoinOp uses) must
    // own those paths so rematerialize does not grow unbounded. Child allocator
    // is testing.allocator so a bypass shows up as a leak.
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const base = "l2h_join_dir_remat_probe";
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().deleteTree(io, base) catch {};
    std.Io.Dir.cwd().createDir(io, base, perms) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};
    {
        var f = try std.Io.Dir.cwd().createFile(io, base ++ "/a.txt", .{});
        defer f.close(io);
        try f.writeStreamingAll(io, "a");
    }
    {
        var f = try std.Io.Dir.cwd().createFile(io, base ++ "/b.txt", .{});
        defer f.close(io);
        try f.writeStreamingAll(io, "bb");
    }

    var inners_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer inners_arena.deinit();

    var out_buf: [64]u8 = undefined;
    var out_w: std.Io.Writer = .fixed(&out_buf);
    const dir: value.DirVal = .{ .path = base, .max_depth = 0, .skip_errors = false };

    // Act
    var cycle: usize = 0;
    while (cycle < 2) : (cycle += 1) {
        const ctx = testCtx(inners_arena.allocator(), &out_w);
        const inners = try collectDirFiles(ctx, dir);

        // Assert
        try std.testing.expectEqual(@as(usize, 2), inners.len);
        _ = inners_arena.reset(.retain_capacity);
    }
}

/// Test allocator wrapper that records the peak of live bytes.
const PeakAllocator = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,

    fn allocator(self: *PeakAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
        } };
    }

    fn track(self: *PeakAllocator, old_len: usize, new_len: usize) void {
        self.live = self.live - old_len + new_len;
        self.peak = @max(self.peak, self.live);
    }

    fn alloc(ptr: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *PeakAllocator = @ptrCast(@alignCast(ptr));
        const p = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.track(0, len);
        return p;
    }

    fn resize(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *PeakAllocator = @ptrCast(@alignCast(ptr));
        if (!self.child.rawResize(memory, alignment, new_len, ret_addr)) return false;
        self.track(memory.len, new_len);
        return true;
    }

    fn remap(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *PeakAllocator = @ptrCast(@alignCast(ptr));
        const p = self.child.rawRemap(memory, alignment, new_len, ret_addr) orelse return null;
        self.track(memory.len, new_len);
        return p;
    }

    fn free(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *PeakAllocator = @ptrCast(@alignCast(ptr));
        self.child.rawFree(memory, alignment, ret_addr);
        self.track(memory.len, 0);
    }
};

const PerRowShape = enum { nested_from, join };

/// Peak bytes of `from dir d in '<base>' from file f in d <shape> select …`
/// over a per-query arena, as `frontend.handleQueryAst` runs it.
fn peakPerRowQuery(base: []const u8, shape: PerRowShape) !usize {
    var peak: PeakAllocator = .{ .child = std.testing.allocator };
    var arena = std.heap.ArenaAllocator.init(peak.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    var discard_buf: [64]u8 = undefined;
    var discard: std.Io.Writer.Discarding = .init(&discard_buf);
    const ctx: Ctx = .{ .allocator = a, .io = std.testing.io, .out = &discard.writer };

    var dir_lit: Expr = .{ .kind = .{ .string_lit = base } };
    var name_d: Expr = .{ .kind = .{ .name = "d" } };
    var name_f: Expr = .{ .kind = .{ .name = "f" } };
    var name_t: Expr = .{ .kind = .{ .name = "t" } };
    var x_lit: Expr = .{ .kind = .{ .string_lit = "x" } };
    var f_name: Expr = .{ .kind = .{ .prop = .{ .recv = &name_f, .prop = "name", .access = .name } } };

    var select: plan.Select = .{ .expr = &name_t };
    var c_select: plan.Clause = .{ .select = &select };
    var from_t: plan.From = .{ .kind = .string, .range = "t", .source = &f_name, .then = &c_select };
    var join_t: plan.Join = .{
        .kind = .string,
        .range = "t",
        .source = &x_lit,
        .outer_key = &f_name,
        .inner_key = &name_t,
        .then = &c_select,
    };
    var c_inner: plan.Clause = switch (shape) {
        .nested_from => .{ .from = &from_t },
        .join => .{ .join = &join_t },
    };
    var from_f: plan.From = .{ .kind = .file, .range = "f", .source = &name_d, .then = &c_inner };
    var c_from_f: plan.Clause = .{ .from = &from_f };
    var root: plan.From = .{ .kind = .dir, .range = "d", .source = &dir_lit, .then = &c_from_f };

    var outer: Env = .{};
    try runPipeline(ctx, &root, &outer, 0, .sink, a);
    return peak.peak;
}

fn makeProbeFiles(io: std.Io, base: []const u8, count: usize) !void {
    const perms = std.Io.Dir.Permissions.default_dir;
    std.Io.Dir.cwd().deleteTree(io, base) catch {};
    try std.Io.Dir.cwd().createDir(io, base, perms);
    var dir = try std.Io.Dir.cwd().openDir(io, base, .{});
    defer dir.close(io);
    var name_buf: [16]u8 = undefined;
    for (0..count) |i| {
        const name = try std.mem.print(&name_buf, "f{d:0>5}", .{i});
        var f = try dir.createFile(io, name, .{});
        f.close(io);
    }
}

test "nested from and join over a dir walk keep per-row memory bounded" {
    // Per-outer-row state (bound source, outer env copy, join key) used to
    // pile up in the query arena: ~700 bytes per walked file, so a big tree
    // grew without bound. Peak memory must not depend on the file count.
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();
    const few = "l2h_per_row_few_probe";
    const many = "l2h_per_row_many_probe";
    defer std.Io.Dir.cwd().deleteTree(io, few) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, many) catch {};
    try makeProbeFiles(io, few, 16);
    try makeProbeFiles(io, many, 1024);

    for ([_]PerRowShape{ .nested_from, .join }) |shape| {
        // Act
        const peak_few = try peakPerRowQuery(few, shape);
        const peak_many = try peakPerRowQuery(many, shape);

        // Assert
        errdefer std.debug.print("{t}: peak {d} bytes for 16 files, {d} for 1024\n", .{ shape, peak_few, peak_many });
        try std.testing.expect(peak_many < peak_few + 64 * 1024);
    }
}
