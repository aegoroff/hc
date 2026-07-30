const std = @import("std");
const c = @import("c");
const diag = @import("diag.zig");
const expr = @import("expr.zig");
const front = @import("frontend.zig");
const hashes = @import("hashes");
const plan = @import("plan.zig");
const value = @import("value.zig");

pub const Error = error{
    InvalidAst,
    UndefinedName,
    InvalidFromSourceType,
    InvalidProperty,
    TypeMismatch,
    DuplicateField,
    UnsupportedNode,
    UnsupportedMethodCall,
    InvalidRecordField,
};

const LowerError = Error || std.mem.Allocator.Error;

fn fail(sp: expr.Span, err: Error) Error {
    diag.noteSpan(sp);
    return err;
}

fn failNode(node: *const c.fend_node_t, err: Error) Error {
    diag.noteNode(node);
    return err;
}

const TypeInfo = union(enum) {
    unknown,
    string,
    file,
    dir,
    hash,
    int,
    bool,
    record_unknown,
    record: []const RecordFieldType,
    seq: *const TypeInfo,
};

const RecordFieldType = struct {
    name: []const u8,
    ty: *const TypeInfo,
};

fn cloneType(allocator: std.mem.Allocator, ty: TypeInfo) !*const TypeInfo {
    const out = try allocator.create(TypeInfo);
    switch (ty) {
        .seq => |item| out.* = .{ .seq = try cloneType(allocator, item.*) },
        .record => |fields| {
            const copy = try allocator.alloc(RecordFieldType, fields.len);
            for (fields, 0..) |field, i| {
                copy[i] = .{
                    .name = field.name,
                    .ty = try cloneType(allocator, field.ty.*),
                };
            }
            out.* = .{ .record = copy };
        },
        else => out.* = ty,
    }
    return out;
}

fn wrapSeq(allocator: std.mem.Allocator, item_ty: TypeInfo) !TypeInfo {
    return .{ .seq = try cloneType(allocator, item_ty) };
}

fn sameType(a: TypeInfo, b: TypeInfo) bool {
    return switch (a) {
        .unknown => b == .unknown,
        .string => b == .string,
        .file => b == .file,
        .dir => b == .dir,
        .hash => b == .hash,
        .int => b == .int,
        .bool => b == .bool,
        .record_unknown => b == .record_unknown,
        .record => |af| switch (b) {
            .record => |bf| blk: {
                if (af.len != bf.len) break :blk false;
                for (af, bf) |a_field, b_field| {
                    if (!std.mem.eql(u8, a_field.name, b_field.name)) break :blk false;
                    if (!sameType(a_field.ty.*, b_field.ty.*)) break :blk false;
                }
                break :blk true;
            },
            else => false,
        },
        .seq => |ai| switch (b) {
            .seq => |bi| sameType(ai.*, bi.*),
            else => false,
        },
    };
}

fn recordFieldType(rec: []const RecordFieldType, name: []const u8) ?TypeInfo {
    for (rec) |field| {
        if (std.mem.eql(u8, field.name, name)) return field.ty.*;
    }
    return null;
}

fn cloneScope(
    allocator: std.mem.Allocator,
    src: *const std.StringHashMapUnmanaged(TypeInfo),
) !std.StringHashMapUnmanaged(TypeInfo) {
    var out: std.StringHashMapUnmanaged(TypeInfo) = .empty;
    var it = src.iterator();
    while (it.next()) |entry| {
        try out.put(allocator, entry.key_ptr.*, entry.value_ptr.*);
    }
    return out;
}

fn span(s: [*c]u8) []const u8 {
    if (s == null) return "";
    return std.mem.span(@as([*:0]u8, @ptrCast(s)));
}

fn dup(allocator: std.mem.Allocator, s: []const u8) ![]const u8 {
    return try allocator.dupe(u8, s);
}

fn trimQuotes(s_in: []const u8) []const u8 {
    var s = s_in;
    while (s.len > 0 and (s[0] == '\'' or s[0] == '"')) s = s[1..];
    while (s.len > 0 and (s[s.len - 1] == '\'' or s[s.len - 1] == '"')) s = s[0 .. s.len - 1];
    return s;
}

fn lowerType(node: *const c.fend_node_t) Error!plan.SourceKind {
    if (node.type != c.node_type_identifier or node.left == null) return error.InvalidAst;
    const type_node: *c.fend_node_t = node.left orelse return error.InvalidAst;
    if (type_node.type != c.node_type_internal_type) return error.InvalidAst;
    return switch (@as(c_int, @intCast(type_node.value.type))) {
        c.type_def_string => .string,
        c.type_def_file => .file,
        c.type_def_dir => .dir,
        c.type_def_custom => .hash,
        else => error.InvalidAst,
    };
}

fn lowerName(allocator: std.mem.Allocator, node: *const c.fend_node_t) ![]const u8 {
    if (node.type != c.node_type_identifier) return error.InvalidAst;
    return dup(allocator, span(node.value.string));
}

fn flattenEnum(
    allocator: std.mem.Allocator,
    node: ?*c.fend_node_t,
    out: *std.ArrayList(*c.fend_node_t),
) !void {
    const n = node orelse return;
    if (n.type == c.node_type_enum) {
        try flattenEnum(allocator, n.left, out);
        try flattenEnum(allocator, n.right, out);
        return;
    }
    try out.append(allocator, n);
}

fn lowerRecordFields(allocator: std.mem.Allocator, node: *const c.fend_node_t) LowerError![]expr.RecordFieldExpr {
    var items: std.ArrayList(*c.fend_node_t) = .empty;
    defer items.deinit(allocator);
    try flattenEnum(allocator, @constCast(node), &items);

    const fields = try allocator.alloc(expr.RecordFieldExpr, items.items.len);
    for (items.items, 0..) |item, i| {
        if (item.type == c.node_type_let and item.left != null and item.right != null) {
            fields[i] = .{
                .name = try lowerName(allocator, item.left.?),
                .expr = try lowerExpr(allocator, item.right.?),
            };
        } else {
            const field_expr = try lowerExpr(allocator, item);
            fields[i] = .{
                .name = try dup(allocator, expr.autoFieldName(field_expr) catch {
                    return fail(field_expr.span, error.InvalidRecordField);
                }),
                .expr = field_expr,
            };
        }
    }
    return fields;
}

pub fn lowerExpr(allocator: std.mem.Allocator, node: *const c.fend_node_t) LowerError!*expr.Expr {
    const out = try allocator.create(expr.Expr);
    const sp = expr.Span.fromNode(node);
    switch (node.type) {
        c.node_type_unary_expression => {
            if (node.right != null) {
                const rhs: *c.fend_node_t = node.right.?;
                if (rhs.type == c.node_type_property) {
                    const recv = try lowerExpr(allocator, node.left.?);
                    out.* = .{
                        .span = sp,
                        .kind = .{
                            .prop = .{
                                .recv = recv,
                                .prop = try dup(allocator, span(rhs.value.string)),
                            },
                        },
                    };
                    return out;
                }
                if (rhs.type == c.node_type_method_call)
                    return failNode(node, error.UnsupportedMethodCall);
            }
            // Grammar wraps literals/ids in unary nodes and FLOCs the wrapper;
            // the child often has an unset loc — keep the wrapper span.
            const inner = try lowerExpr(allocator, node.left.?);
            if (!inner.span.isSet() and sp.isSet()) inner.span = sp;
            // `out` unused on this path.
            allocator.destroy(out);
            return inner;
        },
        c.node_type_identifier => out.* = .{
            .span = sp,
            .kind = .{ .name = try dup(allocator, span(node.value.string)) },
        },
        c.node_type_string_literal => out.* = .{
            .span = sp,
            .kind = .{ .string_lit = try dup(allocator, trimQuotes(span(node.value.string))) },
        },
        c.node_type_numeric_literal => out.* = .{
            .span = sp,
            .kind = .{ .int_lit = node.value.number },
        },
        c.node_type_relation => out.* = .{
            .span = sp,
            .kind = .{
                .binary = .{
                    .op = switch (@as(c_int, @intCast(node.value.relation_op))) {
                        c.cond_op_eq => .eq,
                        c.cond_op_not_eq => .neq,
                        c.cond_op_ge => .gt,
                        c.cond_op_le => .lt,
                        c.cond_op_ge_eq => .ge,
                        c.cond_op_le_eq => .le,
                        c.cond_op_match => .match,
                        c.cond_op_not_match => .not_match,
                        else => return failNode(node, error.UnsupportedNode),
                    },
                    .left = try lowerExpr(allocator, node.left.?),
                    .right = try lowerExpr(allocator, node.right.?),
                },
            },
        },
        c.node_type_and_rel => out.* = .{
            .span = sp,
            .kind = .{
                .binary = .{
                    .op = .and_,
                    .left = try lowerExpr(allocator, node.left.?),
                    .right = try lowerExpr(allocator, node.right.?),
                },
            },
        },
        c.node_type_or_rel => out.* = .{
            .span = sp,
            .kind = .{
                .binary = .{
                    .op = .or_,
                    .left = try lowerExpr(allocator, node.left.?),
                    .right = try lowerExpr(allocator, node.right.?),
                },
            },
        },
        c.node_type_not_rel => out.* = .{
            .span = sp,
            .kind = .{
                .unary = .{
                    .op = .not_,
                    .arg = try lowerExpr(allocator, node.left.?),
                },
            },
        },
        c.node_type_enum, c.node_type_object => out.* = .{
            .span = sp,
            .kind = .{
                .record = try lowerRecordFields(allocator, if (node.type == c.node_type_object) node.left.? else node),
            },
        },
        c.node_type_query => out.* = .{
            .span = sp,
            .kind = .{ .query_ast = node },
        },
        else => return failNode(node, error.UnsupportedNode),
    }
    return out;
}

fn lowerSourceExpr(
    allocator: std.mem.Allocator,
    kind: plan.SourceKind,
    node: *const c.fend_node_t,
) LowerError!plan.SourceExpr {
    if (kind == .file and node.type == c.node_type_unary_expression and node.right == null) {
        const base: *c.fend_node_t = node.left orelse return error.InvalidAst;
        if (base.type == c.node_type_identifier) {
            const name = span(base.value.string);
            const declared = front.identifierDeclaredType(name);
            if (declared == c.type_def_dir) {
                return .{ .files_in_dir = try dup(allocator, name) };
            }
            if (declared != null) {
                return failNode(node, error.InvalidFromSourceType);
            }
        }
    }
    return .{ .expr = try lowerExpr(allocator, node) };
}

fn lowerOrderKeys(allocator: std.mem.Allocator, node: *const c.fend_node_t) LowerError![]plan.OrderKey {
    var items: std.ArrayList(*c.fend_node_t) = .empty;
    defer items.deinit(allocator);
    try flattenEnum(allocator, @constCast(node), &items);

    const keys = try allocator.alloc(plan.OrderKey, items.items.len);
    for (items.items, 0..) |item, i| {
        if (item.type != c.node_type_ordering or item.left == null) return error.InvalidAst;
        keys[i] = .{
            .expr = try lowerExpr(allocator, item.left.?),
            .descending = item.value.ordering == c.ordering_desc,
        };
    }
    return keys;
}

fn lowerClauseNode(
    allocator: std.mem.Allocator,
    node: *const c.fend_node_t,
    then: *plan.Clause,
) LowerError!*plan.Clause {
    const out = try allocator.create(plan.Clause);
    switch (node.type) {
        c.node_type_from => {
            const decl = node.left orelse return error.InvalidAst;
            const src = node.right orelse return error.InvalidAst;
            const from = try allocator.create(plan.From);
            from.* = .{
                .kind = try lowerType(decl),
                .range = try lowerName(allocator, decl),
                .source = try lowerSourceExpr(allocator, try lowerType(decl), src),
                .then = then,
            };
            out.* = .{ .from = from };
        },
        c.node_type_where => out.* = .{
            .where = .{
                .pred = try lowerExpr(allocator, node.left.?),
                .then = then,
            },
        },
        c.node_type_let => out.* = .{
            .let = .{
                .name = try lowerName(allocator, node.left.?),
                .expr = try lowerExpr(allocator, node.right.?),
                .then = then,
            },
        },
        c.node_type_join => {
            const decl: *c.fend_node_t = node.left orelse return error.InvalidAst;
            const in_node: *c.fend_node_t = node.right orelse return error.InvalidAst;
            if (in_node.type != c.node_type_in or in_node.left == null or in_node.right == null)
                return error.InvalidAst;
            const on_node: *c.fend_node_t = in_node.right.?;
            if (on_node.type != c.node_type_on or on_node.left == null or on_node.right == null)
                return error.InvalidAst;
            const join = try allocator.create(plan.Join);
            join.* = .{
                .kind = try lowerType(decl),
                .range = try lowerName(allocator, decl),
                .source = try lowerSourceExpr(allocator, try lowerType(decl), in_node.left.?),
                .outer_key = try lowerExpr(allocator, on_node.left.?),
                .inner_key = try lowerExpr(allocator, on_node.right.?),
                .then = then,
            };
            out.* = .{ .join = join };
        },
        c.node_type_order_by => out.* = .{
            .order_by = .{
                .keys = try lowerOrderKeys(allocator, node.left.?),
                .then = then,
            },
        },
        else => return error.UnsupportedNode,
    }
    return out;
}

fn lowerContinuationBody(allocator: std.mem.Allocator, node: *const c.fend_node_t) LowerError!*plan.Clause {
    return try lowerBody(allocator, node);
}

fn lowerTerminalClause(
    allocator: std.mem.Allocator,
    terminal: *const c.fend_node_t,
    continuation: ?*c.fend_node_t,
) LowerError!*plan.Clause {
    const out = try allocator.create(plan.Clause);
    switch (terminal.type) {
        c.node_type_group => {
            var into: ?plan.Into = null;
            if (continuation) |cont| {
                if (cont.type != c.node_type_query_continuation or cont.left == null or cont.right == null)
                    return error.InvalidAst;
                into = .{
                    .name = try lowerName(allocator, cont.left.?),
                    .body = try lowerContinuationBody(allocator, cont.right.?),
                };
            }
            out.* = .{
                .group_by = .{
                    .proj = try lowerExpr(allocator, terminal.left.?),
                    .key = try lowerExpr(allocator, terminal.right.?),
                    .into = into,
                },
            };
        },
        else => {
            const sel = try allocator.create(plan.Select);
            var into: ?plan.Into = null;
            if (continuation) |cont| {
                if (cont.type != c.node_type_query_continuation or cont.left == null or cont.right == null)
                    return error.InvalidAst;
                into = .{
                    .name = try lowerName(allocator, cont.left.?),
                    .body = try lowerContinuationBody(allocator, cont.right.?),
                };
            }
            sel.* = .{
                .expr = try lowerExpr(allocator, terminal),
                .into = into,
            };
            out.* = .{ .select = sel };
        },
    }
    return out;
}

fn lowerBody(allocator: std.mem.Allocator, body: *const c.fend_node_t) LowerError!*plan.Clause {
    if (body.type != c.node_type_query_body or body.left == null) return error.InvalidAst;
    const select_wrap: *c.fend_node_t = body.left.?;
    if (select_wrap.type != c.node_type_select or select_wrap.right == null) return error.InvalidAst;

    var tail = try lowerTerminalClause(allocator, select_wrap.right.?, body.right);

    var clauses: std.ArrayList(*c.fend_node_t) = .empty;
    defer clauses.deinit(allocator);
    try flattenEnum(allocator, select_wrap.left, &clauses);

    var i: usize = clauses.items.len;
    while (i > 0) {
        i -= 1;
        const next_then = tail;
        const current = clauses.items[i];

        if (current.type == c.node_type_query_continuation) {
            if (current.left == null or current.right == null) return error.InvalidAst;
            const join_clause: *c.fend_node_t = current.right.?;
            if (join_clause.type != c.node_type_join) return error.InvalidAst;
            const wrapped = try lowerClauseNode(allocator, join_clause, next_then);
            if (wrapped.* != .join) return error.InvalidAst;
            wrapped.join.group_into = try lowerName(allocator, current.left.?);
            tail = wrapped;
            continue;
        }

        tail = try lowerClauseNode(allocator, current, next_then);
    }
    return tail;
}

fn typeFromValue(allocator: std.mem.Allocator, v: value.Value) LowerError!TypeInfo {
    return switch (v) {
        .string => .string,
        .file => .file,
        .dir => .dir,
        .hash => .hash,
        .int => .int,
        .bool => .bool,
        .record => |rec| blk: {
            const fields = try allocator.alloc(RecordFieldType, rec.fields.len);
            for (rec.fields, 0..) |field, i| {
                const field_ty = try typeFromValue(allocator, field.value);
                fields[i] = .{
                    .name = field.name,
                    .ty = try cloneType(allocator, field_ty),
                };
            }
            break :blk .{ .record = fields };
        },
        .seq => |seq| blk: {
            if (seq.items.len == 0) break :blk try wrapSeq(allocator, .unknown);
            const first = try typeFromValue(allocator, seq.items[0]);
            for (seq.items[1..]) |item| {
                const next = try typeFromValue(allocator, item);
                if (!sameType(first, next)) break :blk try wrapSeq(allocator, .unknown);
            }
            break :blk try wrapSeq(allocator, first);
        },
    };
}

fn scopeFromEnv(
    allocator: std.mem.Allocator,
    env: *const value.Env,
) LowerError!std.StringHashMapUnmanaged(TypeInfo) {
    var out: std.StringHashMapUnmanaged(TypeInfo) = .empty;
    var it = env.map.iterator();
    while (it.next()) |entry| {
        try out.put(allocator, entry.key_ptr.*, try typeFromValue(allocator, entry.value_ptr.*));
    }
    return out;
}

fn scalarSourceTypeAllowed(ty: TypeInfo) bool {
    return switch (ty) {
        .unknown, .string, .file, .dir, .hash => true,
        else => false,
    };
}

fn comparableType(ty: TypeInfo) bool {
    return switch (ty) {
        .int, .string, .bool => true,
        else => false,
    };
}

/// Nested query / Seq in predicates means existence (non-empty).
fn asPredicateType(e: *const expr.Expr, ty: TypeInfo) LowerError!void {
    switch (ty) {
        .bool, .seq, .unknown => {},
        else => return fail(e.span, error.TypeMismatch),
    }
}

/// Singleton Seq unwrap for comparisons and order keys (nested queries only).
fn scalarCompareType(e: *const expr.Expr, ty: TypeInfo) LowerError!TypeInfo {
    if (ty == .seq) {
        if (e.kind != .query_ast) return fail(e.span, error.TypeMismatch);
        return switch (ty.seq.*) {
            .int, .string, .bool, .unknown => ty.seq.*,
            else => fail(e.span, error.TypeMismatch),
        };
    }
    return switch (ty) {
        .int, .string, .bool, .unknown => ty,
        else => fail(e.span, error.TypeMismatch),
    };
}

fn groupRecordType(
    allocator: std.mem.Allocator,
    key_ty: TypeInfo,
    item_ty: TypeInfo,
) LowerError!TypeInfo {
    const fields = try allocator.alloc(RecordFieldType, 2);
    fields[0] = .{
        .name = "key",
        .ty = try cloneType(allocator, key_ty),
    };
    fields[1] = .{
        .name = "items",
        .ty = try cloneType(allocator, try wrapSeq(allocator, item_ty)),
    };
    return .{ .record = fields };
}

fn inferQueryResultType(
    allocator: std.mem.Allocator,
    query: *const plan.QueryPlan,
    scope: *const std.StringHashMapUnmanaged(TypeInfo),
) LowerError!TypeInfo {
    var nested = try cloneScope(allocator, scope);
    defer nested.deinit(allocator);
    try nested.put(allocator, query.root.range, switch (query.root.kind) {
        .string => .string,
        .file => .file,
        .dir => .dir,
        .hash => .hash,
    });
    return inferClauseResultType(allocator, &nested, query.root.then);
}

fn inferClauseResultType(
    allocator: std.mem.Allocator,
    scope: *const std.StringHashMapUnmanaged(TypeInfo),
    clause: *const plan.Clause,
) LowerError!TypeInfo {
    switch (clause.*) {
        .where => |w| return inferClauseResultType(allocator, scope, w.then),
        .from => |f| {
            var next = try cloneScope(allocator, scope);
            defer next.deinit(allocator);
            try next.put(allocator, f.range, switch (f.kind) {
                .string => .string,
                .file => .file,
                .dir => .dir,
                .hash => .hash,
            });
            return inferClauseResultType(allocator, &next, f.then);
        },
        .let => |l| {
            var next = try cloneScope(allocator, scope);
            defer next.deinit(allocator);
            try next.put(allocator, l.name, try inferExprType(allocator, scope, l.expr));
            return inferClauseResultType(allocator, &next, l.then);
        },
        .join => |j| {
            var next = try cloneScope(allocator, scope);
            defer next.deinit(allocator);
            const j_ty: TypeInfo = switch (j.kind) {
                .string => .string,
                .file => .file,
                .dir => .dir,
                .hash => .hash,
            };
            if (j.group_into) |name| {
                try next.put(allocator, name, try wrapSeq(allocator, j_ty));
            } else {
                try next.put(allocator, j.range, j_ty);
            }
            return inferClauseResultType(allocator, &next, j.then);
        },
        .order_by => |o| return inferClauseResultType(allocator, scope, o.then),
        .group_by => |g| {
            const rec_ty = try groupRecordType(
                allocator,
                try inferExprType(allocator, scope, g.key),
                try inferExprType(allocator, scope, g.proj),
            );
            return wrapSeq(allocator, rec_ty);
        },
        .select => |s| {
            const item_ty = try inferExprType(allocator, scope, s.expr);
            if (s.into) |into| {
                var next = try cloneScope(allocator, scope);
                defer next.deinit(allocator);
                try next.put(allocator, into.name, item_ty);
                return inferClauseResultType(allocator, &next, into.body);
            }
            return wrapSeq(allocator, item_ty);
        },
    }
}

fn inferExprType(
    allocator: std.mem.Allocator,
    scope: *const std.StringHashMapUnmanaged(TypeInfo),
    e: *const expr.Expr,
) LowerError!TypeInfo {
    return switch (e.kind) {
        .query_ast => |ast| blk: {
            const nested = try lowerQueryWithScope(allocator, ast, scope);
            break :blk try inferQueryResultType(allocator, &nested, scope);
        },
        .string_lit => .string,
        .int_lit => .int,
        .value_lit => |v| try typeFromValue(allocator, v),
        .name => |name| scope.get(name) orelse fail(e.span, error.UndefinedName),
        .unary => |u| switch (u.op) {
            .not_ => blk: {
                const arg_ty = try inferExprType(allocator, scope, u.arg);
                try asPredicateType(u.arg, arg_ty);
                break :blk .bool;
            },
        },
        .binary => |b| blk: {
            const left_ty = try inferExprType(allocator, scope, b.left);
            const right_ty = try inferExprType(allocator, scope, b.right);
            switch (b.op) {
                .and_, .or_ => {
                    try asPredicateType(b.left, left_ty);
                    try asPredicateType(b.right, right_ty);
                },
                .match, .not_match => {
                    const l = try scalarCompareType(b.left, left_ty);
                    const r = try scalarCompareType(b.right, right_ty);
                    if (l != .string and l != .unknown) return fail(e.span, error.TypeMismatch);
                    if (r != .string and r != .unknown) return fail(e.span, error.TypeMismatch);
                },
                .gt, .ge, .lt, .le => {
                    const l = try scalarCompareType(b.left, left_ty);
                    const r = try scalarCompareType(b.right, right_ty);
                    if (l != .int and l != .unknown) return fail(e.span, error.TypeMismatch);
                    if (r != .int and r != .unknown) return fail(e.span, error.TypeMismatch);
                },
                .eq, .neq => {
                    const l = try scalarCompareType(b.left, left_ty);
                    const r = try scalarCompareType(b.right, right_ty);
                    if (l != .unknown and r != .unknown and !sameType(l, r)) return fail(e.span, error.TypeMismatch);
                    if (l != .unknown and !comparableType(l)) return fail(e.span, error.TypeMismatch);
                    if (r != .unknown and !comparableType(r)) return fail(e.span, error.TypeMismatch);
                },
            }
            break :blk .bool;
        },
        .record => |fields| blk: {
            const out = try allocator.alloc(RecordFieldType, fields.len);
            var seen: std.StringHashMapUnmanaged(void) = .empty;
            defer seen.deinit(allocator);
            for (fields, 0..) |field, i| {
                if ((try seen.fetchPut(allocator, field.name, {})) != null)
                    return fail(field.expr.span, error.DuplicateField);
                const field_ty = try inferExprType(allocator, scope, field.expr);
                out[i] = .{
                    .name = field.name,
                    .ty = try cloneType(allocator, field_ty),
                };
            }
            break :blk .{ .record = out };
        },
        .prop => |p| blk: {
            const recv_ty = try inferExprType(allocator, scope, p.recv);
            switch (recv_ty) {
                .string => {
                    if (std.mem.eql(u8, p.prop, "size")) break :blk .int;
                    if (hashes.getHash(p.prop) != null) break :blk .string;
                    return fail(e.span, error.InvalidProperty);
                },
                .file => {
                    if (std.mem.eql(u8, p.prop, "path")) break :blk .string;
                    if (std.mem.eql(u8, p.prop, "size")) break :blk .int;
                    if (hashes.getHash(p.prop) != null) break :blk .string;
                    return fail(e.span, error.InvalidProperty);
                },
                .hash => {
                    if (hashes.getHash(p.prop) != null) break :blk .string;
                    return fail(e.span, error.InvalidProperty);
                },
                .dir => {
                    if (std.mem.eql(u8, p.prop, "path")) break :blk .string;
                    return fail(e.span, error.InvalidProperty);
                },
                .int, .bool => return fail(e.span, error.InvalidProperty),
                .seq => return fail(e.span, error.InvalidProperty),
                .record_unknown => break :blk .record_unknown,
                .record => |rec| break :blk recordFieldType(rec, p.prop) orelse fail(e.span, error.InvalidProperty),
                .unknown => break :blk .unknown,
            }
        },
    };
}

fn validateSource(
    allocator: std.mem.Allocator,
    scope: *const std.StringHashMapUnmanaged(TypeInfo),
    kind: plan.SourceKind,
    source: plan.SourceExpr,
) LowerError!void {
    switch (source) {
        .files_in_dir => |name| {
            if (kind != .file) return error.InvalidFromSourceType;
            if (scope.get(name)) |ty| {
                if (ty != .dir) return error.InvalidFromSourceType;
            }
        },
        .expr => |e| {
            const ty = try inferExprType(allocator, scope, e);
            switch (ty) {
                .seq => |item| {
                    const want: TypeInfo = switch (kind) {
                        .string => .string,
                        .file => .file,
                        .dir => .dir,
                        .hash => .hash,
                    };
                    if (item.* != .unknown and !sameType(item.*, want))
                        return fail(e.span, error.InvalidFromSourceType);
                },
                else => if (!scalarSourceTypeAllowed(ty))
                    return fail(e.span, error.InvalidFromSourceType),
            }
        },
    }
}

fn validateClause(
    allocator: std.mem.Allocator,
    scope: *std.StringHashMapUnmanaged(TypeInfo),
    clause: *const plan.Clause,
) LowerError!void {
    switch (clause.*) {
        .where => |w| {
            const ty = try inferExprType(allocator, scope, w.pred);
            try asPredicateType(w.pred, ty);
            try validateClause(allocator, scope, w.then);
        },
        .from => |f| {
            try validateSource(allocator, scope, f.kind, f.source);
            var next = try cloneScope(allocator, scope);
            defer next.deinit(allocator);
            try next.put(allocator, f.range, switch (f.kind) {
                .string => .string,
                .file => .file,
                .dir => .dir,
                .hash => .hash,
            });
            try validateClause(allocator, &next, f.then);
        },
        .let => |l| {
            const ty = try inferExprType(allocator, scope, l.expr);
            var next = try cloneScope(allocator, scope);
            defer next.deinit(allocator);
            try next.put(allocator, l.name, ty);
            try validateClause(allocator, &next, l.then);
        },
        .join => |j| {
            try validateSource(allocator, scope, j.kind, j.source);
            var with_join = try cloneScope(allocator, scope);
            defer with_join.deinit(allocator);
            const j_ty: TypeInfo = switch (j.kind) {
                .string => .string,
                .file => .file,
                .dir => .dir,
                .hash => .hash,
            };
            try with_join.put(allocator, j.range, j_ty);
            const outer_ty = try inferExprType(allocator, &with_join, j.outer_key);
            const inner_ty = try inferExprType(allocator, &with_join, j.inner_key);
            const outer_scalar = try scalarCompareType(j.outer_key, outer_ty);
            const inner_scalar = try scalarCompareType(j.inner_key, inner_ty);
            if (outer_scalar != .unknown and inner_scalar != .unknown and !sameType(outer_scalar, inner_scalar))
                return fail(j.outer_key.span, error.TypeMismatch);
            if (outer_scalar != .unknown and !comparableType(outer_scalar))
                return fail(j.outer_key.span, error.TypeMismatch);
            if (inner_scalar != .unknown and !comparableType(inner_scalar))
                return fail(j.inner_key.span, error.TypeMismatch);

            if (j.group_into) |name| {
                var next = try cloneScope(allocator, scope);
                defer next.deinit(allocator);
                try next.put(allocator, name, try wrapSeq(allocator, j_ty));
                try validateClause(allocator, &next, j.then);
            } else {
                try validateClause(allocator, &with_join, j.then);
            }
        },
        .order_by => |o| {
            for (o.keys) |k| {
                const key_ty = try inferExprType(allocator, scope, k.expr);
                const scalar = try scalarCompareType(k.expr, key_ty);
                if (scalar != .unknown and !comparableType(scalar))
                    return fail(k.expr.span, error.TypeMismatch);
            }
            try validateClause(allocator, scope, o.then);
        },
        .group_by => |g| {
            const proj_ty = try inferExprType(allocator, scope, g.proj);
            const key_ty = try inferExprType(allocator, scope, g.key);
            const key_scalar = try scalarCompareType(g.key, key_ty);
            if (key_scalar != .unknown and !comparableType(key_scalar))
                return fail(g.key.span, error.TypeMismatch);
            if (g.into) |into| {
                var next = try cloneScope(allocator, scope);
                defer next.deinit(allocator);
                try next.put(allocator, into.name, try groupRecordType(allocator, key_scalar, proj_ty));
                try validateClause(allocator, &next, into.body);
            }
        },
        .select => |s| {
            const ty = try inferExprType(allocator, scope, s.expr);
            if (s.into) |into| {
                var next = try cloneScope(allocator, scope);
                defer next.deinit(allocator);
                try next.put(allocator, into.name, ty);
                try validateClause(allocator, &next, into.body);
            }
        },
    }
}

fn lowerQueryWithScope(
    allocator: std.mem.Allocator,
    root: *const c.fend_node_t,
    outer_scope: ?*const std.StringHashMapUnmanaged(TypeInfo),
) LowerError!plan.QueryPlan {
    if (root.type != c.node_type_query or root.left == null or root.right == null) return error.InvalidAst;
    const from_node: *c.fend_node_t = root.left.?;
    if (from_node.type != c.node_type_from or from_node.left == null or from_node.right == null)
        return error.InvalidAst;

    const decl = from_node.left.?;
    const source = from_node.right.?;
    const kind = try lowerType(decl);
    const root_from = try allocator.create(plan.From);
    root_from.* = .{
        .kind = kind,
        .range = try lowerName(allocator, decl),
        .source = try lowerSourceExpr(allocator, kind, source),
        .then = try lowerBody(allocator, root.right.?),
    };
    var scope: std.StringHashMapUnmanaged(TypeInfo) = if (outer_scope) |s| try cloneScope(allocator, s) else .empty;
    defer scope.deinit(allocator);
    const root_ty: TypeInfo = switch (kind) {
        .string => .string,
        .file => .file,
        .dir => .dir,
        .hash => .hash,
    };
    try validateSource(allocator, &scope, kind, root_from.source);
    try scope.put(allocator, root_from.range, root_ty);
    try validateClause(allocator, &scope, root_from.then);
    return .{ .root = root_from };
}

pub fn lowerQuery(allocator: std.mem.Allocator, root: *const c.fend_node_t) LowerError!plan.QueryPlan {
    return lowerQueryWithScope(allocator, root, null);
}

pub fn lowerQueryInEnv(
    allocator: std.mem.Allocator,
    root: *const c.fend_node_t,
    env: *const value.Env,
) LowerError!plan.QueryPlan {
    var scope = try scopeFromEnv(allocator, env);
    defer scope.deinit(allocator);
    return lowerQueryWithScope(allocator, root, &scope);
}
