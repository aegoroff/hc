//! Compile bison AST nodes into the l2h query plan / expression IR.

const std = @import("std");
const c = @import("c");
const diag = @import("diag.zig");
const expr = @import("expr.zig");
const method = @import("method.zig");
const plan = @import("plan.zig");
const props = @import("props.zig");
const string_lit = @import("string_lit.zig");

pub const Error = error{
    InvalidAst,
    UndefinedName,
    InvalidFromSourceType,
    InvalidProperty,
    TypeMismatch,
    DuplicateField,
    UnsupportedNode,
    UnknownMethod,
    InvalidMethodArity,
    InvalidMethodReceiver,
    InvalidMethodFields,
    InvalidRecordField,
    QueryTooDeep,
    InvalidStringEscape,
};

const CompileError = Error || std.mem.Allocator.Error;

/// Backstop against stack overflow on adversarial input: the compilation/type and
/// eval passes walk the bison AST recursively, and a deeply nested query
/// (`from … let x = from … let y = from …`) can grow the stack without bound.
/// 64 levels is far beyond any meaningful query; beyond it is treated as an
/// error rather than a crash. Compilation and evaluation share this single limit
/// so the same depth budget applies at compile-analysis and run time.
pub const MAX_QUERY_DEPTH: u32 = 64;

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
    record: []const RecordFieldType,
    seq: *const TypeInfo,
};

const RecordFieldType = struct {
    name: []const u8,
    ty: *const TypeInfo,
};

fn typeOfKind(kind: plan.SourceKind) TypeInfo {
    return switch (kind) {
        .string => .string,
        .file => .file,
        .dir => .dir,
        .hash => .hash,
    };
}

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

fn compileType(node: *const c.fend_node_t) Error!plan.SourceKind {
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

fn compileName(allocator: std.mem.Allocator, node: *const c.fend_node_t) ![]const u8 {
    if (node.type != c.node_type_identifier) return error.InvalidAst;
    return try allocator.dupe(u8, span(node.value.string));
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

fn compileExprList(allocator: std.mem.Allocator, node: ?*c.fend_node_t, depth: u32) CompileError![]const *expr.Expr {
    if (node == null) return &.{};
    var items: std.ArrayList(*c.fend_node_t) = .empty;
    defer items.deinit(allocator);
    try flattenEnum(allocator, node.?, &items);
    const args = try allocator.alloc(*expr.Expr, items.items.len);
    for (items.items, 0..) |item, i| {
        args[i] = try compileExpr(allocator, item, depth);
    }
    return args;
}

fn isFormatScalarType(ty: TypeInfo) bool {
    return switch (ty) {
        .string, .int, .bool, .unknown => true,
        else => false,
    };
}

/// Values allowed in `json` / `jsonPretty`: scalars, nested records, and sequences of those.
fn isJsonValueType(ty: TypeInfo) bool {
    return switch (ty) {
        .string, .int, .bool, .unknown => true,
        .record => |fields| blk: {
            for (fields) |field| {
                if (!isJsonValueType(field.ty.*)) break :blk false;
            }
            break :blk true;
        },
        .seq => |item| isJsonValueType(item.*),
        else => false,
    };
}

fn compileRecordFields(allocator: std.mem.Allocator, node: *const c.fend_node_t, depth: u32) CompileError![]expr.RecordFieldExpr {
    var items: std.ArrayList(*c.fend_node_t) = .empty;
    defer items.deinit(allocator);
    try flattenEnum(allocator, @constCast(node), &items);

    const fields = try allocator.alloc(expr.RecordFieldExpr, items.items.len);
    for (items.items, 0..) |item, i| {
        if (item.type == c.node_type_let and item.left != null and item.right != null) {
            fields[i] = .{
                .name = try compileName(allocator, item.left.?),
                .expr = try compileExpr(allocator, item.right.?, depth),
            };
        } else {
            const field_expr = try compileExpr(allocator, item, depth);
            fields[i] = .{
                .name = try allocator.dupe(u8, expr.autoFieldName(field_expr) catch {
                    return fail(field_expr.span, error.InvalidRecordField);
                }),
                .expr = field_expr,
            };
        }
    }
    return fields;
}

pub fn compileExpr(allocator: std.mem.Allocator, node: *const c.fend_node_t, depth: u32) CompileError!*expr.Expr {
    const out = try allocator.create(expr.Expr);
    const sp = expr.Span.fromNode(node);
    switch (node.type) {
        c.node_type_unary_expression => {
            const inner = try compileExpr(allocator, node.left.?, depth);
            if (node.right != null) {
                const rhs: *c.fend_node_t = node.right.?;
                if (rhs.type == c.node_type_property) {
                    out.* = .{
                        .span = sp,
                        .kind = .{
                            .prop = .{
                                .recv = inner,
                                .prop = try allocator.dupe(u8, span(rhs.value.string)),
                            },
                        },
                    };
                    return out;
                }
                if (rhs.type == c.node_type_method_call) {
                    out.* = .{
                        .span = sp,
                        .kind = .{
                            .method = .{
                                .recv = inner,
                                .name = try allocator.dupe(u8, span(rhs.value.string)),
                                .args = try compileExprList(allocator, rhs.left, depth),
                            },
                        },
                    };
                    return out;
                }
            }
            // Grammar wraps literals/ids in unary nodes and FLOCs the wrapper;
            // the child often has an unset loc — keep the wrapper span.
            if (!inner.span.isSet() and sp.isSet()) inner.span = sp;
            allocator.destroy(out);
            return inner;
        },
        c.node_type_identifier => out.* = .{
            .span = sp,
            .kind = .{ .name = try allocator.dupe(u8, span(node.value.string)) },
        },
        c.node_type_string_literal => out.* = .{
            .span = sp,
            .kind = .{
                .string_lit = string_lit.decode(allocator, span(node.value.string)) catch |err| switch (err) {
                    error.InvalidStringEscape => return fail(sp, error.InvalidStringEscape),
                    else => |e| return e,
                },
            },
        },
        c.node_type_numeric_literal => out.* = .{
            .span = sp,
            .kind = .{ .int_lit = node.value.number },
        },
        c.node_type_boolean_literal => out.* = .{
            .span = sp,
            .kind = .{ .bool_lit = node.value.number != 0 },
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
                    .left = try compileExpr(allocator, node.left.?, depth),
                    .right = try compileExpr(allocator, node.right.?, depth),
                },
            },
        },
        c.node_type_and_rel => out.* = .{
            .span = sp,
            .kind = .{
                .binary = .{
                    .op = .and_,
                    .left = try compileExpr(allocator, node.left.?, depth),
                    .right = try compileExpr(allocator, node.right.?, depth),
                },
            },
        },
        c.node_type_or_rel => out.* = .{
            .span = sp,
            .kind = .{
                .binary = .{
                    .op = .or_,
                    .left = try compileExpr(allocator, node.left.?, depth),
                    .right = try compileExpr(allocator, node.right.?, depth),
                },
            },
        },
        c.node_type_not_rel => out.* = .{
            .span = sp,
            .kind = .{ .not = try compileExpr(allocator, node.left.?, depth) },
        },
        c.node_type_enum, c.node_type_object => out.* = .{
            .span = sp,
            .kind = .{
                .record = try compileRecordFields(allocator, if (node.type == c.node_type_object) node.left.? else node, depth),
            },
        },
        c.node_type_query => out.* = .{
            .span = sp,
            .kind = .{ .nested_query = try compileNestedQuery(allocator, node, depth + 1) },
        },
        else => return failNode(node, error.UnsupportedNode),
    }
    return out;
}

fn compileOrderKeys(allocator: std.mem.Allocator, node: *const c.fend_node_t, depth: u32) CompileError![]plan.OrderKey {
    var items: std.ArrayList(*c.fend_node_t) = .empty;
    defer items.deinit(allocator);
    try flattenEnum(allocator, @constCast(node), &items);

    const keys = try allocator.alloc(plan.OrderKey, items.items.len);
    for (items.items, 0..) |item, i| {
        if (item.type != c.node_type_ordering or item.left == null) return error.InvalidAst;
        keys[i] = .{
            .expr = try compileExpr(allocator, item.left.?, depth),
            .descending = item.value.ordering == c.ordering_desc,
        };
    }
    return keys;
}

fn compileClauseNode(
    allocator: std.mem.Allocator,
    node: *const c.fend_node_t,
    then: *plan.Clause,
    depth: u32,
) CompileError!*plan.Clause {
    const out = try allocator.create(plan.Clause);
    switch (node.type) {
        c.node_type_from => {
            const decl = node.left orelse return error.InvalidAst;
            const src = node.right orelse return error.InvalidAst;
            const kind = try compileType(decl);
            const from = try allocator.create(plan.From);
            from.* = .{
                .kind = kind,
                .range = try compileName(allocator, decl),
                .source = try compileExpr(allocator, src, depth),
                .then = then,
            };
            out.* = .{ .from = from };
        },
        c.node_type_where => out.* = .{
            .where = .{
                .pred = try compileExpr(allocator, node.left.?, depth),
                .then = then,
            },
        },
        c.node_type_let => out.* = .{
            .let = .{
                .name = try compileName(allocator, node.left.?),
                .expr = try compileExpr(allocator, node.right.?, depth),
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
            const kind = try compileType(decl);
            const join = try allocator.create(plan.Join);
            join.* = .{
                .kind = kind,
                .range = try compileName(allocator, decl),
                .source = try compileExpr(allocator, in_node.left.?, depth),
                .outer_key = try compileExpr(allocator, on_node.left.?, depth),
                .inner_key = try compileExpr(allocator, on_node.right.?, depth),
                .then = then,
            };
            out.* = .{ .join = join };
        },
        c.node_type_order_by => out.* = .{
            .order_by = .{
                .keys = try compileOrderKeys(allocator, node.left.?, depth),
                .then = then,
            },
        },
        else => return error.UnsupportedNode,
    }
    return out;
}

fn compileTerminalClause(
    allocator: std.mem.Allocator,
    terminal: *const c.fend_node_t,
    continuation: ?*c.fend_node_t,
    depth: u32,
) CompileError!*plan.Clause {
    const out = try allocator.create(plan.Clause);
    switch (terminal.type) {
        c.node_type_group => {
            var into: ?plan.Into = null;
            if (continuation) |cont| {
                if (cont.type != c.node_type_query_continuation or cont.left == null or cont.right == null)
                    return error.InvalidAst;
                into = .{
                    .name = try compileName(allocator, cont.left.?),
                    .body = try compileBody(allocator, cont.right.?, depth),
                };
            }
            out.* = .{
                .group_by = .{
                    .proj = try compileExpr(allocator, terminal.left.?, depth),
                    .key = try compileExpr(allocator, terminal.right.?, depth),
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
                    .name = try compileName(allocator, cont.left.?),
                    .body = try compileBody(allocator, cont.right.?, depth),
                };
            }
            sel.* = .{
                .expr = try compileExpr(allocator, terminal, depth),
                .into = into,
            };
            out.* = .{ .select = sel };
        },
    }
    return out;
}

fn compileBody(allocator: std.mem.Allocator, body: *const c.fend_node_t, depth: u32) CompileError!*plan.Clause {
    if (body.type != c.node_type_query_body or body.left == null) return error.InvalidAst;
    const select_wrap: *c.fend_node_t = body.left.?;
    if (select_wrap.type != c.node_type_select or select_wrap.right == null) return error.InvalidAst;

    var tail = try compileTerminalClause(allocator, select_wrap.right.?, body.right, depth);

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
            const wrapped = try compileClauseNode(allocator, join_clause, next_then, depth);
            if (wrapped.* != .join) return error.InvalidAst;
            wrapped.join.group_into = try compileName(allocator, current.left.?);
            tail = wrapped;
            continue;
        }

        tail = try compileClauseNode(allocator, current, next_then, depth);
    }
    return tail;
}

fn scalarSourceTypeAllowed(kind: plan.SourceKind, ty: TypeInfo) bool {
    // Singleton sources: string path/digest only (plus Dir→file walk). No cross-kind coercion.
    return switch (kind) {
        .string, .hash => switch (ty) {
            .unknown, .string => true,
            else => false,
        },
        .dir => switch (ty) {
            .unknown, .string => true,
            else => false,
        },
        .file => switch (ty) {
            .unknown, .string, .dir => true,
            else => false,
        },
    };
}

fn comparableType(ty: TypeInfo) bool {
    return switch (ty) {
        .int, .string, .bool => true,
        else => false,
    };
}

/// Nested query / Seq in predicates means existence (non-empty).
fn asPredicateType(e: *const expr.Expr, ty: TypeInfo) CompileError!void {
    switch (ty) {
        .bool, .seq, .unknown => {},
        else => return fail(e.span, error.TypeMismatch),
    }
}

/// Unwrap a singleton Seq to a scalar type for comparisons / method args.
/// When `allow_named_seq` is false (compare / orderby / join keys), only a
/// nested-query Seq may unwrap; named Seq is TypeMismatch.
fn scalarType(e: *const expr.Expr, ty: TypeInfo, allow_named_seq: bool) CompileError!TypeInfo {
    if (ty == .seq) {
        if (!allow_named_seq and e.kind != .nested_query) return fail(e.span, error.TypeMismatch);
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
) CompileError!TypeInfo {
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

fn inferExprType(
    allocator: std.mem.Allocator,
    scope: *const std.StringHashMapUnmanaged(TypeInfo),
    e: *expr.Expr,
    depth: u32,
) CompileError!TypeInfo {
    return switch (e.kind) {
        .nested_query => |q| blk: {
            // Plan already compiled in compileExpr; typecheck against this scope.
            break :blk try inferPlanResultType(allocator, scope, q, depth + 1);
        },
        .string_lit => .string,
        .int_lit => .int,
        .bool_lit => .bool,
        .name => |name| scope.get(name) orelse fail(e.span, error.UndefinedName),
        .not => |arg| blk: {
            const arg_ty = try inferExprType(allocator, scope, arg, depth);
            try asPredicateType(arg, arg_ty);
            break :blk .bool;
        },
        .binary => |b| blk: {
            const left_ty = try inferExprType(allocator, scope, b.left, depth);
            const right_ty = try inferExprType(allocator, scope, b.right, depth);
            switch (b.op) {
                .and_, .or_ => {
                    try asPredicateType(b.left, left_ty);
                    try asPredicateType(b.right, right_ty);
                },
                .match, .not_match => {
                    const l = try scalarType(b.left, left_ty, false);
                    const r = try scalarType(b.right, right_ty, false);
                    if (l != .string and l != .unknown) return fail(e.span, error.TypeMismatch);
                    if (r != .string and r != .unknown) return fail(e.span, error.TypeMismatch);
                },
                .gt, .ge, .lt, .le => {
                    const l = try scalarType(b.left, left_ty, false);
                    const r = try scalarType(b.right, right_ty, false);
                    if (l != .int and l != .unknown) return fail(e.span, error.TypeMismatch);
                    if (r != .int and r != .unknown) return fail(e.span, error.TypeMismatch);
                },
                .eq, .neq => {
                    const l = try scalarType(b.left, left_ty, false);
                    const r = try scalarType(b.right, right_ty, false);
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
                const field_ty = try inferExprType(allocator, scope, field.expr, depth);
                out[i] = .{
                    .name = field.name,
                    .ty = try cloneType(allocator, field_ty),
                };
            }
            break :blk .{ .record = out };
        },
        .prop => |p| blk: {
            const recv_ty = try inferExprType(allocator, scope, p.recv, depth);
            const access = switch (recv_ty) {
                .string => props.lookup(.string, p.prop),
                .file => props.lookup(.file, p.prop),
                .hash => props.lookup(.hash, p.prop),
                .dir => props.lookup(.dir, p.prop),
                .int, .bool, .seq => return fail(e.span, error.InvalidProperty),
                .record => |rec| break :blk recordFieldType(rec, p.prop) orelse return fail(e.span, error.InvalidProperty),
                .unknown => break :blk .unknown,
            };
            break :blk switch (access orelse return fail(e.span, error.InvalidProperty)) {
                .path, .name, .hash_algo => .string,
                .size, .offset, .limit => .int,
                .readable => .bool,
            };
        },
        .method => |m| blk: {
            const kind = method.lookup(m.name) orelse return fail(e.span, error.UnknownMethod);
            if (!method.arityOk(kind, m.args.len)) return fail(e.span, error.InvalidMethodArity);

            const recv_ty = try inferExprType(allocator, scope, m.recv, depth);
            switch (kind) {
                .formatter => |f| {
                    switch (recv_ty) {
                        .record => |fields| {
                            if (method.pairLabelField(f) != null) {
                                var names: [2][]const u8 = undefined;
                                if (fields.len > names.len) return fail(e.span, error.InvalidMethodFields);
                                for (fields, 0..) |field, i| names[i] = field.name;
                                method.validatePairFields(f, names[0..fields.len]) catch
                                    return fail(e.span, error.InvalidMethodFields);
                            }
                            for (fields) |field| {
                                const ok = if (method.allowsNestedValues(f))
                                    isJsonValueType(field.ty.*)
                                else
                                    isFormatScalarType(field.ty.*);
                                if (!ok) return fail(e.span, error.TypeMismatch);
                            }
                        },
                        .unknown => {},
                        else => return fail(e.span, error.InvalidMethodReceiver),
                    }
                    for (m.args) |arg| {
                        _ = try inferExprType(allocator, scope, arg, depth);
                    }
                },
                .hash_check => {
                    switch (recv_ty) {
                        .file, .string, .unknown => {},
                        else => return fail(e.span, error.InvalidMethodReceiver),
                    }
                    const arg_ty = try scalarType(m.args[0], try inferExprType(allocator, scope, m.args[0], depth), true);
                    if (arg_ty != .string and arg_ty != .unknown) return fail(e.span, error.TypeMismatch);
                },
                .dir_tree => {
                    switch (recv_ty) {
                        .dir, .unknown => {},
                        else => return fail(e.span, error.InvalidMethodReceiver),
                    }
                    if (m.args.len == 1) {
                        const arg_ty = try scalarType(m.args[0], try inferExprType(allocator, scope, m.args[0], depth), true);
                        if (arg_ty != .int and arg_ty != .unknown) return fail(e.span, error.TypeMismatch);
                    }
                },
                .dir_skip_errors => {
                    switch (recv_ty) {
                        .dir, .unknown => {},
                        else => return fail(e.span, error.InvalidMethodReceiver),
                    }
                },
                .file_offset, .file_limit => {
                    switch (recv_ty) {
                        .file, .unknown => {},
                        else => return fail(e.span, error.InvalidMethodReceiver),
                    }
                    const arg_ty = try scalarType(m.args[0], try inferExprType(allocator, scope, m.args[0], depth), true);
                    if (arg_ty != .int and arg_ty != .unknown) return fail(e.span, error.TypeMismatch);
                },
            }

            break :blk switch (kind) {
                .formatter => .string,
                .hash_check => .bool,
                .dir_tree, .dir_skip_errors => .dir,
                .file_offset, .file_limit => .file,
            };
        },
    };
}

/// Result type of an already-compiled nested query against `outer_scope`.
fn inferPlanResultType(
    allocator: std.mem.Allocator,
    outer_scope: *const std.StringHashMapUnmanaged(TypeInfo),
    q: *const plan.From,
    depth: u32,
) CompileError!TypeInfo {
    if (depth > MAX_QUERY_DEPTH) return error.QueryTooDeep;
    var scope = try cloneScope(allocator, outer_scope);
    defer scope.deinit(allocator);
    try validateSource(allocator, &scope, q.kind, q.source, depth);
    try scope.put(allocator, q.range, typeOfKind(q.kind));
    return validateClause(allocator, &scope, q.then, depth);
}

fn validateSource(
    allocator: std.mem.Allocator,
    scope: *const std.StringHashMapUnmanaged(TypeInfo),
    kind: plan.SourceKind,
    source: *expr.Expr,
    depth: u32,
) CompileError!void {
    const ty = try inferExprType(allocator, scope, source, depth);
    switch (ty) {
        .seq => |item| {
            const want = typeOfKind(kind);
            if (item.* != .unknown and !sameType(item.*, want))
                return fail(source.span, error.InvalidFromSourceType);
        },
        else => if (!scalarSourceTypeAllowed(kind, ty))
            return fail(source.span, error.InvalidFromSourceType),
    }
}

/// Walk a clause pipeline: check types and return the query result type
/// (`Seq(item)` for a terminal select/group, or the continuation body's type).
fn validateClause(
    allocator: std.mem.Allocator,
    scope: *std.StringHashMapUnmanaged(TypeInfo),
    clause: *const plan.Clause,
    depth: u32,
) CompileError!TypeInfo {
    switch (clause.*) {
        .where => |w| {
            const ty = try inferExprType(allocator, scope, w.pred, depth);
            try asPredicateType(w.pred, ty);
            return validateClause(allocator, scope, w.then, depth);
        },
        .from => |f| {
            try validateSource(allocator, scope, f.kind, f.source, depth);
            var next = try cloneScope(allocator, scope);
            defer next.deinit(allocator);
            try next.put(allocator, f.range, typeOfKind(f.kind));
            return validateClause(allocator, &next, f.then, depth);
        },
        .let => |l| {
            const ty = try inferExprType(allocator, scope, l.expr, depth);
            var next = try cloneScope(allocator, scope);
            defer next.deinit(allocator);
            try next.put(allocator, l.name, ty);
            return validateClause(allocator, &next, l.then, depth);
        },
        .join => |j| {
            try validateSource(allocator, scope, j.kind, j.source, depth);
            var with_join = try cloneScope(allocator, scope);
            defer with_join.deinit(allocator);
            const j_ty = typeOfKind(j.kind);
            try with_join.put(allocator, j.range, j_ty);
            // Outer key is evaluated in the outer env only (§6.4); do not see `j.range`.
            const outer_ty = try inferExprType(allocator, scope, j.outer_key, depth);
            const inner_ty = try inferExprType(allocator, &with_join, j.inner_key, depth);
            const outer_scalar = try scalarType(j.outer_key, outer_ty, false);
            const inner_scalar = try scalarType(j.inner_key, inner_ty, false);
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
                return validateClause(allocator, &next, j.then, depth);
            }
            return validateClause(allocator, &with_join, j.then, depth);
        },
        .order_by => |o| {
            for (o.keys) |k| {
                const key_ty = try inferExprType(allocator, scope, k.expr, depth);
                const scalar = try scalarType(k.expr, key_ty, false);
                if (scalar != .unknown and !comparableType(scalar))
                    return fail(k.expr.span, error.TypeMismatch);
            }
            return validateClause(allocator, scope, o.then, depth);
        },
        .group_by => |g| {
            const proj_ty = try inferExprType(allocator, scope, g.proj, depth);
            const key_ty = try inferExprType(allocator, scope, g.key, depth);
            const key_scalar = try scalarType(g.key, key_ty, false);
            if (key_scalar != .unknown and !comparableType(key_scalar))
                return fail(g.key.span, error.TypeMismatch);
            const rec_ty = try groupRecordType(allocator, key_scalar, proj_ty);
            if (g.into) |into| {
                // Continuation sees only `into.name` (matches runtime Env; §6.8).
                var next: std.StringHashMapUnmanaged(TypeInfo) = .empty;
                defer next.deinit(allocator);
                try next.put(allocator, into.name, rec_ty);
                return validateClause(allocator, &next, into.body, depth);
            }
            return wrapSeq(allocator, rec_ty);
        },
        .select => |s| {
            const ty = try inferExprType(allocator, scope, s.expr, depth);
            if (s.into) |into| {
                // Continuation sees only `into.name` (matches runtime Env; §6.8).
                var next: std.StringHashMapUnmanaged(TypeInfo) = .empty;
                defer next.deinit(allocator);
                try next.put(allocator, into.name, ty);
                return validateClause(allocator, &next, into.body, depth);
            }
            return wrapSeq(allocator, ty);
        },
    }
}

/// Compile a query AST into a `*From` plan without typechecking. Nested queries
/// are typed later in `inferExprType` against the enclosing scope.
fn compileNestedQuery(
    allocator: std.mem.Allocator,
    root: *const c.fend_node_t,
    depth: u32,
) CompileError!*plan.From {
    if (depth > MAX_QUERY_DEPTH) return error.QueryTooDeep;
    if (root.type != c.node_type_query or root.left == null or root.right == null) return error.InvalidAst;
    const from_node: *c.fend_node_t = root.left.?;
    if (from_node.type != c.node_type_from or from_node.left == null or from_node.right == null)
        return error.InvalidAst;

    const decl = from_node.left.?;
    const source = from_node.right.?;
    const kind = try compileType(decl);
    const root_from = try allocator.create(plan.From);
    root_from.* = .{
        .kind = kind,
        .range = try compileName(allocator, decl),
        .source = try compileExpr(allocator, source, depth),
        .then = try compileBody(allocator, root.right.?, depth),
    };
    return root_from;
}

fn compileQueryWithScope(
    allocator: std.mem.Allocator,
    root: *const c.fend_node_t,
    depth: u32,
) CompileError!*plan.From {
    // Single depth gate for every nesting level (compile + validate recurse
    // through here). Bounds the stack against adversarial queries.
    const from = try compileNestedQuery(allocator, root, depth);
    var scope: std.StringHashMapUnmanaged(TypeInfo) = .empty;
    defer scope.deinit(allocator);
    try validateSource(allocator, &scope, from.kind, from.source, depth);
    try scope.put(allocator, from.range, typeOfKind(from.kind));
    _ = try validateClause(allocator, &scope, from.then, depth);
    return from;
}

pub fn compileQuery(allocator: std.mem.Allocator, root: *const c.fend_node_t) CompileError!*plan.From {
    return try compileQueryWithScope(allocator, root, 0);
}
