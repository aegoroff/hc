//! Expression IR (see docs/l2h-semantics.md §9). Separate from query plan operators.

const std = @import("std");
const c = @import("c");
const method = @import("method.zig");
const plan = @import("plan.zig");
const props = @import("props.zig");

pub const BinaryOp = enum {
    eq,
    neq,
    gt,
    ge,
    lt,
    le,
    match,
    not_match,
    and_,
    or_,
};

/// 1-based source range from the parser (`fend_node_t.loc`). Unset when all zero.
pub const Span = struct {
    first_line: c_int = 0,
    first_column: c_int = 0,
    last_line: c_int = 0,
    last_column: c_int = 0,

    pub fn fromNode(n: *const c.fend_node_t) Span {
        return .{
            .first_line = n.loc.first_line,
            .first_column = n.loc.first_column,
            .last_line = n.loc.last_line,
            .last_column = n.loc.last_column,
        };
    }

    pub fn isSet(self: Span) bool {
        return self.first_line > 0;
    }
};

pub const Kind = union(enum) {
    string_lit: []const u8,
    int_lit: i64,
    bool_lit: bool,
    /// Nested query compiled to a `From` plan in `compileExpr` (typed in `inferExprType`).
    nested_query: *plan.From,
    name: []const u8,
    /// Property access. `access` is filled during typecheck for range-kind builtins;
    /// null means record field or recv type was unknown at compile time.
    prop: struct { recv: *Expr, prop: []const u8, access: ?props.Access = null },
    /// Method call: `recv.name(args…)`. `kind` is resolved in `compileExpr`.
    method: struct { recv: *Expr, name: []const u8, args: []const *Expr, kind: method.Kind },
    /// Logical not (`not pred`).
    not: *Expr,
    binary: struct { op: BinaryOp, left: *Expr, right: *Expr },
    record: []RecordFieldExpr,
};

pub const Expr = struct {
    span: Span = .{},
    kind: Kind,
};

pub const RecordFieldExpr = struct {
    name: []const u8,
    expr: *Expr,
};

/// Derive auto-name for a record field expression, or error if not name/prop.
pub fn autoFieldName(e: *const Expr) error{InvalidRecordField}![]const u8 {
    return switch (e.kind) {
        .name => |n| n,
        .prop => |p| p.prop,
        else => error.InvalidRecordField,
    };
}

test "autoFieldName accepts name and prop only" {
    // Arrange
    var id: Expr = .{ .kind = .{ .name = "f" } };
    var prop_expr: Expr = .{ .kind = .{ .prop = .{ .recv = &id, .prop = "md5" } } };
    var lit: Expr = .{ .kind = .{ .int_lit = 1 } };

    // Act
    const n1 = try autoFieldName(&id);
    const n2 = try autoFieldName(&prop_expr);

    // Assert
    try std.testing.expectEqualStrings("f", n1);
    try std.testing.expectEqualStrings("md5", n2);
    try std.testing.expectError(error.InvalidRecordField, autoFieldName(&lit));
}
