const std = @import("std");
const c = @import("c");

/// Expression IR (see docs/l2h-semantics.md §9). Separate from query plan operators.

pub const UnaryOp = enum { not_ };

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
    /// Nested query kept as parser AST and compiled on demand.
    query_ast: *const c.fend_node_t,
    name: []const u8,
    prop: struct { recv: *Expr, prop: []const u8 },
    unary: struct { op: UnaryOp, arg: *Expr },
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
