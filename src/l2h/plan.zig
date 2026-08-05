//! Query plan IR — tree of From / Clause (docs/l2h-semantics.md §9).

const std = @import("std");
const expr = @import("expr.zig");

pub const SourceKind = enum { string, file, dir, hash };

pub const OrderKey = struct {
    expr: *expr.Expr,
    descending: bool = false,
};

pub const Into = struct {
    name: []const u8,
    /// Continuation body, or `null` for script-level bind (`select … into id;`).
    body: ?*Clause = null,
};

pub const Select = struct {
    expr: *expr.Expr,
    into: ?Into = null,
};

pub const Join = struct {
    kind: SourceKind,
    range: []const u8,
    /// Path, string payload, digest, Dir (→ file listing), or Seq source expression.
    source: *expr.Expr,
    outer_key: *expr.Expr,
    inner_key: *expr.Expr,
    group_into: ?[]const u8 = null,
    then: *Clause,
};

pub const From = struct {
    kind: SourceKind,
    range: []const u8,
    /// Path, string payload, digest, Dir (→ file listing), or Seq source expression.
    source: *expr.Expr,
    then: *Clause,
};

pub const Clause = union(enum) {
    from: *From,
    let: struct { name: []const u8, expr: *expr.Expr, then: *Clause },
    where: struct { pred: *expr.Expr, then: *Clause },
    join: *Join,
    order_by: struct { keys: []OrderKey, then: *Clause },
    /// `group proj by key` — yields Record `{ key, items }` per group (semantics §6).
    /// Terminal (into=null) sinks those records; with into, each group binds to `into.name`.
    group_by: struct {
        proj: *expr.Expr,
        key: *expr.Expr,
        into: ?Into = null,
    },
    select: *Select,

    /// Whether this clause chain hits `orderby` / `groupby` before a terminal select.
    /// Those operators need all rows materialized (stream barrier).
    pub fn hasBarrier(self: *const Clause) bool {
        var c: *const Clause = self;
        while (true) {
            switch (c.*) {
                .order_by, .group_by => return true,
                .where => |w| c = w.then,
                .let => |l| c = l.then,
                .from => |f| c = f.then,
                .join => |j| c = j.then,
                .select => |s| {
                    if (s.into) |into| {
                        if (into.body) |b| c = b else return false;
                    } else return false;
                },
            }
        }
    }
};

test "Clause.hasBarrier is false for terminal select" {
    // Arrange
    var lit: expr.Expr = .{ .kind = .{ .int_lit = 1 } };
    var sel: Select = .{ .expr = &lit };
    var terminal: Clause = .{ .select = &sel };

    // Act
    const got = terminal.hasBarrier();

    // Assert
    try std.testing.expect(!got);
}

test "Clause.hasBarrier is true for orderby" {
    // Arrange
    var lit: expr.Expr = .{ .kind = .{ .int_lit = 1 } };
    var sel: Select = .{ .expr = &lit };
    var terminal: Clause = .{ .select = &sel };
    var ordered: Clause = .{ .order_by = .{ .keys = &.{}, .then = &terminal } };

    // Act
    const got = ordered.hasBarrier();

    // Assert
    try std.testing.expect(got);
}

test "Clause.hasBarrier is true for groupby" {
    // Arrange
    var lit: expr.Expr = .{ .kind = .{ .int_lit = 1 } };
    var grouped: Clause = .{ .group_by = .{ .proj = &lit, .key = &lit } };

    // Act
    const got = grouped.hasBarrier();

    // Assert
    try std.testing.expect(got);
}

test "Clause.hasBarrier follows where to a barrier" {
    // Arrange
    var lit: expr.Expr = .{ .kind = .{ .int_lit = 1 } };
    var sel: Select = .{ .expr = &lit };
    var terminal: Clause = .{ .select = &sel };
    var ordered: Clause = .{ .order_by = .{ .keys = &.{}, .then = &terminal } };
    var where_clause: Clause = .{ .where = .{ .pred = &lit, .then = &ordered } };

    // Act
    const got = where_clause.hasBarrier();

    // Assert
    try std.testing.expect(got);
}

test "Clause.hasBarrier follows where to a terminal select" {
    // Arrange
    var lit: expr.Expr = .{ .kind = .{ .int_lit = 1 } };
    var sel: Select = .{ .expr = &lit };
    var terminal: Clause = .{ .select = &sel };
    var where_clause: Clause = .{ .where = .{ .pred = &lit, .then = &terminal } };

    // Act
    const got = where_clause.hasBarrier();

    // Assert
    try std.testing.expect(!got);
}

test "Clause.hasBarrier follows select into without barrier" {
    // Arrange
    var lit: expr.Expr = .{ .kind = .{ .int_lit = 1 } };
    var sel: Select = .{ .expr = &lit };
    var into_body: Clause = .{ .select = &sel };
    var sel_into: Select = .{ .expr = &lit, .into = .{ .name = "x", .body = &into_body } };
    var with_into: Clause = .{ .select = &sel_into };

    // Act
    const got = with_into.hasBarrier();

    // Assert
    try std.testing.expect(!got);
}

test "Clause.hasBarrier follows select into to a barrier" {
    // Arrange
    var lit: expr.Expr = .{ .kind = .{ .int_lit = 1 } };
    var sel: Select = .{ .expr = &lit };
    var terminal: Clause = .{ .select = &sel };
    var into_order: Clause = .{ .order_by = .{ .keys = &.{}, .then = &terminal } };
    var sel_into: Select = .{ .expr = &lit, .into = .{ .name = "x", .body = &into_order } };
    var with_into: Clause = .{ .select = &sel_into };

    // Act
    const got = with_into.hasBarrier();

    // Assert
    try std.testing.expect(got);
}
