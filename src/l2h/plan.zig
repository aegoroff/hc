const expr = @import("expr.zig");

/// Query plan IR — tree of From / Clause (docs/l2h-semantics.md §11).

pub const SourceKind = enum { string, file, dir, hash };

pub const SourceExpr = union(enum) {
    /// Path, string payload, or digest expression.
    expr: *expr.Expr,
    /// Flat regular files in a Dir bound to this range name.
    files_in_dir: []const u8,
};

pub const OrderKey = struct {
    expr: *expr.Expr,
    descending: bool = false,
};

pub const Into = struct {
    name: []const u8,
    body: *Clause,
};

pub const Select = struct {
    expr: *expr.Expr,
    into: ?Into = null,
};

pub const Join = struct {
    kind: SourceKind,
    range: []const u8,
    source: SourceExpr,
    outer_key: *expr.Expr,
    inner_key: *expr.Expr,
    group_into: ?[]const u8 = null,
    then: *Clause,
};

pub const From = struct {
    kind: SourceKind,
    range: []const u8,
    source: SourceExpr,
    then: *Clause,
};

pub const Clause = union(enum) {
    from: *From,
    let: struct { name: []const u8, expr: *expr.Expr, then: *Clause },
    where: struct { pred: *expr.Expr, then: *Clause },
    join: *Join,
    order_by: struct { keys: []OrderKey, then: *Clause },
    /// `group proj by key` — yields Record `{ key, items }` per group (semantics §6.6).
    /// Terminal (into=null) sinks those records; with into, each group binds to `into.name`.
    group_by: struct {
        proj: *expr.Expr,
        key: *expr.Expr,
        into: ?Into = null,
    },
    select: *Select,
};

pub const QueryPlan = struct {
    root: *From,
};
