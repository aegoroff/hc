//! Query plan IR — tree of From / Clause (docs/l2h-semantics.md §9).

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
    /// `group proj by key` — yields Record `{ key, items }` per group (semantics §6.6).
    /// Bare terminal (into=null) is not a printable sink: `items` is a `Seq`, so §7
    /// raises TypeMismatch; with into, each group binds to `into.name` (§6.6 / §6.7 / §10).
    group_by: struct {
        proj: *expr.Expr,
        key: *expr.Expr,
        into: ?Into = null,
    },
    select: *Select,
};

/// Name bound in the script env by a terminal `… into id;` (no continuation
/// body), or `null` when the query prints instead.
pub fn scriptBindName(root: *const From) ?[]const u8 {
    var clause: *const Clause = root.then;
    while (true) {
        const into: Into = switch (clause.*) {
            .from => |f| {
                clause = f.then;
                continue;
            },
            .let => |l| {
                clause = l.then;
                continue;
            },
            .where => |w| {
                clause = w.then;
                continue;
            },
            .join => |j| {
                clause = j.then;
                continue;
            },
            .order_by => |o| {
                clause = o.then;
                continue;
            },
            .group_by => |g| g.into orelse return null,
            .select => |s| s.into orelse return null,
        };
        clause = into.body orelse return into.name;
    }
}
