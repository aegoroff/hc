const std = @import("std");
const c = @import("c");
const lib = @import("lib");
const proc = @import("processor.zig");

// Zig port of src/l2h/backend.c.
//
// Lowers the AST built by frontend.zig into a flat list of triples (a small
// stack-machine program) by walking the tree in postorder, then hands the
// program to the processor for execution. The triple encoding is preserved
// verbatim from the C original so observable behavior matches l2h.

pub const Opcode = enum(u8) {
    from = 0,
    def = 1,
    let_ = 2,
    select = 3,
    call = 4,
    property = 5,
    type_ = 6,
    usage = 7,
    integer = 8,
    string = 9,
    and_rel = 10,
    or_rel = 11,
    not_rel = 12,
    relation = 13,
    query_continuation = 14,
    into = 15,
};

pub const OpValue = extern union {
    type: c.type_def_t,
    number: c_longlong,
    string: [*c]u8,
    relation_op: c.cond_op_t,
};

pub const Triple = extern struct {
    code: Opcode,
    op1: ?*OpValue = null,
    op2: ?*OpValue = null,
};

var alloc: std.mem.Allocator = undefined;
var instructions: std.ArrayListUnmanaged(*Triple) = .empty;

fn span(s: [*c]u8) []const u8 {
    return std.mem.span(@as([*:0]u8, @ptrCast(s)));
}

fn isCustomType(t: c.type_def_t) bool {
    return @as(c_int, @intCast(t)) == c.type_def_custom;
}

// --- op value constructors (port of prbend_create_string / _number) -------

fn createStringOp(node: *c.fend_node_t) ?*OpValue {
    // lib_trim(apr_psprintf(pool, "%s", node->value.string), "'\"")
    const result = alloc.create(OpValue) catch return null;
    result.* = .{ .string = unquoteDup(span(node.value.string)) };
    return result;
}

fn createNumberOp(node: *c.fend_node_t) ?*OpValue {
    const result = alloc.create(OpValue) catch return null;
    result.* = .{ .number = node.value.number };
    return result;
}

/// Duplicates `s` into the backend arena with surrounding quote characters
/// stripped (mirrors lib_trim with seps = "'\""), NUL-terminated for C interop.
fn unquoteDup(s_in: []const u8) [*c]u8 {
    var s = s_in;
    while (s.len > 0 and (s[0] == '\'' or s[0] == '"')) s = s[1..];
    while (s.len > 0 and (s[s.len - 1] == '\'' or s[s.len - 1] == '"')) s = s[0 .. s.len - 1];
    const mem = alloc.allocSentinel(u8, s.len, 0) catch return @constCast("".ptr);
    @memcpy(mem[0..s.len], s);
    return mem.ptr;
}

// --- triple constructors (port of prbend_create_*_triple) -----------------

fn newTriple(code: Opcode) ?*Triple {
    const t = alloc.create(Triple) catch return null;
    t.* = .{ .code = code };
    return t;
}

fn createFromTriple() ?*Triple {
    const count: c_longlong = @intCast(instructions.items.len);
    const t = newTriple(.from) orelse return null;
    const op1 = alloc.create(OpValue) catch return null;
    op1.* = .{ .number = count - 2 };
    const op2 = alloc.create(OpValue) catch return null;
    op2.* = .{ .number = count - 1 };
    t.op1 = op1;
    t.op2 = op2;
    return t;
}

fn createRelTriple(code: Opcode) ?*Triple {
    return newTriple(code);
}

fn createRelationTriple(node: *c.fend_node_t) ?*Triple {
    const t = newTriple(.relation) orelse return null;
    const op1 = alloc.create(OpValue) catch return null;
    op1.* = .{ .relation_op = node.value.relation_op };
    t.op1 = op1;
    return t;
}

fn createInternalTypeTriple(node: *c.fend_node_t) ?*Triple {
    const t = newTriple(.type_) orelse return null;
    const op1 = alloc.create(OpValue) catch return null;
    op1.* = .{ .type = node.value.type };
    t.op1 = op1;
    if (isCustomType(node.value.type)) {
        // node->left holds the type-name string literal.
        if (node.left) |lp| {
            const left: *c.fend_node_t = lp;
            const op2 = alloc.create(OpValue) catch return null;
            op2.* = .{ .string = unquoteDup(span(left.value.string)) };
            t.op2 = op2;
        }
    }
    return t;
}

fn createStringLiteralTriple(node: *c.fend_node_t) ?*Triple {
    const t = newTriple(.string) orelse return null;
    t.op1 = createStringOp(node);
    return t;
}

fn createNumericLiteralTriple(node: *c.fend_node_t) ?*Triple {
    const t = newTriple(.integer) orelse return null;
    t.op1 = createNumberOp(node);
    return t;
}

fn createIdentifierTriple(node: *c.fend_node_t) ?*Triple {
    const t = alloc.create(Triple) catch return null;
    t.* = .{ .code = .usage };

    if (instructions.items.len > 0) {
        const prev = instructions.items[instructions.items.len - 1];
        switch (prev.code) {
            .type_ => {
                _ = instructions.pop(); // consume the type triple
                t.code = .def;
                if (prev.op1) |p1| {
                    if (isCustomType(p1.type)) {
                        t.op1 = prev.op2;
                    } else {
                        t.op1 = prev.op1;
                    }
                }
            },
            .select => {
                t.code = .into;
                const op1 = alloc.create(OpValue) catch return null;
                op1.* = .{ .number = @intCast(instructions.items.len - 1) };
                t.op1 = op1;
            },
            else => {
                t.code = .usage;
            },
        }
    }
    t.op2 = createStringOp(node);
    return t;
}

fn createPropertyTriple(node: *c.fend_node_t) ?*Triple {
    const prev = instructions.pop() orelse return null; // the identifier triple this property qualifies
    const t = newTriple(.property) orelse return null;
    t.op1 = prev.op2;
    t.op2 = createStringOp(node);
    return t;
}

fn createLetTriple() ?*Triple {
    return newTriple(.let_);
}

fn createSelectTriple() ?*Triple {
    return newTriple(.select);
}

fn createMethodCallTriple(node: *c.fend_node_t) ?*Triple {
    const t = newTriple(.call) orelse return null;
    if (node.left == null and node.right == null) {
        // method without parameters: pop the preceding identifier triple
        if (instructions.pop()) |prev| {
            t.op1 = prev.op2;
        } else {
            const op1 = alloc.create(OpValue) catch return null;
            op1.* = .{ .string = @constCast("".ptr) };
            t.op1 = op1;
        }
    } else {
        const op1 = alloc.create(OpValue) catch return null;
        op1.* = .{ .string = @constCast("".ptr) };
        t.op1 = op1;
    }
    t.op2 = createStringOp(node);
    return t;
}

// --- emit + lifecycle (port of bend_init / bend_emit / bend_complete) -----

pub fn init(arena: std.mem.Allocator) void {
    alloc = arena;
    instructions = .empty;
    proc.init(arena);
}

pub fn complete() void {
    proc.run(instructions.items);
    proc.complete();
}

pub fn emit(node: *c.fend_node_t) void {
    const t: ?*Triple = switch (node.type) {
        @intCast(c.node_type_from) => createFromTriple(),
        @intCast(c.node_type_not_rel) => createRelTriple(.not_rel),
        @intCast(c.node_type_and_rel) => createRelTriple(.and_rel),
        @intCast(c.node_type_or_rel) => createRelTriple(.or_rel),
        @intCast(c.node_type_relation) => createRelationTriple(node),
        @intCast(c.node_type_internal_type) => createInternalTypeTriple(node),
        @intCast(c.node_type_string_literal) => createStringLiteralTriple(node),
        @intCast(c.node_type_numeric_literal) => createNumericLiteralTriple(node),
        @intCast(c.node_type_identifier) => createIdentifierTriple(node),
        @intCast(c.node_type_property) => createPropertyTriple(node),
        @intCast(c.node_type_let) => createLetTriple(),
        @intCast(c.node_type_select) => createSelectTriple(),
        @intCast(c.node_type_method_call) => createMethodCallTriple(node),
        else => null,
    };

    if (t) |instruction| {
        // Custom hash-type declarations carry a trailing type-name triple that
        // must be dropped from the stack (see backend.c bend_emit).
        if (instruction.code == .type_ and instruction.op1 != null and
            isCustomType(instruction.op1.?.type))
        {
            if (instructions.items.len > 0) _ = instructions.pop();
        }
        instructions.append(alloc, instruction) catch {};
    }
}

/// Iterative postorder traversal (port of treeutil.c tree_postorder). Visits
/// each node after its subtrees, invoking `action` exactly once per node.
pub fn postorder(root_arg: ?*c.fend_node_t, action: *const fn (*c.fend_node_t) void) void {
    var root = root_arg orelse return;
    var stack: std.ArrayListUnmanaged(*c.fend_node_t) = .empty;
    defer stack.deinit(alloc);
    stack.append(alloc, root) catch return;

    while (stack.items.len > 0) {
        const next = stack.items[stack.items.len - 1];
        const finished_subtrees = (next.right == root or next.left == root);
        const is_leaf = (next.left == null and next.right == null);
        if (finished_subtrees or is_leaf) {
            _ = stack.pop();
            action(next);
            root = next;
        } else {
            if (next.right) |r| stack.append(alloc, r) catch {};
            if (next.left) |l| stack.append(alloc, l) catch {};
        }
    }
}

/// Drives the full backend for one completed query AST: lower to triples via
/// postorder emit, then execute the program.
pub fn processQuery(root: ?*c.fend_node_t, arena: std.mem.Allocator) void {
    init(arena);
    postorder(root, emit);
    complete();
}
