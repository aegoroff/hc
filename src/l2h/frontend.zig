const std = @import("std");
const c = @import("c");
const state = @import("state.zig");
const diag = @import("diag.zig");

// Zig port of src/l2h/frontend.c.
//
// The bison/flex-generated parser (l2h.tab.c / l2h.flex.c, compiled into the
// l2h-c static lib) invokes the `fend_on_*` / `fend_query_*` callbacks below to
// build the AST. The grammar and lexer are fixed, so every export MUST keep its
// exact C name and signature. APR pools are replaced by ArenaAllocator + a
// std.StringHashMap identifier table. Continuation ids (`into`) are registered
// with a null type so they are considered defined for later clauses.

// --- C globals referenced by the generated parser -------------------------

/// Incremented by lyyerror/yyerror in l2h.tab.c. Must be a real C symbol.
pub export var fend_error_count: c_int = 0;

// --- front-end state (was: fend_pool / translation_unit_pool / query_pool) -

var tu_arena: std.heap.ArenaAllocator = undefined;
var query_arena: std.heap.ArenaAllocator = undefined;
var query_active: bool = false;

/// Identifier table: name -> declared type (null slot means "registered but
/// untyped", e.g. `into` / group-join range variables).
var identifiers: std.StringHashMapUnmanaged(?*c.type_info_t) = .empty;

/// Registered by fend_translation_unit_init, fired by fend_query_cleanup with
/// each completed query AST (or NULL on parse error).
var on_complete: ?*const fn (?*c.fend_node_t) callconv(.c) void = null;

fn qalloc() std.mem.Allocator {
    if (query_active) return query_arena.allocator();
    return state.gpa;
}

fn span(s: [*c]u8) []const u8 {
    // [*c] is nullable; a NULL value.string (e.g. on a unary-expression node)
    // must behave like the C apr_hash_get(NULL) path rather than trap.
    if (s == null) return "";
    return std.mem.span(@as([*:0]u8, @ptrCast(s)));
}

/// Recovers a fend_node_t* the grammar passed through a void* slot.
fn asNode(p: ?*anyopaque) ?*c.fend_node_t {
    if (p == null) return null;
    return @ptrCast(@alignCast(p));
}

// --- node construction helpers (port of prfend_create_*) ------------------

fn createNode(left: ?*c.fend_node_t, right: ?*c.fend_node_t, t: c_int) ?*c.fend_node_t {
    const node = qalloc().create(c.fend_node_t) catch {
        signalOom();
        return null;
    };
    node.* = .{
        .type = @intCast(t),
        .left = left,
        .right = right,
        .loc = .{
            .first_line = 0,
            .first_column = 0,
            .last_line = 0,
            .last_column = 0,
        },
    };
    return node;
}

fn createStringNode(left: ?*c.fend_node_t, right: ?*c.fend_node_t, t: c_int, value: [*c]u8) ?*c.fend_node_t {
    const node = createNode(left, right, t) orelse return null;
    node.value = .{ .string = value };
    return node;
}

fn createNumberNode(left: ?*c.fend_node_t, right: ?*c.fend_node_t, t: c_int, value: c_longlong) ?*c.fend_node_t {
    const node = createNode(left, right, t) orelse return null;
    node.value = .{ .number = value };
    return node;
}

/// Boolean literal as a unary-wrapped `node_type_boolean_literal` (value 0/1).
/// Separate from `fend_on_unary_expression` so `false` is not passed as a null void*.
pub export fn fend_on_boolean_literal(value: c_int) ?*c.fend_node_t {
    const lit = createNumberNode(null, null, c.node_type_boolean_literal, value) orelse return null;
    return createNode(lit, null, c.node_type_unary_expression);
}

fn signalOom() void {
    // The C build aborts on allocation failure; here we surface a parser error
    // and let yyparse unwind rather than crash the process.
    fend_error_count += 1;
    diag.report("out of memory during AST construction");
}

/// Attach a bison `YYLTYPE` to an AST node (called from grammar actions).
pub export fn fend_node_set_loc(
    node: ?*c.fend_node_t,
    first_line: c_int,
    first_column: c_int,
    last_line: c_int,
    last_column: c_int,
) void {
    const n = node orelse return;
    n.loc = .{
        .first_line = first_line,
        .first_column = first_column,
        .last_line = last_line,
        .last_column = last_column,
    };
}

/// Called from bison `lyyerror` (same contract as grok's `fend_print_error`).
export fn fend_print_error(
    first_line: c_int,
    first_column: c_int,
    last_line: c_int,
    last_column: c_int,
    message: [*:0]const u8,
) callconv(.c) void {
    fend_error_count += 1;
    diag.reportParse(first_line, first_column, last_line, last_column, std.mem.span(message));
}

// --- translation-unit lifecycle (called from main, not the grammar) --------

pub export fn fend_translation_unit_init(pfn: ?*const fn (?*c.fend_node_t) callconv(.c) void) void {
    tu_arena = std.heap.ArenaAllocator.init(state.gpa);
    on_complete = pfn;
}

pub export fn fend_translation_unit_cleanup() void {
    if (on_complete != null) {
        tu_arena.deinit();
    }
    on_complete = null;
}

pub export fn fend_translation_unit_strdup(str: [*c]u8) [*c]u8 {
    return dupInto(tu_arena.allocator(), str);
}

// --- query lifecycle (grammar: query rule) --------------------------------

pub export fn fend_query_init() void {
    query_arena = std.heap.ArenaAllocator.init(state.gpa);
    query_active = true;
    identifiers = .empty;
}

pub export fn fend_query_cleanup(result: ?*c.fend_node_t) void {
    // Hand the AST to the registered consumer (backend) BEFORE releasing the
    // query arena: the consumer walks the tree while nodes are still live.
    if (on_complete) |cb| cb(result);
    if (query_active) {
        query_arena.deinit();
        query_active = false;
    }
    identifiers.deinit(state.gpa);
    identifiers = .empty;
}

pub export fn fend_query_complete(from: ?*c.fend_node_t, body: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(from, body, c.node_type_query);
}

pub export fn fend_query_strdup(str: [*c]u8) [*c]u8 {
    return dupInto(qalloc(), str);
}

fn dupInto(alloc: std.mem.Allocator, str: [*c]u8) [*c]u8 {
    const s = span(str);
    const mem = alloc.allocSentinel(u8, s.len, 0) catch {
        signalOom();
        return str;
    };
    @memcpy(mem[0..s.len], s);
    return mem.ptr;
}

pub export fn fend_to_number(str: [*c]u8) c_longlong {
    // apr_strtoff(&result, str, NULL, 0): base-0 parse, 0 on failure.
    return std.fmt.parseInt(c_longlong, span(str), 0) catch 0;
}

// --- type definitions (lexer DEFINITION state) ----------------------------

pub export fn fend_on_simple_type_def(t: c_int) ?*c.type_info_t {
    const ti = qalloc().create(c.type_info_t) catch {
        signalOom();
        return null;
    };
    ti.* = .{ .type = @intCast(t), .info = null };
    return ti;
}

pub export fn fend_on_complex_type_def(t: c_int, info: [*c]u8) ?*c.type_info_t {
    const ti = fend_on_simple_type_def(t) orelse return null;
    ti.info = fend_query_strdup(info);
    return ti;
}

// --- AST builders (grammar semantic actions) ------------------------------

pub export fn fend_on_identifier_declaration(
    type_info: ?*c.type_info_t,
    identifier: ?*c.fend_node_t,
) ?*c.fend_node_t {
    const id = identifier orelse return null;
    const ti = type_info orelse return id;
    const key = span(id.value.string);
    identifiers.put(state.gpa, key, ti) catch signalOom();
    id.left = fend_on_type_attribute(ti);
    return id;
}

pub export fn fend_on_unary_expression(
    t: c_int,
    left_value: ?*anyopaque,
    right_value: ?*anyopaque,
) ?*c.fend_node_t {
    const expr = createNode(null, null, c.node_type_unary_expression) orelse return null;
    switch (t) {
        c.unary_exp_type_identifier => {
            expr.left = asNode(left_value);
        },
        c.unary_exp_type_string => {
            expr.left = createStringNode(null, null, c.node_type_string_literal, @ptrCast(left_value));
        },
        c.unary_exp_type_number => {
            // The grammar stores the integer in a void* slot; recover it.
            const num: c_longlong = if (left_value) |lv| @intCast(@intFromPtr(lv)) else 0;
            expr.left = createNumberNode(null, null, c.node_type_numeric_literal, num);
        },
        c.unary_exp_type_property_call, c.unary_exp_type_mehtod_call => {
            expr.left = asNode(left_value);
            expr.right = asNode(right_value);
        },
        else => {},
    }
    return expr;
}

pub export fn fend_on_from(type_node: ?*c.fend_node_t, datasource: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(type_node, datasource, c.node_type_from);
}

pub export fn fend_on_where(expr: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(expr, null, c.node_type_where);
}

pub export fn fend_on_releational_expr(
    left: ?*c.fend_node_t,
    right: ?*c.fend_node_t,
    relation: c_int,
) ?*c.fend_node_t {
    const node = createNode(left, right, c.node_type_relation) orelse return null;
    node.value = .{ .relation_op = relation };
    return node;
}

pub export fn fend_on_predicate(
    left: ?*c.fend_node_t,
    right: ?*c.fend_node_t,
    t: c_int,
) ?*c.fend_node_t {
    return createNode(left, right, t);
}

pub export fn fend_on_enum(left: ?*c.fend_node_t, right: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(left, right, c.node_type_enum);
}

pub export fn fend_on_group(left: ?*c.fend_node_t, right: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(left, right, c.node_type_group);
}

pub export fn fend_on_let(id: ?*c.fend_node_t, expr: ?*c.fend_node_t) ?*c.fend_node_t {
    if (id) |id_node| {
        const id_key = span(id_node.value.string);
        var declared_type: ?*c.type_info_t = null;

        if (expr) |e| {
            if (e.type == c.node_type_identifier) {
                const expr_key = span(e.value.string);
                declared_type = identifiers.get(expr_key) orelse null;
            }
        }

        identifiers.put(state.gpa, id_key, declared_type) catch signalOom();
    }
    return createNode(id, expr, c.node_type_let);
}

pub export fn fend_on_named_field(id: ?*c.fend_node_t, expr: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(id, expr, c.node_type_let);
}

pub export fn fend_on_object(fields: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(fields, null, c.node_type_object);
}

pub export fn fend_on_query_body(
    opt_query_body_clauses: ?*c.fend_node_t,
    select_or_group_clause: ?*c.fend_node_t,
    opt_query_continuation: ?*c.fend_node_t,
) ?*c.fend_node_t {
    const select = createNode(opt_query_body_clauses, select_or_group_clause, c.node_type_select) orelse return null;
    return createNode(select, opt_query_continuation, c.node_type_query_body);
}

pub export fn fend_on_string_attribute(str: [*c]u8) ?*c.fend_node_t {
    return createStringNode(null, null, c.node_type_property, str);
}

pub export fn fend_on_type_attribute(type_info: ?*c.type_info_t) ?*c.fend_node_t {
    const ti = type_info orelse return null;
    const node = createNode(null, null, c.node_type_internal_type) orelse return null;
    node.value = .{ .type = ti.type };
    if (ti.info != null) {
        node.left = createStringNode(null, null, c.node_type_string_literal, ti.info);
    }
    return node;
}

pub export fn fend_on_continuation(id: ?*c.fend_node_t, query_body: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(id, query_body, c.node_type_query_continuation);
}

pub export fn fend_on_method_call(method: [*c]u8, arguments: ?*c.fend_node_t) ?*c.fend_node_t {
    return createStringNode(arguments, null, c.node_type_method_call, method);
}

pub export fn fend_on_identifier(id: [*c]u8) ?*c.fend_node_t {
    return createStringNode(null, null, c.node_type_identifier, id);
}

pub export fn fend_on_join(
    identifier: ?*c.fend_node_t,
    in: ?*c.fend_node_t,
    on_first: ?*c.fend_node_t,
    on_second: ?*c.fend_node_t,
) ?*c.fend_node_t {
    const on_node = createNode(on_first, on_second, c.node_type_on) orelse return null;
    const in_node = createNode(in, on_node, c.node_type_in) orelse return null;
    return createNode(identifier, in_node, c.node_type_join);
}

pub export fn fend_on_order_by(ordering: ?*c.fend_node_t) ?*c.fend_node_t {
    return createNode(ordering, null, c.node_type_order_by);
}

pub export fn fend_on_ordering(ordering: ?*c.fend_node_t, direction: c_int) ?*c.fend_node_t {
    const node = createNode(ordering, null, c.node_type_ordering) orelse return null;
    node.value = .{ .ordering = @intCast(direction) };
    return node;
}

// --- identifier table ------------------------------------------------------

/// Returns 0/1 as `int` (not Zig `bool` / C `_Bool`). On Windows MSVC the
/// bison-generated caller treats this as `BOOL`=`int`; a 1-byte bool return
/// leaves garbage in the high bits of RAX and breaks identifier checks.
pub export fn fend_is_identifier_defined(id: ?*c.fend_node_t) c_int {
    const node = id orelse return 0;
    // Key present (including continuation ids registered with a null type).
    const key = span(node.value.string);
    return @intFromBool(identifiers.get(key) != null);
}

pub export fn fend_register_identifier(id: ?*c.fend_node_t) void {
    const node = id orelse return;
    const key = span(node.value.string);
    // Bind a continuation / group-join range variable into scope (semantics
    // `into`). Legacy APR code deleted the key; that made `into` unusable.
    identifiers.put(state.gpa, key, null) catch signalOom();
}

/// Exposed for compilation: current-query declared type of an identifier, if any.
/// `null` means either "unknown/untyped" (e.g. `into`) or absent.
pub fn identifierDeclaredType(name: []const u8) ?c.type_def_t {
    if (identifiers.get(name)) |maybe_ti| {
        if (maybe_ti) |ti| return ti.type;
    }
    return null;
}

test "fend_to_number parses decimal and hex" {
    try std.testing.expectEqual(@as(c_longlong, 255), fend_to_number(@constCast("255".ptr)));
    try std.testing.expectEqual(@as(c_longlong, 0xff), fend_to_number(@constCast("0xff".ptr)));
    try std.testing.expectEqual(@as(c_longlong, 0), fend_to_number(@constCast("not-a-number".ptr)));
}

test "fend_on_identifier builds identifier node within a query" {
    state.gpa = std.testing.allocator;
    const Callback = struct {
        var captured: ?*c.fend_node_t = null;
        fn cb(ast: ?*c.fend_node_t) callconv(.c) void {
            captured = ast;
        }
    };
    Callback.captured = null;
    fend_translation_unit_init(Callback.cb);
    defer fend_translation_unit_cleanup();

    fend_query_init();
    defer fend_query_cleanup(null);

    const node = fend_on_identifier(@constCast("foo".ptr));
    try std.testing.expect(node != null);
    try std.testing.expectEqual(@as(c.node_type_t, @intCast(c.node_type_identifier)), node.?.type);
    try std.testing.expectEqualStrings("foo", span(node.?.value.string));
    try std.testing.expect(Callback.captured == null); // cleanup(NULL) callback ran with null
}

