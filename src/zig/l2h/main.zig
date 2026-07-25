const std = @import("std");
const c = @import("c");

// Skeleton entry point for the l2h (linq2hash) Zig port.
//
// Task 7 only wires the flex/bison pipeline: the parser is generated, compiled
// into a static lib, and its token constants/types are surfaced to Zig through
// translate-c. main() exercises the translated `c` module by reading a token;
// the real query REPL arrives in Task 8.
//
// The grammar (src/l2h/l2h.y) calls the fend_on_* callbacks below, so the Zig
// exports MUST keep those exact names (the task brief's "prl2h_on_*" prefix
// would break linking without grammar edits, which are explicitly out of scope).
// These are no-op stubs; Task 8 implements real AST construction.

pub fn main(init: std.process.Init) !void {
    var buf: [256]u8 = undefined;
    var w = std.Io.File.stdout().writer(init.io, &buf);
    defer w.flush() catch {};

    // Prove translate-c surfaced the bison token table.
    try w.interface.print("l2h skeleton: FROM token = {d}, SELECT = {d}\n", .{ c.FROM, c.SELECT });
}

// --- fend_on_* callback stubs (Task 8 fills in AST semantics) ---

pub export fn fend_query_init() void {}

pub export fn fend_query_cleanup(result: ?*c.fend_node_t) void {
    _ = result;
}

pub export fn fend_query_complete(from: ?*c.fend_node_t, body: ?*c.fend_node_t) ?*c.fend_node_t {
    _ = from;
    _ = body;
    return null;
}

pub export fn fend_query_strdup(str: [*c]u8) [*c]u8 {
    return str;
}

pub export fn fend_to_number(str: [*c]u8) c_longlong {
    _ = str;
    return 0;
}

pub export fn fend_on_identifier(id: [*c]u8) ?*c.fend_node_t {
    _ = id;
    return null;
}

pub export fn fend_on_from(node_type: ?*c.fend_node_t, datasource: ?*c.fend_node_t) ?*c.fend_node_t {
    _ = node_type;
    _ = datasource;
    return null;
}
