//! GoogleTest TreeTest parity: verifies the postorder tree traversal ported into
//! the l2h backend (backend.postorder, the port of treeutil.c tree_postorder).
//!
//! The C++ suite also covers tree_inorder / tree_preorder, but only postorder is
//! needed by the backend (AST lowering) and therefore is the only traversal
//! ported to Zig; the other two are C-only helpers and are out of scope.
//!
//! Tree shape (matches TreeTest.cpp SetUp):
//!          1
//!         / \
//!        2   3
//!       /   / \
//!      4   5   6
//! postorder visit order: 4, 2, 5, 6, 3, 1.

const std = @import("std");
const c = @import("c");
const backend = @import("backend.zig");

var path: [16]c_longlong = undefined;
var path_len: usize = 0;

fn onVisit(node: *c.fend_node_t) void {
    path[path_len] = node.value.number;
    path_len += 1;
}

fn createNode(allocator: std.mem.Allocator, value: c_longlong) !*c.fend_node_t {
    const node = try allocator.create(c.fend_node_t);
    node.type = 0;
    node.left = null;
    node.right = null;
    node.value = .{ .number = value };
    return node;
}

test "TreeTest postorder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // backend.postorder uses the backend arena for its traversal stack.
    backend.init(a);
    defer backend.complete();

    path_len = 0;

    // Build the tree. fend_node_t.left/right are C pointers ([*c]); assign each
    // link through a held single-pointer variable rather than chaining field
    // access through the [*c] fields.
    const root = try createNode(a, 1);
    const n2 = try createNode(a, 2);
    const n3 = try createNode(a, 3);
    const n4 = try createNode(a, 4);
    const n5 = try createNode(a, 5);
    const n6 = try createNode(a, 6);
    root.left = n2;
    root.right = n3;
    n2.left = n4;
    n3.left = n5;
    n3.right = n6;

    backend.postorder(root, onVisit);

    try std.testing.expectEqual(@as(usize, 6), path_len);
    try std.testing.expectEqual(@as(c_longlong, 4), path[0]);
    try std.testing.expectEqual(@as(c_longlong, 2), path[1]);
    try std.testing.expectEqual(@as(c_longlong, 5), path[2]);
    try std.testing.expectEqual(@as(c_longlong, 6), path[3]);
    try std.testing.expectEqual(@as(c_longlong, 3), path[4]);
    try std.testing.expectEqual(@as(c_longlong, 1), path[5]);
}
