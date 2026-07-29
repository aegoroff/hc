//! GoogleTest TreeTest parity: postorder traversal (tree.zig).

const std = @import("std");
const c = @import("c");
const tree = @import("tree.zig");

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
    // Arrange
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    path_len = 0;

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

    // Act
    tree.postorder(a, root, onVisit);

    // Assert
    try std.testing.expectEqual(@as(usize, 6), path_len);
    try std.testing.expectEqual(@as(c_longlong, 4), path[0]);
    try std.testing.expectEqual(@as(c_longlong, 2), path[1]);
    try std.testing.expectEqual(@as(c_longlong, 5), path[2]);
    try std.testing.expectEqual(@as(c_longlong, 6), path[3]);
    try std.testing.expectEqual(@as(c_longlong, 3), path[4]);
    try std.testing.expectEqual(@as(c_longlong, 1), path[5]);
}
