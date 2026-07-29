const std = @import("std");
const c = @import("c");

/// Iterative postorder traversal over `fend_node_t` trees (port of treeutil.c
/// tree_postorder). Used by tests and (later) AST lowerers.

pub fn postorder(
    allocator: std.mem.Allocator,
    root_arg: ?*c.fend_node_t,
    action: *const fn (*c.fend_node_t) void,
) void {
    var root = root_arg orelse return;
    var stack: std.ArrayListUnmanaged(*c.fend_node_t) = .empty;
    defer stack.deinit(allocator);
    stack.append(allocator, root) catch return;

    while (stack.items.len > 0) {
        const next = stack.items[stack.items.len - 1];
        const finished_subtrees = (next.right == root or next.left == root);
        const is_leaf = (next.left == null and next.right == null);
        if (finished_subtrees or is_leaf) {
            _ = stack.pop();
            action(next);
            root = next;
        } else {
            if (next.right) |r| stack.append(allocator, r) catch {};
            if (next.left) |l| stack.append(allocator, l) catch {};
        }
    }
}
