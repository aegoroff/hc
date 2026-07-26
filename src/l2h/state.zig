const std = @import("std");

// Shared runtime context for the l2h Zig frontend/backend/processor.
//
// main() initializes these once (Juicy Main supplies the Io + allocators); the
// C-ABI fend_on_* callbacks and the evaluation pipeline read them through this
// single namespace rather than threading context through the generated parser.

/// General-purpose allocator (process arena from Juicy Main).
pub var gpa: std.mem.Allocator = undefined;

/// Default Io implementation (Juicy Main).
pub var io: std.Io = undefined;

/// Buffered stdout writer the backend writes results to. Set up by main().
pub var out: ?*std.Io.Writer = null;

/// Convenience accessor that assumes main() has wired `out`.
pub fn writer() *std.Io.Writer {
    return out orelse {
        std.debug.print("l2h: stdout writer not initialized\n", .{});
        @panic("l2h state.out is null");
    };
}
