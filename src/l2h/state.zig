const std = @import("std");

// Shared runtime context for the l2h Zig frontend and interpreter.
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

/// Display name for the unit being compiled (`path`, `<query>`, or `<stdin>`).
/// Used by fehler diagnostics as the source file label.
pub var source_name: []const u8 = "<query>";

/// Full source text of the current compilation unit (not owned; set by main/tests).
pub var source_text: []const u8 = "";

/// Convenience accessor that assumes main() has wired `out`.
pub fn writer() *std.Io.Writer {
    return out orelse {
        std.debug.print("l2h: stdout writer not initialized\n", .{});
        @panic("l2h state.out is null");
    };
}
