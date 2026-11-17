//! Shared runtime context for the l2h Zig frontend and interpreter.
//!
//! main() initializes these once (Juicy Main supplies the Io + allocators); the
//! C-ABI fend_on_* callbacks and the evaluation pipeline read them through this
//! single namespace rather than threading context through the generated parser.

const std = @import("std");

/// Process storage. `main` sets this to the process arena.
pub var gpa: std.mem.Allocator = undefined;

/// Owns script `into` keys and values. Individual frees must reclaim:
/// `main` passes the libc allocator, tests pass `std.testing.allocator`.
pub var script_alloc: std.mem.Allocator = undefined;

/// Default Io implementation (Juicy Main).
pub var io: std.Io = undefined;

/// Buffered stdout writer the backend writes results to. Set up by main().
pub var out: ?*std.Io.Writer = null;

/// Display name for the unit being compiled (`path`, `<query>`, or `<stdin>`).
/// Used by fehler diagnostics as the source file label.
pub var source_name: []const u8 = "<query>";

/// Full source text of the current compilation unit (not owned; set by main/tests).
pub var source_text: []const u8 = "";

/// True when any query in the compilation unit failed (file read, parse,
/// compile, or runtime). Checked by main() to set a non-zero process exit code.
pub var had_error: bool = false;

/// When true, parse and compile queries but do not interpret them.
pub var syntax_check: bool = false;

/// Open walk-root handles that `FileVal.walk` borrows, keyed by root display
/// path. One handle per distinct root; closed at translation-unit cleanup.
/// Values never close these: row arenas free values without destructors, and
/// script `into` bindings outlive the query that produced them.
pub var walk_roots: WalkRoots = .{};

pub const WalkRoots = struct {
    map: std.StringHashMapUnmanaged(std.Io.Dir) = .empty,

    /// Returns the cached handle for `root_path`, or opens `.` relative to
    /// `walk_root` (no path resolution, so roots deeper than PATH_MAX work)
    /// and caches it. Keys are duped with `gpa_`; `deinit` frees them.
    pub fn getOrOpen(
        self: *WalkRoots,
        io_: std.Io,
        gpa_: std.mem.Allocator,
        root_path: []const u8,
        walk_root: std.Io.Dir,
    ) (std.mem.Allocator.Error || std.Io.Dir.OpenError)!std.Io.Dir {
        if (self.map.get(root_path)) |dir| return dir;
        const dir = try walk_root.openDir(io_, ".", .{});
        errdefer dir.close(io_);
        const key = try gpa_.dupe(u8, root_path);
        errdefer gpa_.free(key);
        try self.map.put(gpa_, key, dir);
        return dir;
    }

    pub fn deinit(self: *WalkRoots, io_: std.Io, gpa_: std.mem.Allocator) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.close(io_);
            gpa_.free(entry.key_ptr.*);
        }
        self.map.deinit(gpa_);
        self.* = .{};
    }
};

test "WalkRoots caches one handle per root and deinit releases the keys" {
    // Arrange
    const io_ = std.Io.Threaded.global_single_threaded.io();
    const gpa_ = std.testing.allocator;
    const base = "l2h_walk_roots_probe";
    std.Io.Dir.cwd().deleteTree(io_, base) catch {};
    defer std.Io.Dir.cwd().deleteTree(io_, base) catch {};
    try std.Io.Dir.cwd().createDir(io_, base, std.Io.Dir.Permissions.default_dir);
    var walk_root = try std.Io.Dir.cwd().openDir(io_, base, .{ .iterate = true });
    defer walk_root.close(io_);
    var f = try walk_root.createFile(io_, "x.txt", .{});
    f.close(io_);

    var roots: WalkRoots = .{};
    defer roots.deinit(io_, gpa_);

    // Act
    const a = try roots.getOrOpen(io_, gpa_, base, walk_root);
    const b = try roots.getOrOpen(io_, gpa_, base, walk_root);
    const st = try a.statFile(io_, "x.txt", .{});

    // Assert
    try std.testing.expectEqual(a.handle, b.handle);
    try std.testing.expectEqual(@as(u32, 1), roots.map.count());
    try std.testing.expectEqual(std.Io.File.Kind.file, st.kind);
}

/// Convenience accessor that assumes main() has wired `out`.
pub fn writer() *std.Io.Writer {
    return out orelse {
        std.debug.print("l2h: stdout writer not initialized\n", .{});
        @panic("l2h state.out is null");
    };
}
