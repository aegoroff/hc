const std = @import("std");
const builtin = @import("builtin");
const modes = @import("modes");
const builtins = @import("builtins.zig");
const state = @import("state.zig");
const value = @import("value.zig");
const diag = @import("diag.zig");

const Error = builtins.Error;

/// Yields regular files under a Dir one at a time (walk order, no sort).
/// On POSIX each file carries the open walk-root handle plus its path
/// relative to that root, so later open/stat never resolve a full path that
/// may exceed PATH_MAX. The handle is borrowed from `state.walk_roots` and
/// stays open until translation-unit cleanup.
pub const DirFileIter = struct {
    skip_errors: bool,
    io: std.Io,
    root_path: []const u8,
    /// `null` when the root could not be opened for listing under
    /// `skipErrors()`: the walk yields no files.
    walk: ?modes.dir.FileWalk,
    /// Registered walk-root handle; null until the first file is yielded.
    root: ?std.Io.Dir = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, dir: value.DirVal) Error!DirFileIter {
        // `from dir` already rejected a missing path or a non-directory; a root
        // that exists but cannot be listed is a walk failure like any other.
        const walk = modes.dir.FileWalk.init(gpa, io, dir.path, dir.max_depth) catch |err| switch (err) {
            error.OpenFailed => if (dir.skip_errors) null else return diag.ioFail(dir.path),
            error.OutOfMemory => return error.OutOfMemory,
        };
        return .{
            .skip_errors = dir.skip_errors,
            .io = io,
            .root_path = dir.path,
            .walk = walk,
        };
    }

    pub fn deinit(self: *DirFileIter) void {
        if (self.walk) |*w| w.deinit();
    }

    /// Owned file value allocated with `gpa`, or `null` when exhausted.
    pub fn next(self: *DirFileIter, gpa: std.mem.Allocator) Error!?value.FileVal {
        const walk = if (self.walk) |*w| w else return null;
        while (true) {
            const step = (try walk.next(gpa)) orelse return null;
            switch (step) {
                .file => |wf| {
                    if (comptime builtin.target.os.tag == .windows) return .{ .path = wf.path };
                    const root = self.root orelse blk: {
                        const r = state.walk_roots.getOrOpen(self.io, state.gpa, self.root_path, walk.root) catch |err| {
                            gpa.free(wf.path);
                            if (err == error.OutOfMemory) return error.OutOfMemory;
                            if (self.skip_errors) {
                                walk.deinit();
                                self.walk = null;
                                return null;
                            }
                            return diag.ioFail(self.root_path);
                        };
                        self.root = r;
                        break :blk r;
                    };
                    const rel = gpa.dupe(u8, wf.rel) catch |err| {
                        gpa.free(wf.path);
                        return err;
                    };
                    return .{ .path = wf.path, .walk = .{ .root = root, .rel = rel } };
                },
                .failed => |fail| {
                    defer gpa.free(fail.path);
                    if (self.skip_errors) continue;
                    return diag.ioFail(fail.path);
                },
            }
        }
    }
};
