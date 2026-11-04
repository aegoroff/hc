const std = @import("std");
const modes = @import("modes");
const builtins = @import("builtins.zig");
const value = @import("value.zig");
const diag = @import("diag.zig");

const Error = builtins.Error;

/// Yields regular-file paths under a Dir one at a time (walk order, no sort).
pub const DirFileIter = struct {
    skip_errors: bool,
    /// `null` when the root could not be opened for listing under
    /// `skipErrors()`: the walk yields no files.
    walk: ?modes.dir.FileWalk,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, dir: value.DirVal) Error!DirFileIter {
        // `from dir` already rejected a missing path or a non-directory; a root
        // that exists but cannot be listed is a walk failure like any other.
        const walk = modes.dir.FileWalk.init(gpa, io, dir.path, dir.max_depth) catch |err| switch (err) {
            error.OpenFailed => if (dir.skip_errors) null else return diag.ioFail(dir.path),
            error.OutOfMemory => return error.OutOfMemory,
        };
        return .{ .skip_errors = dir.skip_errors, .walk = walk };
    }

    pub fn deinit(self: *DirFileIter) void {
        if (self.walk) |*w| w.deinit();
    }

    /// Owned path allocated with `path_gpa`, or `null` when exhausted.
    pub fn next(self: *DirFileIter, path_gpa: std.mem.Allocator) Error!?[]const u8 {
        const walk = if (self.walk) |*w| w else return null;
        while (true) {
            const step = (try walk.next(path_gpa)) orelse return null;
            switch (step) {
                .file => |p| return p,
                .failed => |fail| {
                    defer path_gpa.free(fail.path);
                    if (self.skip_errors) continue;
                    return diag.ioFail(fail.path);
                },
            }
        }
    }
};
