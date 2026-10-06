//! Fuzz-only `modes` stand-in (paired with `fuzz_stub/hashes`).
//!
//! Shares `types` with production, so constants, digest parsing and
//! the file/walk/restore types cannot drift. The functions below do not
//! touch the filesystem or run a brute-force restore. A path whose base name
//! starts with `missing` (or an empty path) fails to open, so the fuzzer
//! still reaches the I/O error diagnostics; any other path is a readable
//! empty regular file or an existing empty directory.
//! `parity_test.zig` checks that production exposes the same functions.
//!
//! Not used by the production `l2h` executable.

const std = @import("std");
const hashes = @import("hashes");
pub const types = @import("types");

pub const DEFAULT_ALPHABET = @import("lib").DEFAULT_ALPHABET;

fn isMissing(path: []const u8) bool {
    return path.len == 0 or std.mem.startsWith(u8, std.Io.Dir.path.basename(path), "missing");
}

pub const file = struct {
    pub fn createFileDigest(
        def: *const hashes.HashDefinition,
        path: []const u8,
        _: types.FileWindow,
        _: std.Io,
    ) types.FileDigestError!types.FileDigest {
        if (isMissing(path)) return error.OpenFailed;
        return .{ .len = def.hash_length, .file_size = 0 };
    }

    pub fn pathKind(_: std.Io, path: []const u8) types.PathError!std.Io.File.Kind {
        if (isMissing(path)) return error.OpenFailed;
        return .file;
    }

    pub fn fileSize(_: std.Io, path: []const u8) types.PathError!u64 {
        if (isMissing(path)) return error.OpenFailed;
        return 0;
    }
};

pub const hash = struct {
    pub fn restore(
        _: *const hashes.HashDefinition,
        _: []const u8,
        _: types.RestoreOpts,
        _: std.mem.Allocator,
        _: std.Io,
        _: *std.Io.Writer,
    ) types.RestoreError!?[]u8 {
        return null;
    }
};

pub const dir = struct {
    pub const WalkStep = types.WalkStep;

    pub fn dirExists(_: std.Io, path: []const u8) bool {
        return !isMissing(path);
    }

    pub const FileWalk = struct {
        pub fn init(
            _: std.mem.Allocator,
            _: std.Io,
            path: []const u8,
            _: ?u32,
        ) error{ OpenFailed, OutOfMemory }!FileWalk {
            if (isMissing(path)) return error.OpenFailed;
            return .{};
        }

        pub fn deinit(_: *FileWalk) void {}

        pub fn next(_: *FileWalk, _: std.mem.Allocator) error{OutOfMemory}!?WalkStep {
            return null;
        }
    };
};
