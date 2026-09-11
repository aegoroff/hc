//! Fuzz-only `modes` stand-in (paired with `fuzz_stub/hashes`).
//!
//! - string digests / hash-check: via `@import("hashes")` (std-backed stub)
//! - `file.createFileDigest`, `dir.FileWalk`, `hash.restore`: no-ops so fuzz
//!   does not touch the filesystem or run brute-force restore
//!
//! Not used by the production `l2h` executable.

const std = @import("std");
const hashes = @import("hashes");

pub const types = struct {
    pub const SFV_SEPARATOR = "    ";
    pub const CHECKSUM_SEPARATOR = " ";
    pub const MAX_DIGEST_SIZE: usize = 64;

    pub fn hashToHex(digest: []const u8, _: bool, buf: []u8) []const u8 {
        const hex_len = digest.len * 2;
        const digits = "0123456789abcdef";
        for (digest, 0..) |byte, i| {
            buf[i * 2] = digits[byte >> 4];
            buf[i * 2 + 1] = digits[byte & 0xf];
        }
        return buf[0..hex_len];
    }

    /// Same contract as production; used only by hash-restore (stubbed below).
    pub fn parseSearchHash(
        search_hash: []const u8,
        is_base64: bool,
        hash_def: *const hashes.HashDefinition,
        out: []u8,
    ) !void {
        if (is_base64) {
            const dec = std.base64.standard.Decoder;
            const expected_len = hash_def.hash_length;
            const decoded_size = dec.calcSizeForSlice(search_hash) catch return error.InvalidArgument;
            if (decoded_size != expected_len) return error.InvalidArgument;
            dec.decode(out[0..expected_len], search_hash) catch return error.InvalidArgument;
        } else {
            const expected_len = hash_def.hash_length;
            if (search_hash.len != expected_len * 2) return error.InvalidArgument;
            _ = std.fmt.hexToBytes(out[0..expected_len], search_hash) catch return error.InvalidArgument;
        }
    }
};

pub const file = struct {
    pub const OFFSET_TOO_BIG = "Offset is greater than file size";

    pub const FileDigest = struct {
        bytes: [types.MAX_DIGEST_SIZE]u8 = undefined,
        len: usize = 0,
        pub fn slice(self: *const FileDigest) []const u8 {
            return self.bytes[0..self.len];
        }
    };

    pub fn createFileDigest(
        def: *const hashes.HashDefinition,
        _: []const u8,
        _: anytype,
        _: std.Io,
    ) error{ OpenFailed, StatFailed, ReadFailed, OffsetPastEof, OutOfMemory }!FileDigest {
        var d: FileDigest = .{ .len = def.hash_length };
        @memset(d.bytes[0..def.hash_length], 0);
        return d;
    }
};

pub const hash = struct {
    pub fn restore(
        _: *const hashes.HashDefinition,
        _: []const u8,
        _: anytype,
        _: std.mem.Allocator,
        _: std.Io,
        _: *std.Io.Writer,
    ) !?[]u8 {
        return null;
    }
};

pub const dir = struct {
    pub const WalkStep = union(enum) {
        file: []u8,
        failed: struct { path: []u8, err: anyerror },
    };

    pub const FileWalk = struct {
        pub fn init(
            _: std.mem.Allocator,
            _: std.Io,
            _: []const u8,
            _: ?u32,
        ) error{ OpenFailed, OutOfMemory }!FileWalk {
            return .{};
        }

        pub fn deinit(_: *FileWalk) void {}

        pub fn next(_: *FileWalk, _: std.mem.Allocator) error{OutOfMemory}!?WalkStep {
            return null;
        }
    };
};

pub const defaultAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
