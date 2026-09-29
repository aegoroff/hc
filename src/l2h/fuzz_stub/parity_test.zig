//! Checks that the fuzz stubs expose the same interface as the production
//! `modes` / `hashes` modules they replace in the l2h fuzz build.

const std = @import("std");
const real = @import("modes");
const stub = @import("stub_modes");
const real_hashes = @import("hashes");
const stub_hashes = @import("stub_hashes");

fn expectSameType(comptime name: []const u8, comptime Real: type, comptime Stub: type) void {
    if (Real != Stub) @compileError("fuzz stub " ++ name ++ " is " ++ @typeName(Stub) ++ ", production is " ++ @typeName(Real));
}

fn expectSameParams(comptime name: []const u8, comptime real_fn: anytype, comptime stub_fn: anytype, comptime skip_first: bool) void {
    const r = @typeInfo(@TypeOf(real_fn)).@"fn".params;
    const s = @typeInfo(@TypeOf(stub_fn)).@"fn".params;
    if (r.len != s.len) @compileError("fuzz stub " ++ name ++ " takes a different number of parameters");
    for (r, s, 0..) |rp, sp, i| {
        if (skip_first and i == 0) continue;
        if (rp.type != sp.type) @compileError("fuzz stub " ++ name ++ " parameter types differ from production");
    }
}

test "fuzz stub modes expose the production functions" {
    comptime {
        // Arrange / Act / Assert: all checks happen at compile time.
        expectSameType("types", real.types, stub.types);
        expectSameType("file.createFileDigest", @TypeOf(real.file.createFileDigest), @TypeOf(stub.file.createFileDigest));
        expectSameType("file.pathKind", @TypeOf(real.file.pathKind), @TypeOf(stub.file.pathKind));
        expectSameType("file.fileSize", @TypeOf(real.file.fileSize), @TypeOf(stub.file.fileSize));
        expectSameType("dir.dirExists", @TypeOf(real.dir.dirExists), @TypeOf(stub.dir.dirExists));
        expectSameType("dir.WalkStep", real.dir.WalkStep, stub.dir.WalkStep);
        expectSameParams("dir.FileWalk.init", real.dir.FileWalk.init, stub.dir.FileWalk.init, false);
        expectSameParams("dir.FileWalk.next", real.dir.FileWalk.next, stub.dir.FileWalk.next, true);
        expectSameParams("dir.FileWalk.deinit", real.dir.FileWalk.deinit, stub.dir.FileWalk.deinit, true);
        expectSameType("hash.restore", @TypeOf(real.hash.restore), @TypeOf(stub.hash.restore));
    }
}

test "fuzz stub alphabet matches production" {
    // Arrange
    const want = real.DEFAULT_ALPHABET;

    // Act
    const got = stub.DEFAULT_ALPHABET;

    // Assert
    try std.testing.expectEqualStrings(want, got);
}

test "fuzz stub hash table lists the production hashes" {
    // Arrange
    const want = &real_hashes.hashes;

    // Act
    const got = &stub_hashes.hashes;

    // Assert
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| {
        try std.testing.expectEqualStrings(w.name, g.name);
        try std.testing.expectEqual(w.hash_length, g.hash_length);
        try std.testing.expectEqual(w.use_wide_string, g.use_wide_string);
    }
}

test "fuzz stub fails to open missing* paths and opens the rest" {
    // Arrange
    const io = std.Io.Threaded.global_single_threaded.io();

    // Act
    const missing_kind = stub.file.pathKind(io, "/tmp/missing.txt");
    const empty_size = stub.file.fileSize(io, "");
    const missing_dir = stub.dir.dirExists(io, "missing");
    const kind = try stub.file.pathKind(io, "/tmp/a.txt");
    const dir = stub.dir.dirExists(io, "/tmp");

    // Assert
    try std.testing.expectError(error.OpenFailed, missing_kind);
    try std.testing.expectError(error.OpenFailed, empty_size);
    try std.testing.expect(!missing_dir);
    try std.testing.expectEqual(std.Io.File.Kind.file, kind);
    try std.testing.expect(dir);
}
