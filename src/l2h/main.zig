//! l2h (linq2hash) Zig driver.
//!
//! Parses queries via bison/flex, compiles the AST to a From plan, then executes the
//! replacement interpreter.

const std = @import("std");
const state = @import("state.zig");
const front = @import("frontend.zig");
const cli = @import("cli.zig");
const diag = @import("diag.zig");
const driver = @import("driver.zig");

pub const run = driver.run;

pub fn main(init: std.process.Init) !void {
    @import("lib").setupConsoleUtf8();
    // Same static-libcrypto CPUID issue as hc: activate SHA-NI before digests.
    @import("hashes").ensureOpenSslReady();

    var stdout_buf: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buf);
    defer stdout_writer.flush() catch {};

    const gpa = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(gpa);

    try driver.run(gpa, &stdout_writer.interface, init.io, argv[1..]);

    try stdout_writer.interface.flush();
    if (state.had_error) std.process.exit(1);
}

test {
    _ = front;
    _ = cli;
    _ = diag;
    _ = driver;
    _ = @import("value.zig");
    _ = @import("expr.zig");
    _ = @import("plan.zig");
    _ = @import("props.zig");
    _ = @import("method.zig");
    _ = @import("builtins.zig");
    _ = @import("compile.zig");
    _ = @import("string_lit.zig");
    _ = @import("interpret.zig");
    _ = @import("match_re.zig");
    _ = @import("test_stderr.zig");
    _ = @import("frontend_test.zig");
    _ = @import("compile_test.zig");
}

const test_stderr = @import("test_stderr.zig");

var syntax_out_buf: [4096]u8 = undefined;
var syntax_out_writer: std.Io.Writer = undefined;

fn setupSyntaxTest() void {
    syntax_out_writer = .fixed(&syntax_out_buf);
    state.had_error = false;
    diag.clearLast();
}

test "syntax-check skips interpret for missing file" {
    // Arrange
    setupSyntaxTest();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const argv = [_][:0]const u8{
        "-n",
        "-q",
        "from file f in '/definitely-missing-l2h-syntax-check' select f.size;",
    };

    // Act
    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);
    try driver.run(arena.allocator(), &syntax_out_writer, std.testing.io, &argv);

    // Assert
    try std.testing.expect(!state.had_error);
    try std.testing.expectEqualStrings("", std.Io.Writer.buffered(&syntax_out_writer));
}

test "without syntax-check missing file fails at runtime" {
    // Arrange
    setupSyntaxTest();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const argv = [_][:0]const u8{
        "-q",
        "from file f in '/definitely-missing-l2h-syntax-check' select f.size;",
    };

    // Act
    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);
    try driver.run(arena.allocator(), &syntax_out_writer, std.testing.io, &argv);

    // Assert
    try std.testing.expect(state.had_error);
}

test "syntax-check still reports compile errors" {
    // Arrange
    setupSyntaxTest();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const argv = [_][:0]const u8{
        "-n",
        "-q",
        "from string s in 'a' select s.no_such_prop;",
    };

    // Act
    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);
    try driver.run(arena.allocator(), &syntax_out_writer, std.testing.io, &argv);

    // Assert
    try std.testing.expect(state.had_error);
}
