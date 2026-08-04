const std = @import("std");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");
const cli = @import("cli.zig");
const compile = @import("compile.zig");
const interpret = @import("interpret.zig");
const diag = @import("diag.zig");

// l2h (linq2hash) Zig driver.
//
// Parses queries via bison/flex, compiles the AST to a From plan, then executes the
// replacement interpreter.

pub fn main(init: std.process.Init) !void {
    @import("lib").setupConsoleUtf8();
    // Same static-libcrypto CPUID issue as hc: activate SHA-NI before digests.
    @import("hashes").ensureOpenSslReady();

    state.gpa = init.arena.allocator();
    state.io = init.io;

    var stdout_buf: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buf);
    defer stdout_writer.flush() catch {};
    state.out = &stdout_writer.interface;

    const argv = try init.minimal.args.toSlice(state.gpa);

    const cli_result = cli.run(state.gpa, init.io, argv[1..]) catch |err| switch (err) {
        error.InvalidOptions => {
            try stdout_writer.interface.flush();
            std.process.exit(1);
        },
        else => return err,
    };
    state.syntax_check = cli_result.syntax_check;

    front.fend_translation_unit_init(onQueryComplete);
    defer front.fend_translation_unit_cleanup();

    switch (cli_result.input) {
        .query => |q| try compileString("<query>", q),
        .file => |p| try compileFile(p),
        .stdin => try compileStdin(),
    }

    try stdout_writer.interface.flush();
    if (state.had_error) std.process.exit(1);
}

/// Compile (and optionally interpret) one query AST handed up from the parser.
/// Extracted so unit tests can drive the same path without going through `main`.
fn handleQueryAst(ast: ?*c.fend_node_t) void {
    const root = ast orelse return;
    // Grammar may still hand us an AST after semantic lyyerror (e.g. undefined id).
    if (front.fend_error_count != 0) return;

    var arena = std.heap.ArenaAllocator.init(state.gpa);
    defer arena.deinit();

    const plan_root = compile.compileQuery(arena.allocator(), root) catch |err| {
        _ = diag.report(diag.messageForCompile(err));
        state.had_error = true;
        return;
    };
    if (state.syntax_check) return;

    const ctx: interpret.Ctx = .{
        .allocator = arena.allocator(),
        .io = state.io,
        .out = state.writer(),
    };
    interpret.run(ctx, plan_root) catch |err| {
        _ = diag.report(diag.messageForRuntime(err));
        state.had_error = true;
    };
}

fn onQueryComplete(ast: ?*c.fend_node_t) callconv(.c) void {
    handleQueryAst(ast);
}

fn compileString(name: []const u8, text: []const u8) !void {
    state.source_name = name;
    state.source_text = text;
    diag.clearLast();

    const result = try front.parseQuery(text, false);
    if (!front.parseOk(result)) {
        try state.writer().print(
            "Compilation failed. {d} errors occurred during compilation\n",
            .{front.fend_error_count},
        );
        state.had_error = true;
    }
}

fn compileFile(path: []const u8) !void {
    const contents = std.Io.Dir.cwd().readFileAlloc(state.io, path, state.gpa, .unlimited) catch |e| {
        try state.writer().print("Cannot read file: {s}: {}\n", .{ path, e });
        state.had_error = true;
        return;
    };
    defer state.gpa.free(contents);
    try compileString(path, contents);
}

fn compileStdin() !void {
    var buf: [16 * 1024]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(state.io, &buf);
    var mem: std.Io.Writer.Allocating = .init(state.gpa);
    defer mem.deinit();
    _ = try stdin_reader.interface.streamRemaining(&mem.writer);
    try compileString("<stdin>", mem.written());
}

test {
    _ = front;
    _ = cli;
    _ = diag;
    _ = @import("value.zig");
    _ = @import("expr.zig");
    _ = @import("plan.zig");
    _ = @import("props.zig");
    _ = @import("method.zig");
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
    state.gpa = std.testing.allocator;
    state.io = std.testing.io;
    syntax_out_writer = .fixed(&syntax_out_buf);
    state.out = &syntax_out_writer;
    state.had_error = false;
    diag.clearLast();
}

fn parseWithHandle(query: []const u8) !void {
    state.source_name = "<query>";
    state.source_text = query;

    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);

    front.fend_translation_unit_init(onQueryComplete);
    defer front.fend_translation_unit_cleanup();

    _ = try front.parseQuery(query, false);
}

test "syntax-check skips interpret for missing file" {
    setupSyntaxTest();
    state.syntax_check = true;
    defer state.syntax_check = false;

    try parseWithHandle("from file f in '/definitely-missing-l2h-syntax-check' select f.size;");

    try std.testing.expect(!state.had_error);
    try std.testing.expectEqualStrings("", std.Io.Writer.buffered(&syntax_out_writer));
}

test "without syntax-check missing file fails at runtime" {
    setupSyntaxTest();
    state.syntax_check = false;

    try parseWithHandle("from file f in '/definitely-missing-l2h-syntax-check' select f.size;");

    try std.testing.expect(state.had_error);
}

test "syntax-check still reports compile errors" {
    setupSyntaxTest();
    state.syntax_check = true;
    defer state.syntax_check = false;

    try parseWithHandle("from string s in 'a' select s.no_such_prop;");

    try std.testing.expect(state.had_error);
}
