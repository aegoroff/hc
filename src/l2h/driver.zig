//! Shared l2h driver: CLI → parse → compile → optional interpret.
//!
//! Used by `main.zig` (process entry) and `fuzz.zig` (syntax fuzzing). Kept
//! separate from `main.zig` so the fuzz test binary does not pull in main's
//! unit tests.

const std = @import("std");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");
const cli = @import("cli.zig");
const diag = @import("diag.zig");

/// Parse CLI (`argv` without program name), then parse/compile (and interpret
/// unless `-n` / `--syntax-check`).
pub fn run(
    gpa: std.mem.Allocator,
    out: *std.Io.Writer,
    io: std.Io,
    argv: []const [:0]const u8,
) !void {
    state.gpa = gpa;
    state.io = io;
    state.out = out;
    state.had_error = false;
    state.syntax_check = false;

    const cli_result = cli.run(gpa, io, argv) catch |err| switch (err) {
        error.InvalidOptions => {
            state.had_error = true;
            return;
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
}

fn onQueryComplete(ast: ?*c.fend_node_t) callconv(.c) void {
    _ = front.handleQueryAst(ast);
}

fn compileString(name: []const u8, text: []const u8) !void {
    state.source_name = name;
    state.source_text = text;
    diag.clearLast();

    const result = try front.parseQuery(text);
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
