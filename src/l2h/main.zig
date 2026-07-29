const std = @import("std");
const builtin = @import("builtin");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");
const backend = @import("backend.zig");
const cli = @import("cli.zig");

// l2h (linq2hash) Zig driver.
//
// Wires the runtime context (Juicy Main), feeds query text into the generated
// bison/flex parser (yy_scan_string + yyparse), and routes each completed AST
// to the backend via the fend_query_cleanup -> onQueryComplete callback. The
// parser/lexer themselves are untouched C; all semantics live in
// frontend/backend/processor.zig.

const utf8_console = if (builtin.os.tag == .windows)
    @import("utf8_console.zig")
else
    struct {
        pub fn setupConsole() void {}
    };

pub fn main(init: std.process.Init) !void {
    utf8_console.setupConsole();

    state.gpa = init.arena.allocator();
    state.io = init.io;

    var stdout_buf: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buf);
    defer stdout_writer.flush() catch {};
    state.out = &stdout_writer.interface;

    const argv = try init.minimal.args.toSlice(state.gpa);

    const outcome = try cli.run(state.gpa, init.io, argv[1..]);
    const input = switch (outcome) {
        .ok => |inp| inp,
        .invalid_options => {
            try stdout_writer.interface.flush();
            std.process.exit(1);
        },
    };

    front.fend_translation_unit_init(onQueryComplete);
    defer front.fend_translation_unit_cleanup();

    switch (input) {
        .query => |q| try compileString(q),
        .file => |p| try compileFile(p),
        .stdin => try compileStdin(),
    }
}

/// Grammar callback: a full query AST is ready. Evaluate it on a throwaway
/// backend arena; the AST itself lives in the frontend query arena and stays
/// valid for the duration of this call.
fn onQueryComplete(ast: ?*c.fend_node_t) callconv(.c) void {
    if (ast == null) return;
    var arena = std.heap.ArenaAllocator.init(state.gpa);
    defer arena.deinit();
    backend.processQuery(ast, arena.allocator());
}

fn compileString(text: []const u8) !void {
    const z = try state.gpa.dupeSentinel(u8, text, 0);
    defer state.gpa.free(z);

    _ = c.yy_scan_string(z.ptr);
    defer _ = c.yypop_buffer_state();

    // Initialize location tracking before parsing (mirrors grok compileFile).
    c.yyset_lineno(1);
    c.yycolumn = 1;
    c.yylloc = .{
        .first_line = 1,
        .first_column = 1,
        .last_line = 1,
        .last_column = 1,
    };
    front.fend_error_count = 0;

    const result = c.yyparse();
    if (front.fend_error_count != 0 or result != 0) {
        try state.writer().print(
            "Compilation failed. {d} errors occurred during compilation\n",
            .{front.fend_error_count},
        );
    }
}

fn compileFile(path: []const u8) !void {
    const contents = std.Io.Dir.cwd().readFileAlloc(state.io, path, state.gpa, .unlimited) catch |e| {
        try state.writer().print("Cannot read file: {s}: {}\n", .{ path, e });
        return;
    };
    defer state.gpa.free(contents);
    try compileString(contents);
}

fn compileStdin() !void {
    var buf: [16 * 1024]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(state.io, &buf);
    var mem: std.Io.Writer.Allocating = .init(state.gpa);
    defer mem.deinit();
    _ = try stdin_reader.interface.streamRemaining(&mem.writer);
    try compileString(mem.written());
}

test {
    // Pull in the semantics modules so their tests run under `zig build test`.
    _ = front;
    _ = backend;
    _ = @import("processor.zig");
    _ = cli;
    // GoogleTest parity suites (co-located in src/l2h/*.zig) for the
    // frontend parser, backend tree traversal, and processor regex match. These
    // live under l2h/ (not src/tests/) because the l2h module root is
    // main.zig and Zig forbids @import of files outside that module path; they
    // reuse the wired c/re/lib/hashes/modes deps via the l2h module graph.
    _ = @import("frontend_test.zig");
    _ = @import("tree_test.zig");
    _ = @import("processor_test.zig");
}
