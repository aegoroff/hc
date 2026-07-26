const std = @import("std");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");
const backend = @import("backend.zig");

// l2h (linq2hash) Zig driver.
//
// Wires the runtime context (Juicy Main), feeds query text into the generated
// bison/flex parser (yy_scan_string + yyparse), and routes each completed AST
// to the backend via the fend_query_cleanup -> onQueryComplete callback. The
// parser/lexer themselves are untouched C; all semantics live in
// frontend/backend/processor.zig.

pub fn main(init: std.process.Init) !void {
    state.gpa = init.arena.allocator();
    state.io = init.io;

    var stdout_buf: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buf);
    defer stdout_writer.flush() catch {};
    state.out = &stdout_writer.interface;

    const argv = try init.minimal.args.toSlice(state.gpa);

    var query_text: ?[]const u8 = null;
    var file_path: ?[]const u8 = null;
    var show_help = false;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "-q") or std.mem.eql(u8, a, "--query")) {
            if (i + 1 < argv.len) {
                query_text = argv[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, a, "-f") or std.mem.eql(u8, a, "--file")) {
            if (i + 1 < argv.len) {
                file_path = argv[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            show_help = true;
        }
    }

    if (show_help) {
        try printUsage(state.writer());
        return;
    }

    front.fend_translation_unit_init(onQueryComplete);
    defer front.fend_translation_unit_cleanup();

    if (query_text) |q| {
        try compileString(q);
    } else if (file_path) |p| {
        try compileFile(p);
    } else {
        try compileStdin();
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

fn printUsage(w: *std.Io.Writer) !void {
    try w.print(
        \\linq2hash (l2h) - hash query language
        \\
        \\Usage:
        \\  l2h -q "<query>"      query text from the command line
        \\  l2h -f <file>         query from one or more files
        \\  l2h                   read query from standard input
        \\
        \\Each query ends with a semicolon, e.g.
        \\  from string s in "abc" select s.tiger;
        \\
    , .{});
}

test {
    // Pull in the semantics modules so their tests run under `zig build test`.
    _ = front;
    _ = backend;
    _ = @import("processor.zig");
    // GoogleTest parity suites (co-located in src/zig/l2h/*.zig) for the
    // frontend parser, backend tree traversal, and processor regex match. These
    // live under l2h/ (not src/zig/tests/) because the l2h module root is
    // main.zig and Zig forbids @import of files outside that module path; they
    // reuse the wired c/re/lib/hashes/modes deps via the l2h module graph.
    _ = @import("frontend_test.zig");
    _ = @import("tree_test.zig");
    _ = @import("processor_test.zig");
}
