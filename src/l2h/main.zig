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
// Parses queries via bison/flex, compiles the AST to QueryPlan, then executes the
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
        .query => |q| try compileString("<query>", q),
        .file => |p| try compileFile(p),
        .stdin => try compileStdin(),
    }

    try stdout_writer.interface.flush();
    if (state.had_error) std.process.exit(1);
}

fn onQueryComplete(ast: ?*c.fend_node_t) callconv(.c) void {
    const root = ast orelse return;
    // Grammar may still hand us an AST after semantic lyyerror (e.g. undefined id).
    if (front.fend_error_count != 0) return;

    var arena = std.heap.ArenaAllocator.init(state.gpa);
    defer arena.deinit();

    const plan_root = compile.compileQuery(arena.allocator(), root) catch |err| {
        diag.report(diag.messageForCompile(err));
        state.had_error = true;
        return;
    };
    const ctx: interpret.Ctx = .{
        .allocator = arena.allocator(),
        .io = state.io,
        .out = state.writer(),
    };
    interpret.run(ctx, &plan_root) catch |err| {
        diag.report(diag.messageForRuntime(err));
        state.had_error = true;
    };
}

fn compileString(name: []const u8, text: []const u8) !void {
    state.source_name = name;
    state.source_text = text;
    diag.clearLast();

    const z = try state.gpa.dupeSentinel(u8, text, 0);
    defer state.gpa.free(z);

    _ = c.yy_scan_string(z.ptr);
    defer _ = c.yypop_buffer_state();

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
    _ = @import("interpret.zig");
    _ = @import("match_re.zig");
    _ = @import("frontend_test.zig");
    _ = @import("compile_test.zig");
}
