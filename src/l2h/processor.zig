const std = @import("std");
const c = @import("c");
const lib = @import("lib");
const hashes = @import("hashes");
const modes = @import("modes");
const re = @import("re");
const state = @import("state.zig");
const backend = @import("backend.zig");

// Zig port of src/l2h/processor.c.
//
// Executes the triple program produced by backend.zig. The stack of `source_t`
// records models the C apr_array; prproc_on_select walks it backward to decide
// which hash algorithm to apply to which data source, then dispatches to the
// reusable modes module (string / file / dir / hash-reversal runners).

pub const InstrType = enum(u8) {
    file_decl = 0,
    dir_decl = 1,
    string_decl = 2,
    hash_decl = 3,
    string_def = 4,
    prop_call = 5,
    hash_definition = 6,
};

pub const Source = struct {
    type: InstrType,
    name: ?[*c]u8 = null,
    value: ?[*c]u8 = null,
};

const Triple = backend.Triple;

var alloc: std.mem.Allocator = undefined;
var sources: std.ArrayListUnmanaged(*Source) = .empty;

fn span(s: ?[*c]u8) []const u8 {
    if (s == null) return "";
    return std.mem.span(@as([*:0]u8, @ptrCast(s)));
}

fn isType(t: c.type_def_t, want: c_int) bool {
    return @as(c_int, @intCast(t)) == want;
}

// --- lifecycle (port of proc_init / proc_complete) ------------------------

pub fn init(arena: std.mem.Allocator) void {
    alloc = arena;
    sources = .empty;
    // hsh_initialize_hashes(proc_pool): the Zig hashes table is comptime, so no
    // runtime initialization is required. pcre_context is created per-match.
}

pub fn complete() void {
    // APR pools are released by the caller (backend arena). Nothing to free.
}

// --- triple dispatch (port of proc_run + proc_processors[]) ---------------

pub fn run(program: []const *Triple) void {
    for (program) |triple| {
        switch (triple.code) {
            .from => onFrom(triple),
            .def => onDef(triple),
            .select => onSelect(triple),
            .property => onProperty(triple),
            .string => onString(triple),
            // let_, call, type_, usage, integer, *_rel, relation, into,
            // query_continuation: no processor in the C original (NULL slots).
            else => {},
        }
    }
}

fn pushSource(s: Source) void {
    const ptr = alloc.create(Source) catch return;
    ptr.* = s;
    sources.append(alloc, ptr) catch {};
}

// --- def: a typed variable declaration (port of prproc_on_def) -----------

fn onDef(triple: *Triple) void {
    const op1 = triple.op1 orelse return;
    const name = triple.op2 orelse return;
    // op1 holds the declared type_def_* tag (string/file/dir/custom).
    const t = op1.type;
    if (isType(t, c.type_def_string)) {
        pushSource(.{ .type = .string_decl, .name = name.string });
    } else if (isType(t, c.type_def_file)) {
        pushSource(.{ .type = .file_decl, .name = name.string });
    } else if (isType(t, c.type_def_dir)) {
        pushSource(.{ .type = .dir_decl, .name = name.string });
    } else if (isType(t, c.type_def_custom)) {
        pushSource(.{ .type = .hash_decl, .name = name.string });
    }
}

// --- string literal operand (port of prproc_on_string) -------------------

fn onString(triple: *Triple) void {
    const op1 = triple.op1 orelse return;
    pushSource(.{ .type = .string_def, .value = op1.string });
}

// --- from: bind datasource to its declaration (port of prproc_on_from) ---

fn onFrom(triple: *Triple) void {
    const op1 = triple.op1 orelse return;
    const op2 = triple.op2 orelse return;
    const to_idx: usize = @intCast(op1.number);
    const from_idx: usize = @intCast(op2.number);
    if (to_idx >= sources.items.len or from_idx >= sources.items.len) return;

    const to = sources.items[to_idx];
    const from = sources.items[from_idx];

    if (from.value == null) return;

    if (to.type != .hash_decl) {
        // Bind the datasource into the declaration; drop the def source.
        to.value = from.value;
        _ = sources.pop();
    } else {
        // Custom hash type: record a hash_definition (algo + digest).
        _ = sources.pop();
        pushSource(.{
            .type = .hash_definition,
            .name = to.value,
            .value = from.value,
        });
    }
}

// --- property call: var.algo (port of prproc_on_property) ----------------

fn onProperty(triple: *Triple) void {
    const op1 = triple.op1 orelse return;
    const op2 = triple.op2 orelse return;
    pushSource(.{ .type = .prop_call, .name = op1.string, .value = op2.string });
}

// --- select: compute hashes (port of prproc_on_select) -------------------

const hash_value_to_restore = "digest";

fn onSelect(_: *Triple) void {
    var properties: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer properties.deinit(alloc);

    var i: usize = sources.items.len;
    while (i > 0) {
        i -= 1;
        const instr = sources.items[i];

        switch (instr.type) {
            .prop_call => {
                const algo = span(instr.value);
                if (hashes.getHash(algo) != null) {
                    properties.put(alloc, span(instr.name), algo) catch {};
                }
            },
            .hash_definition => {
                properties.put(alloc, hash_value_to_restore, span(instr.value)) catch {};
            },
            .string_def => {
                // Dynamic (untyped) source: take the first requested algorithm.
                if (firstValue(&properties)) |algo| {
                    const val = span(instr.value);
                    switch (classifyPath(val)) {
                        .dir => calculateDir(algo, val),
                        .file => calculateFile(algo, val),
                        .none => calculateString(algo, val),
                    }
                }
            },
            .string_decl => {
                if (properties.get(span(instr.name))) |algo| {
                    calculateString(algo, span(instr.value));
                }
            },
            .hash_decl => {
                const algo = properties.get(span(instr.name));
                const digest = properties.get(hash_value_to_restore);
                if (algo != null and digest != null) {
                    calculateHash(algo.?, digest.?);
                }
            },
            .file_decl => {
                if (properties.get(span(instr.name))) |algo| {
                    calculateFile(algo, span(instr.value));
                }
            },
            .dir_decl => {
                if (properties.get(span(instr.name))) |algo| {
                    calculateDir(algo, span(instr.value));
                }
            },
        }
    }
}

fn firstValue(map: *std.StringHashMapUnmanaged([]const u8)) ?[]const u8 {
    var it = map.valueIterator();
    if (it.next()) |v| return v.*;
    return null;
}

const PathKind = enum { dir, file, none };

fn classifyPath(path: []const u8) PathKind {
    if (std.Io.Dir.cwd().openDir(state.io, path, .{})) |d| {
        d.close(state.io);
        return .dir;
    } else |_| {}
    if (std.Io.Dir.cwd().openFile(state.io, path, .{ .mode = .read_only })) |f| {
        f.close(state.io);
        return .file;
    } else |_| {}
    return .none;
}

// --- hash computation (port of prproc_calculate_*) -----------------------

fn env() modes.RunEnv {
    return .{ .io = state.io, .allocator = alloc, .out = state.writer() };
}

fn calculateString(algo: []const u8, string: []const u8) void {
    const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = algo };
    var sctx: modes.StringCtx = .{ .builtin = &bctx, .string = string };
    modes.builtinRun(modes.StringCtx, &bctx, &sctx, modes.strRun, env()) catch |e| {
        reportRunError("string", algo, e);
    };
}

fn calculateFile(algo: []const u8, path: []const u8) void {
    const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = algo };
    var fctx: modes.FileCtx = .{ .builtin = &bctx, .file_path = path };
    modes.builtinRun(modes.FileCtx, &bctx, &fctx, modes.fileRun, env()) catch |e| {
        reportRunError("file", algo, e);
    };
}

fn calculateDir(algo: []const u8, path: []const u8) void {
    const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = algo };
    var dctx: modes.DirCtx = .{ .builtin = &bctx, .dir_path = path };
    modes.builtinRun(modes.DirCtx, &bctx, &dctx, modes.dirRun, env()) catch |e| {
        reportRunError("dir", algo, e);
    };
}

fn calculateHash(algo: []const u8, digest: []const u8) void {
    const bctx = modes.BuiltinCtx{ .is_print_low_case = true, .hash_algorithm = algo };
    var hctx: modes.HashCtx = .{ .builtin = &bctx, .hash = digest };
    modes.builtinRun(modes.HashCtx, &bctx, &hctx, modes.hashRun, env()) catch |e| {
        reportRunError("hash", algo, e);
    };
}

fn reportRunError(ctx: []const u8, algo: []const u8, err: anyerror) void {
    const w = state.writer();
    w.print("l2h: {s} hash '{s}' failed: {s}\n", .{ ctx, algo, @errorName(err) }) catch {};
}

// --- PCRE2 match (port of proc_match_re) ----------------------------------
//
// Invoked by the relational (~ / !~) operators. The current triple pipeline
// does not route WHERE clauses here (the C proc_processors slots for the
// relation opcodes are NULL), so this is dormant in the running pipeline.
//
// pcre2.h exposes its API as width-suffixed macros (pcre2_compile ->
// pcre2_compile_8) which translate-c cannot resolve (they expand to
// @compileError). The underlying _8 functions DO translate, so the binding
// below calls those directly and passes NULL for the (otherwise default)
// compile/match contexts.

comptime {
    _ = re; // keep the pcre2 translate-c dependency reachable
}

pub fn matchRe(pattern: []const u8, subject: []const u8) bool {
    var errnumber: c_int = 0;
    var erroffset: usize = 0;
    const compiled = re.pcre2_compile_8(pattern.ptr, pattern.len, 0, &errnumber, &erroffset, null) orelse return false;
    defer _ = re.pcre2_code_free_8(compiled);

    const match_data = re.pcre2_match_data_create_from_pattern_8(compiled, null) orelse return false;
    defer _ = re.pcre2_match_data_free_8(match_data);

    var flags: u32 = re.PCRE2_NOTEMPTY;
    // C proc_match_re derives PCRE2_NOTBOL/NOTEOL from the *subject* (strchr),
    // not the pattern — anchor flags reflect whether the subject is anchored.
    if (std.mem.indexOfScalar(u8, subject, '^') == null) flags |= re.PCRE2_NOTBOL;
    if (std.mem.indexOfScalar(u8, subject, '$') == null) flags |= re.PCRE2_NOTEOL;

    const rc = re.pcre2_match_8(compiled, subject.ptr, subject.len, 0, flags, match_data, null);
    return rc >= 0;
}

test "onDef/onString/onFrom string binding computes a hash" {
    state.gpa = std.testing.allocator;
    state.io = std.testing.io;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var buf: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    state.out = &writer;

    init(arena.allocator());
    defer complete();

    // Model: from string s in "abc" select s.tiger
    pushSource(.{ .type = .string_decl, .name = @constCast("s".ptr) }); // def s
    pushSource(.{ .type = .string_def, .value = @constCast("abc".ptr) }); // "abc"

    // from: bind "abc" into the string_decl, dropping the def.
    var from_op1 = backend.OpValue{ .number = 0 };
    var from_op2 = backend.OpValue{ .number = 1 };
    var from_triple = backend.Triple{ .code = .from, .op1 = &from_op1, .op2 = &from_op2 };
    onFrom(&from_triple);

    // property s.tiger
    var prop_op1 = backend.OpValue{ .string = @constCast("s".ptr) };
    var prop_op2 = backend.OpValue{ .string = @constCast("tiger".ptr) };
    var prop_triple = backend.Triple{ .code = .property, .op1 = &prop_op1, .op2 = &prop_op2 };
    onProperty(&prop_triple);

    var sel_triple = backend.Triple{ .code = .select };
    onSelect(&sel_triple);

    // tiger("abc") = 2aab1484e8c158f2bfb8c5ff41b57a525129131c957b5f93
    try std.testing.expectEqualStrings(
        "2aab1484e8c158f2bfb8c5ff41b57a525129131c957b5f93",
        std.mem.trim(u8, std.Io.Writer.buffered(&writer), "\n"),
    );
}

test "onDef type_def_custom pushes hash_decl" {
    state.gpa = std.testing.allocator;
    state.io = std.testing.io;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    init(arena.allocator());
    defer complete();

    var type_op = backend.OpValue{ .type = @as(c.type_def_t, @intCast(c.type_def_custom)) };
    var name_op = backend.OpValue{ .string = @constCast("x".ptr) };
    var def_triple = backend.Triple{ .code = .def, .op1 = &type_op, .op2 = &name_op };
    onDef(&def_triple);

    try std.testing.expectEqual(@as(usize, 1), sources.items.len);
    try std.testing.expectEqual(InstrType.hash_decl, sources.items[0].type);
    try std.testing.expectEqualStrings("x", span(sources.items[0].name));
}

test "hash_decl select restores empty-string md5 digest" {
    // Regression for custom-type def: without keeping type_def_custom in op1,
    // this path hashed the digest text as a string instead of restoring.
    // Empty-string MD5 is found immediately (no probe / crack loop).
    state.gpa = std.testing.allocator;
    state.io = std.testing.io;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var buf: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    state.out = &writer;

    init(arena.allocator());
    defer complete();

    var type_op = backend.OpValue{ .type = @as(c.type_def_t, @intCast(c.type_def_custom)) };
    var name_op = backend.OpValue{ .string = @constCast("x".ptr) };
    var def_triple = backend.Triple{ .code = .def, .op1 = &type_op, .op2 = &name_op };
    onDef(&def_triple);

    pushSource(.{ .type = .string_def, .value = @constCast("d41d8cd98f00b204e9800998ecf8427e".ptr) });

    var from_op1 = backend.OpValue{ .number = 0 };
    var from_op2 = backend.OpValue{ .number = 1 };
    var from_triple = backend.Triple{ .code = .from, .op1 = &from_op1, .op2 = &from_op2 };
    onFrom(&from_triple);

    var prop_op1 = backend.OpValue{ .string = @constCast("x".ptr) };
    var prop_op2 = backend.OpValue{ .string = @constCast("md5".ptr) };
    var prop_triple = backend.Triple{ .code = .property, .op1 = &prop_op1, .op2 = &prop_op2 };
    onProperty(&prop_triple);

    var sel_triple = backend.Triple{ .code = .select };
    onSelect(&sel_triple);

    const out = std.Io.Writer.buffered(&writer);
    try std.testing.expect(std.mem.indexOf(u8, out, "Initial string is: Empty string") != null);
}
