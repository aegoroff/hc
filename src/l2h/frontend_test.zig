//! GoogleTest FrontendTest parity: compile success/failure for the l2h query
//! language, driven through the bison/flex parser (the same C parser the Zig
//! frontend hooks via the fend_on_* callbacks).
//!
//! `compile(q)` mirrors FrontendTest.cpp's Compile(): scan the query, run
//! yyparse, and report success iff yyparse returns 0 and no semantic error
//! incremented fend_error_count. A no-op on-query-complete callback is registered
//! (matching the C++ ftest_on_each_query_callback) so the AST is parsed but not
//! executed. Grammar diagnostics go to stderr and never touch the test IPC.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");

const NoOp = struct {
    fn cb(_: ?*c.fend_node_t) callconv(.c) void {}
};

// Module-level writer so state.out stays valid across compile() calls (a
// function-local fixed writer would dangle once setup() returns).
var out_buf: [4096]u8 = undefined;
var out_writer: std.Io.Writer = undefined;

fn setup() void {
    state.gpa = std.testing.allocator;
    state.io = std.testing.io;
    out_writer = .fixed(&out_buf);
    state.out = &out_writer;
}

/// Returns true iff `q` compiles cleanly (no syntax/semantic errors).
///
/// Mirrors FrontendTest.cpp's Compile(): scan the query, run yyparse, succeed
/// iff yyparse returns 0 and fend_error_count stayed 0. The scan buffer is NOT
/// popped (matching the C++ test) so yylineno remains observable afterwards.
///
/// stderr is muted around the parse: the intentional failure scenarios emit
/// grammar diagnostics (lib_fprintf(stderr, ...)) that are expected and would
/// otherwise clutter the build output / surface as a misleading "failed
/// command" diagnostic. We only check the compile() return value.
fn compile(q: []const u8) bool {
    front.fend_error_count = 0;

    const z = state.gpa.dupeSentinel(u8, q, 0) catch return false;
    defer state.gpa.free(z);

    const saved_stderr = muteStderr();
    defer if (saved_stderr >= 0) restoreStderr(saved_stderr);

    _ = c.yy_scan_string(z.ptr);

    c.yyset_lineno(1);
    c.yycolumn = 1;
    c.yylloc = .{
        .first_line = 1,
        .first_column = 1,
        .last_line = 1,
        .last_column = 1,
    };

    const result = c.yyparse();
    return result == 0 and front.fend_error_count == 0;
}

fn muteStderr() c_int {
    // The dup2/close dance is POSIX-only (std.c.open's flag type is invalid
    // under the x86_64_win calling convention). On Windows the early return is
    // comptime-taken, so the POSIX body below is never analyzed; grammar
    // diagnostics from intentional-failure queries then leak to stderr (cosmetic
    // — the tests assert on the compile() return value, not stderr).
    if (builtin.os.tag == .windows) return -1;
    const null_fd = std.c.open("/dev/null", .{ .ACCMODE = .WRONLY });
    if (null_fd < 0) return -1;
    const saved = std.c.dup(std.posix.STDERR_FILENO);
    if (saved < 0) {
        _ = std.c.close(null_fd);
        return -1;
    }
    if (std.c.dup2(null_fd, std.posix.STDERR_FILENO) < 0) {
        _ = std.c.close(saved);
        _ = std.c.close(null_fd);
        return -1;
    }
    _ = std.c.close(null_fd);
    return saved;
}

fn restoreStderr(saved: c_int) void {
    if (builtin.os.tag == .windows) return;
    _ = std.c.dup2(saved, std.posix.STDERR_FILENO);
    _ = std.c.close(saved);
}


fn expectSuccess(q: []const u8) !void {
    setup();
    front.fend_translation_unit_init(NoOp.cb);
    defer front.fend_translation_unit_cleanup();
    try std.testing.expect(compile(q));
}

fn expectFailure(q: []const u8) !void {
    setup();
    front.fend_translation_unit_init(NoOp.cb);
    defer front.fend_translation_unit_cleanup();
    try std.testing.expect(!compile(q));
}

// --- syntax / semantic failures (COMPILE_FAIL) -----------------------------

test "SynErr_NoSemicolon_Fail" {
    try expectFailure("from file x in 'dfg' select x.md5");
}

test "SynErr_UnclosedString_Fail" {
    try expectFailure("from file x in 'dfg select x.md5;");
}

test "SynErr_SeveralLineQWithoutSemicolon_Fail" {
    try expectFailure("from file x in\n 'dfg'\n select x.md5");
}

test "SynErr_InvalidStart_Fail" {
    try expectFailure("select x.md4 from file x in 'dfg' select x.md5;");
}

test "SynErr_UndefinedVariable_Fail" {
    try expectFailure("from file x in 'dfg' select y.md5;");
}

// Several-line query without semicolon must advance yylineno to the last line.
test "SynErr_SeveralLineQWithoutSemicolon_AdvancesLineNo" {
    setup();
    front.fend_translation_unit_init(NoOp.cb);
    defer front.fend_translation_unit_cleanup();
    _ = compile("from file x in\n 'dfg'\n select x.md5");
    try std.testing.expectEqual(@as(c_int, 3), c.yylineno);
}

// --- valid queries (COMPILE_SUCCESS) ---------------------------------------

test "Select_SingleObjectProp_Success" {
    try expectSuccess("from file x in 'dfg' select x.md5;");
}

test "Select_ManyStringQuery_Success" {
    try expectSuccess("from file x in \n'dfg' \nselect x.md5;");
}

test "Select_ManyPropInNewDynamicType_Success" {
    try expectSuccess("from file x in 'dfg' select { x.md5, x.md2 };");
}

test "Select_MethodWithoutParamsInSelectClause_Success" {
    try expectSuccess("from file x in 'dfg' select x.m();");
}

test "Select_MethodOneParamInSelectClause_Success" {
    try expectSuccess("from file x in 'dfg' select x.m(1);");
}

test "Select_MethodManyParamsInSelectClause_Success" {
    try expectSuccess("from file x in 'dfg' select x.m(1, '123');");
}

// into / join are unimplemented in the grammar (TODO in the C++ suite too):
// both currently fail to compile.
test "SelectInto_CorrectSyntax_NotYetImplemented_Fail" {
    try expectFailure("from file x in 'dfg' select x.md5 into x select x.crc32;");
}

test "Join_CorrectSyntax_NotYetImplemented_Fail" {
    try expectFailure("from string a in x join y in z on a.i equals y.i into gr select a.md5;");
}

test "RestoreString_FromHash_Success" {
    try expectSuccess("from hash x in '202CB962AC59075B964B07152D234B70' select x.md5;");
}

test "CreateHash_FromString_Success" {
    try expectSuccess("from string x in '123' select x.md5;");
}

test "CreateHash_FromDir_Success" {
    try expectSuccess("from dir x in 'D:\\' select x.sha1;");
}

test "Comment_CommentAndQuerString_Success" {
    try expectSuccess("# test\r\nfrom string x in '123' select x.md5;");
}

test "Comment_OnlyComment_Success" {
    try expectSuccess("# test");
}
