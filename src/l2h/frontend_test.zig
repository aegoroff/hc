//! Compile success/failure for the l2h query
//! language, driven through the bison/flex parser (the same C parser the Zig
//! frontend hooks via the fend_on_* callbacks).
//!
//! `compile(q)` scan the query, run
//! yyparse, and report success iff yyparse returns 0 and no semantic error
//! incremented fend_error_count. A no-op on-query-complete callback is registered
//! so the AST is parsed but not
//! executed. Grammar diagnostics go to stderr and never touch the test IPC.

const std = @import("std");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");
const diag = @import("diag.zig");
const test_stderr = @import("test_stderr.zig");

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

/// Capture `diag.Reported` from the `reportParse` path (void return).
var capture_buf: [768]u8 = undefined;
var capture_len: usize = 0;

fn captureOnReported(r: diag.Reported) void {
    const n = @min(r.message.len, capture_buf.len);
    @memcpy(capture_buf[0..n], r.message[0..n]);
    capture_len = n;
}

fn installCapture() void {
    capture_len = 0;
    diag.setOnReported(captureOnReported);
}

fn capturedMessage() []const u8 {
    return capture_buf[0..capture_len];
}

/// Returns true iff `q` compiles cleanly (no syntax/semantic errors).
///
/// Scans the query, runs yyparse, succeeds iff yyparse returns 0 and
/// fend_error_count stayed 0. The scan buffer is NOT popped so yylineno
/// remains observable afterwards.
///
/// stderr is muted around the parse: intentional failure scenarios emit
/// fehler diagnostics (via `std.debug.print`) that are expected and would
/// otherwise clutter the build output / surface as a misleading "failed
/// command" diagnostic. We only check the compile() return value.
fn compile(q: []const u8) bool {
    state.source_name = "<query>";
    state.source_text = q;

    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);

    const result = front.parseQuery(q) catch return false;
    return front.parseOk(result);
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
    // Arrange
    const q = "from file x in 'dfg' select x.md5";
    // Act
    try expectFailure(q);
    // Assert
}

test "SynErr_UnclosedString_Fail" {
    // Arrange
    const q = "from file x in 'dfg select x.md5;";
    // Act
    try expectFailure(q);
    // Assert
}

test "SynErr_SeveralLineQWithoutSemicolon_Fail" {
    // Arrange
    const q = "from file x in\n 'dfg'\n select x.md5";
    // Act
    try expectFailure(q);
    // Assert
}

test "SynErr_InvalidStart_Fail" {
    // Arrange
    const q = "select x.md4 from file x in 'dfg' select x.md5;";
    // Act
    try expectFailure(q);
    // Assert
}

test "SynErr_UndefinedVariable_Fail" {
    // Arrange
    const q = "from file x in 'dfg' select y.md5;";
    // Act
    try expectFailure(q);
    // Assert
}

// Several-line query without semicolon must advance yylineno to the last line.
test "SynErr_SeveralLineQWithoutSemicolon_AdvancesLineNo" {
    // Arrange
    setup();
    front.fend_translation_unit_init(NoOp.cb);
    defer front.fend_translation_unit_cleanup();
    // Act
    _ = compile("from file x in\n 'dfg'\n select x.md5");
    // Assert
    try std.testing.expectEqual(@as(c_int, 3), c.yylineno);
}

// --- valid queries (COMPILE_SUCCESS) ---------------------------------------

test "Select_SingleObjectProp_Success" {
    // Arrange
    const q = "from file x in 'dfg' select x.md5;";
    // Act
    // Assert
    try expectSuccess(q);
}

test "Select_ManyStringQuery_Success" {
    // Arrange
    const q = "from file x in \n'dfg' \nselect x.md5;";
    // Act
    // Assert
    try expectSuccess(q);
}

test "Select_ManyPropInNewDynamicType_Success" {
    // Arrange
    const q = "from file x in 'dfg' select { x.md5, x.md2 };";
    // Act
    // Assert
    try expectSuccess(q);
}

test "Select_MethodWithoutParamsInSelectClause_Success" {
    // Arrange
    const q = "from file x in 'dfg' select x.m();";
    // Act
    // Assert
    try expectSuccess(q);
}

test "Select_MethodOneParamInSelectClause_Success" {
    // Arrange
    const q = "from file x in 'dfg' select x.m(1);";
    // Act
    // Assert
    try expectSuccess(q);
}

test "Select_MethodManyParamsInSelectClause_Success" {
    // Arrange
    const q = "from file x in 'dfg' select x.m(1, '123');";
    // Act
    // Assert
    try expectSuccess(q);
}

// into / join: parse + identifier scope (execution needs AST→plan compiler).
test "SelectInto_CorrectSyntax_Success" {
    // Arrange
    const q = "from file x in 'dfg' select x.md5 into h select h;";
    // Act
    // Assert
    try expectSuccess(q);
}

test "Join_CorrectSyntax_Success" {
    // Arrange
    const q = "from string a in 'abc' join string b in 'abc' on a.md5 equals b.md5 select a.md5;";
    // Act
    // Assert
    try expectSuccess(q);
}

test "RestoreString_FromHash_Success" {
    // Arrange
    const q = "from hash x in '202CB962AC59075B964B07152D234B70' select x.md5;";
    // Act
    // Assert
    try expectSuccess(q);
}

test "CreateHash_FromString_Success" {
    // Arrange
    const q = "from string x in '123' select x.md5;";
    // Act
    // Assert
    try expectSuccess(q);
}

test "MultipleQueries_SemicolonSeparated_Success" {
    // Arrange
    const q =
        \\from string s in '123' select s.sha1;
        \\from hash h in '40bd001563085fc35165329ea1ff5c5ecbdbbeef' select h.sha1;
    ;
    // Act
    // Assert
    try expectSuccess(q);
}

test "CreateHash_FromDir_Success" {
    // Arrange — ordinary strings are raw, so one `\` in the query is one path char
    const q = "from dir x in 'D:\\' select x.sha1;";
    // Act
    // Assert
    try expectSuccess(q);
}

test "StringLiteral_HexEscapes_Success" {
    const q = "from string x in b\"\\xDE\\xAD\\xBE\\xEF\" select x.md5;";
    try expectSuccess(q);
}

test "StringLiteral_ByteSingleQuotes_Success" {
    const q = "from string x in b'\\xef\\xbb\\xbf' select x.md5;";
    try expectSuccess(q);
}

test "Comment_CommentAndQuerString_Success" {
    // Arrange
    const q = "# test\r\nfrom string x in '123' select x.md5;";
    // Act
    // Assert
    try expectSuccess(q);
}

test "Comment_OnlyComment_Success" {
    // Arrange
    const q = "# test";
    // Act
    // Assert
    try expectSuccess(q);
}

test "parse error reports syntax text" {
    setup();
    diag.clearLast();
    front.fend_translation_unit_init(NoOp.cb);
    defer front.fend_translation_unit_cleanup();

    installCapture();
    defer diag.setOnReported(null);

    state.source_name = "<query>";
    state.source_text = "from string s in";

    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);

    const result = try front.parseQuery(state.source_text);
    try std.testing.expect(!front.parseOk(result));
    try std.testing.expect(std.mem.indexOf(u8, capturedMessage(), "syntax error") != null);
}

test "undefined property receiver reports identifier undefined" {
    setup();
    diag.clearLast();
    front.fend_translation_unit_init(NoOp.cb);
    defer front.fend_translation_unit_cleanup();

    installCapture();
    defer diag.setOnReported(null);

    state.source_name = "<query>";
    state.source_text = "from string s in 'a' select x.md5;";

    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);

    const result = try front.parseQuery(state.source_text);
    try std.testing.expect(!front.parseOk(result));
    try std.testing.expectEqualStrings("identifier x undefined", capturedMessage());
}
