const std = @import("std");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");
const compile = @import("compile.zig");
const diag = @import("diag.zig");
const expr = @import("expr.zig");
const test_stderr = @import("test_stderr.zig");

var out_buf: [4096]u8 = undefined;
var out_writer: std.Io.Writer = undefined;

/// Stash `diag.report` return for this module's assertions (not production state).
var run_err_buf: [768]u8 = undefined;
var run_err_len: usize = 0;
var run_span: expr.Span = .{};

/// Result storage for the whole test binary: runQuery hands out stable copies
/// so an earlier result stays valid after later runs overwrite the shared
/// out/err scratch buffers above. Page-backed and never reset — freeing would
/// force deinit plumbing through every test for no bug-detection gain.
var result_arena: ?std.heap.ArenaAllocator = null;

fn resultAlloc() std.mem.Allocator {
    if (result_arena == null) result_arena = .init(std.heap.page_allocator);
    return result_arena.?.allocator();
}

fn setup() void {
    state.gpa = std.testing.allocator;
    state.io = std.testing.io;
    out_writer = .fixed(&out_buf);
    state.out = &out_writer;
}

fn noteReported(r: diag.Reported) void {
    const n = @min(r.message.len, run_err_buf.len);
    @memcpy(run_err_buf[0..n], r.message[0..n]);
    run_err_len = n;
    run_span = r.span;
}

const RunResult = struct {
    out: []const u8,
    err: []const u8,
};

fn runQuery(query: []const u8) !RunResult {
    setup();
    state.source_name = "<query>";
    state.source_text = query;
    state.had_error = false;
    state.syntax_check = false;
    diag.clearLast();
    // Capture parse-time reports too (frontend_test asserts them via the same
    // hook); runtime reports keep flowing through the AST callback as well.
    diag.setOnReported(noteReported);
    defer diag.setOnReported(null);
    run_err_len = 0;
    run_span = .{};
    out_writer = .fixed(&out_buf);

    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);

    const Callback = struct {
        fn cb(ast: ?*c.fend_node_t) callconv(.c) void {
            if (front.handleQueryAst(ast)) |r| noteReported(r);
        }
    };

    front.fend_translation_unit_init(Callback.cb);
    defer front.fend_translation_unit_cleanup();

    _ = try front.parseQuery(query);
    const alloc = resultAlloc();
    return .{
        .out = try alloc.dupe(u8, std.Io.Writer.buffered(&out_writer)),
        .err = try alloc.dupe(u8, run_err_buf[0..run_err_len]),
    };
}

fn tmpQueryPath(allocator: std.mem.Allocator, tmp: anytype) ![]u8 {
    return try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}

/// Join under `tmpQueryPath` with `/` so the result is safe inside l2h `'…'` literals
/// (Windows `path.join` would insert `\`, which is now an escape introducer).
fn tmpFileQueryPath(allocator: std.mem.Allocator, dir_path: []const u8, name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, name });
}

test "compile+run where/select query string" {
    // Arrange
    const query = "from string s in 'abc' where s.size > 0 select s.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run multiple top-level queries" {
    // Arrange — semantics §5: several semicolon-separated queries in one unit
    const query =
        \\from string s in '123' select s.sha1;
        \\from string t in 'abc' select t.md5;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        \\40bd001563085fc35165329ea1ff5c5ecbdbbeef
        \\900150983cd24fb0d6963f7d28e17f72
        \\
    ,
        got.out,
    );
}

test "compile+run script into shared across statements" {
    // Arrange — terminal `into h;` binds script env; next query reads `h`
    const query =
        \\from string s in 'abc' select s.md5 into h;
        \\from string t in 'xyz' where t.md5 != h select t;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("xyz\n", got.out);
}

test "compile+run script into does not print" {
    // Arrange
    const query = "from string s in 'abc' select s.md5 into h;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("", got.out);
}

test "compile+run script group into shared across statements" {
    // Arrange — terminal `group … into g;` binds via ScriptBind(GroupOut); one group → scalar Record
    const query =
        \\from string s in 'abc' group s by s.size into g;
        \\from string t in 'x' select g.key;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("3\n", got.out);
}

test "compile+run multiple queries reuse range id" {
    // Arrange — each query resets identifier scope
    const query =
        \\from string s in '123' select s.sha1;
        \\from string s in 'abc' select s.md5;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        \\40bd001563085fc35165329ea1ff5c5ecbdbbeef
        \\900150983cd24fb0d6963f7d28e17f72
        \\
    ,
        got.out,
    );
}

test "compile+run local range shadows script env without clobbering it" {
    // Arrange — same name in script Env (`into h`) and a later local `from` range.
    // Local shadows for that query; script binding must still be intact afterward (§3.2 / §5).
    const query =
        \\from string s in 'abc' select s.md5 into h;
        \\from string h in 'xyz' select h;
        \\from string t in 'q' select h;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        \\xyz
        \\900150983cd24fb0d6963f7d28e17f72
        \\
    ,
        got.out,
    );
}

test "compile+run from range same as script source uses script then shadows" {
    // Arrange — `from string h in h`: source reads script `h`, then range binds local `h`
    const query =
        \\from string s in 'abc' select s.md5 into h;
        \\from string h in h select h;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run let shadows script env name" {
    // Arrange — `let h = …` shadows script `h` for the row; script value unused in select
    const query =
        \\from string s in 'abc' select s.md5 into h;
        \\from string t in 'x' let h = t select h;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("x\n", got.out);
}

test "compile+run let/into query string" {
    // Arrange
    const query = "from string s in 'abc' let d = s.md5 select d into h select h;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run join/orderby query string" {
    // Arrange
    const query =
        \\from string a in 'bb'
        \\join string b in 'a' on a.size equals b.size
        \\into g
        \\from string x in g
        \\orderby x.size descending
        \\select x;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.out);
}

test "compile+run regex where query string" {
    // Arrange
    const query = "from string s in 'abc123' where s ~ '[0-9]+' select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc123\n", got.out);
}

test "compile+run regex empty match keeps empty string" {
    // Arrange — §5.3: zero-length PCRE2 matches succeed (`^$`, `a*`).
    const anchors = "from string s in '' where s ~ '^$' select s.size;";
    const star = "from string s in '' where s ~ 'a*' select s.size;";
    const star_on_text = "from string s in 'abc' where s ~ 'x*' select s;";

    // Act
    const got_anchors = try runQuery(anchors);
    const got_star = try runQuery(star);
    const got_text = try runQuery(star_on_text);

    // Assert
    try std.testing.expectEqualStrings("", got_anchors.err);
    try std.testing.expectEqualStrings("0\n", got_anchors.out);
    try std.testing.expectEqualStrings("", got_star.err);
    try std.testing.expectEqualStrings("0\n", got_star.out);
    try std.testing.expectEqualStrings("", got_text.err);
    try std.testing.expectEqualStrings("abc\n", got_text.out);
}

test "compile+run not-match operator keeps non-matching rows" {
    // Arrange — §5.3: `!~` is the negation of `~` for String operands.
    const keep = "from string s in 'abc' where s !~ '[0-9]+' select s;";
    const drop = "from string s in 'abc123' where s !~ '[0-9]+' select s;";

    // Act
    const got_keep = try runQuery(keep);
    const got_drop = try runQuery(drop);

    // Assert
    try std.testing.expectEqualStrings("", got_keep.err);
    try std.testing.expectEqualStrings("abc\n", got_keep.out);
    try std.testing.expectEqualStrings("", got_drop.err);
    try std.testing.expectEqualStrings("", got_drop.out);
}

test "compile+run boolean operators and parentheses" {
    // Arrange — §5.2: `&&` / `||` / `!` with grouped predicates. Sizes are
    // fixed per literal so both branches are deterministic.
    const both =
        \\from string s in 'abc'
        \\where (s.size == 2 || s.size == 3) && !(s ~ 'x') select s;
    ;
    const neither =
        \\from string s in 'abc'
        \\where (s.size == 2 || s.size == 4) || !(s ~ '[a-c]') select s;
    ;

    // Act
    const got_both = try runQuery(both);
    const got_neither = try runQuery(neither);

    // Assert
    try std.testing.expectEqualStrings("", got_both.err);
    try std.testing.expectEqualStrings("abc\n", got_both.out);
    try std.testing.expectEqualStrings("", got_neither.err);
    try std.testing.expectEqualStrings("", got_neither.out);
}

test "compile+run dir from file orderby skips symlink" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "bb.txt", .data = "bb" });
    // Creating symlinks on Windows needs Developer Mode or SeCreateSymbolicLinkPrivilege.
    tmp.dir.symLink(state.io, "a.txt", "link.txt", .{}) catch |err| switch (err) {
        error.PermissionDenied => return error.SkipZigTest,
        else => return err,
    };

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d orderby f.size select f.size;",
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("1\n2\n", got.out);
}

test "compile+run orderby descending by file size" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "bb.txt", .data = "bb" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d orderby f.size descending select f.size;",
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("2\n1\n", got.out);
}

test "compile+run orderby ascending over string sequence" {
    // Arrange — multi-string seq via nested `from dir`…`select f.path`.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "bb", .data = "bb" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const a_path = try std.fs.path.join(std.testing.allocator, &.{ path, "a" });
    defer std.testing.allocator.free(a_path);
    const bb_path = try std.fs.path.join(std.testing.allocator, &.{ path, "bb" });
    defer std.testing.allocator.free(bb_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from string s in from dir d in '{s}' from file f in d select f.path
        \\orderby s.size
        \\select s;
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    const expect = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}\n{s}\n",
        .{ a_path, bb_path },
    );
    defer std.testing.allocator.free(expect);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings(expect, got.out);
}

test "compile+run orderby descending over string sequence" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "bb", .data = "bb" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const a_path = try std.fs.path.join(std.testing.allocator, &.{ path, "a" });
    defer std.testing.allocator.free(a_path);
    const bb_path = try std.fs.path.join(std.testing.allocator, &.{ path, "bb" });
    defer std.testing.allocator.free(bb_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from string s in from dir d in '{s}' from file f in d select f.path
        \\orderby s.size descending
        \\select s;
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    const expect = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}\n{s}\n",
        .{ bb_path, a_path },
    );
    defer std.testing.allocator.free(expect);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings(expect, got.out);
}

test "compile+run group by over string sequence" {
    // Arrange — paths `a`/`b` share length; `cc` is longer (same grouping shape as size 1/1/2).
    // Terminal group Record has a Seq `items` field; sink does not expand it (§7), so use into + from.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b", .data = "b" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "cc", .data = "cc" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const a_path = try std.fs.path.join(std.testing.allocator, &.{ path, "a" });
    defer std.testing.allocator.free(a_path);
    const b_path = try std.fs.path.join(std.testing.allocator, &.{ path, "b" });
    defer std.testing.allocator.free(b_path);
    const cc_path = try std.fs.path.join(std.testing.allocator, &.{ path, "cc" });
    defer std.testing.allocator.free(cc_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from string s in from dir d in '{s}' from file f in d orderby f.path select f.path
        \\group s by s.size into g
        \\from string x in g.items
        \\select {{ key = g.key, item = x }};
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    const key1 = a_path.len;
    const key2 = cc_path.len;
    const expect = try std.fmt.allocPrint(
        std.testing.allocator,
        "{d}\n{s}\n{d}\n{s}\n{d}\n{s}\n",
        .{ key1, a_path, key1, b_path, key2, cc_path },
    );
    defer std.testing.allocator.free(expect);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings(expect, got.out);
}

test "compile+run group by into over string sequence" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "bb", .data = "bb" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const a_path = try std.fs.path.join(std.testing.allocator, &.{ path, "a" });
    defer std.testing.allocator.free(a_path);
    const bb_path = try std.fs.path.join(std.testing.allocator, &.{ path, "bb" });
    defer std.testing.allocator.free(bb_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from string s in from dir d in '{s}' from file f in d orderby f.path select f.path
        \\group s by s.size into g
        \\select g.key;
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    const expect = try std.fmt.allocPrint(
        std.testing.allocator,
        "{d}\n{d}\n",
        .{ a_path.len, bb_path.len },
    );
    defer std.testing.allocator.free(expect);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings(expect, got.out);
}

test "compile+run group by into over directory" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b.txt", .data = "b" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "cc.txt", .data = "cc" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from dir d in '{s}'
        \\from file f in d
        \\orderby f.path
        \\group f by f.size into g
        \\select g.key;
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("1\n2\n", got.out);
}

test "compile+run terminal group by over directory" {
    // Arrange — sink of group Record must not expand `items` (§7); flatten via into + from.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b.txt", .data = "b" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "cc.txt", .data = "cc" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from dir d in '{s}' from file f in d orderby f.path
        \\group f by f.size into g
        \\from file x in g.items
        \\select {{ key = g.key, path = x.path }};
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    const a_txt = try std.fs.path.join(std.testing.allocator, &.{ path, "a.txt" });
    defer std.testing.allocator.free(a_txt);
    const b_txt = try std.fs.path.join(std.testing.allocator, &.{ path, "b.txt" });
    defer std.testing.allocator.free(b_txt);
    const cc_txt = try std.fs.path.join(std.testing.allocator, &.{ path, "cc.txt" });
    defer std.testing.allocator.free(cc_txt);

    const expect = try std.fmt.allocPrint(
        std.testing.allocator,
        "1\n{s}\n1\n{s}\n2\n{s}\n",
        .{ a_txt, b_txt, cc_txt },
    );
    defer std.testing.allocator.free(expect);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings(expect, got.out);
}

test "compile+run join into over file sources" {
    // Arrange
    var outer = std.testing.tmpDir(.{});
    defer outer.cleanup();
    var inner = std.testing.tmpDir(.{});
    defer inner.cleanup();

    try outer.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });
    try outer.dir.writeFile(state.io, .{ .sub_path = "bb.txt", .data = "bb" });
    try inner.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });
    try inner.dir.writeFile(state.io, .{ .sub_path = "yy.txt", .data = "yy" });
    try inner.dir.writeFile(state.io, .{ .sub_path = "z.txt", .data = "z" });

    const outer_path = try tmpQueryPath(std.testing.allocator, outer);
    defer std.testing.allocator.free(outer_path);
    const inner_path = try tmpQueryPath(std.testing.allocator, inner);
    defer std.testing.allocator.free(inner_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from dir od in '{s}'
        \\from file of in od
        \\orderby of.path
        \\from dir id in '{s}'
        \\join file jf in id on of.size equals jf.size into g
        \\from file mf in g
        \\select mf.size;
    ,
        .{ outer_path, inner_path },
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("1\n1\n2\n", got.out);
}

test "compile+run invalid property reports runtime error" {
    // Arrange
    const query = "from string s in 'abc' select s.nope;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run non-UTF-8 string payload hash reports payload error" {
    // Arrange — NTLM widens the payload to UTF-16LE; a byte literal can carry
    // non-UTF-8 bytes, which must surface as a payload error, not an I/O one.
    const query = "from string s in b'\\xDE\\xAD\\xBE\\xEF' select s.ntlm;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("string payload is not valid UTF-8 for this algorithm", got.err);
}

test "runQuery results stay valid across subsequent runs" {
    // Arrange — the first output is not a prefix of the second and the second
    // run reports an error, so both fields of the first result would go stale
    // if they aliased the shared scratch buffers.
    const first = try runQuery("from string s in 'ab' select s.size;");

    // Act
    const second = try runQuery("from string s in 'x' select s.nope;");

    // Assert
    try std.testing.expectEqualStrings("2\n", first.out);
    try std.testing.expectEqualStrings("", first.err);
    try std.testing.expectEqualStrings("", second.out);
    try std.testing.expectEqualStrings("invalid property for this value type", second.err);
}

test "compile+run integer literal overflow reports range error" {
    // Arrange — literals beyond i64 must be a compile error, not a silent 0
    // that would quietly rewrite predicates like `f.size > <literal>`.
    const cases = [_][]const u8{
        "from string s in 'a' select 99999999999999999999999;",
        "from string s in 'a' select -99999999999999999999999;",
        "from file f in 'x' where f.size > 18446744073709551616 select f.path;",
    };

    for (cases) |q| {
        // Act
        const got = try runQuery(q);

        // Assert
        try std.testing.expectEqualStrings("integer literal out of range", got.err);
    }
}

test "compile+run integer literal i64 boundaries parse exactly" {
    // Arrange — both i64 extremes are representable and must not trip the
    // overflow check (minInt arrives via the `-{DIGIT}+` lexer rule).
    const cases = [_]struct { q: []const u8, want: []const u8 }{
        .{ .q = "from string s in 'a' select 9223372036854775807;", .want = "9223372036854775807\n" },
        .{ .q = "from string s in 'a' select -9223372036854775808;", .want = "-9223372036854775808\n" },
    };

    for (cases) |tc| {
        // Act
        const got = try runQuery(tc.q);

        // Assert
        try std.testing.expectEqualStrings("", got.err);
        try std.testing.expectEqualStrings(tc.want, got.out);
    }
}

test "compile+run hyphenated hash property names" {
    // Arrange — algorithm ids with '-' must be one IDENTIFIER (issue #333),
    // not split into IDENT + negative INTEGER / INVALID.
    const cases = [_]struct { q: []const u8, want: []const u8 }{
        .{
            .q = "from string s in 'abc' select s.sha-3-224;",
            .want = "e642824c3f8cf24ad09234ee7d3c766fc9a3a5168d0c94ad73b46fdf\n",
        },
        .{
            .q = "from string s in 'abc' select s.crc64-xz;",
            .want = "2cd8094a1a277627\n",
        },
        .{
            .q = "from string s in 'abc' select s.sha-3k-256;",
            .want = "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45\n",
        },
    };

    for (cases) |tc| {
        // Act
        const got = try runQuery(tc.q);

        // Assert
        try std.testing.expectEqualStrings("", got.err);
        try std.testing.expectEqualStrings(tc.want, got.out);
    }
}

test "compile+run hyphenated hash-check method" {
    // Arrange
    const query =
        "from string s in 'abc' select s.sha-3-224('e642824c3f8cf24ad09234ee7d3c766fc9a3a5168d0c94ad73b46fdf');";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run unknown hyphenated property is not a syntax error" {
    // Arrange — kebab name parses; catalog still rejects unknowns.
    const query = "from string s in 'abc' select s.foo-bar;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run negative literal beside hyphenated property" {
    // Arrange — signed INTEGER must stay separate from kebab IDENTIFIERs.
    const query = "from string s in 'abc' where s.size > -1 select s.sha-3-224;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        "e642824c3f8cf24ad09234ee7d3c766fc9a3a5168d0c94ad73b46fdf\n",
        got.out,
    );
}

test "compile+run non-UTF-8 payload hash-check reports payload error" {
    // Arrange — same constraint via the hash-check method form (§4.8).
    const query = "from string s in b'\\xFF' where s.ntlm('00000000000000000000000000000000') select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("string payload is not valid UTF-8 for this algorithm", got.err);
}

test "compile+run undefined select name reports undefined name" {
    // Arrange
    const query = "from string s in 'abc' select missing;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("undefined name", got.err);
}

test "compile+run nested query undefined name stays UndefinedName" {
    // Arrange — Nested query plans are compiled (and typechecked) before eval, so the
    // failure surfaces at compilation.
    const query = "from string s in 'abc' where from string t in missing select t select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("undefined name", got.err);
}

test "plain hex-looking strings compare case-sensitively" {
    // Arrange
    const query = "from string s in 'ab' where s == 'AB' select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run hex escapes hash as binary payload" {
    // Arrange — Zig source doubles backslashes so the query still contains `\xNN` text.
    const query = "from string s in b\"\\x00\\x01\\x02\" select s.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("b95f67f61ebb03619622d798f45fc2d3\n", got.out);
}

test "compile+run hex escapes size is byte count" {
    // Arrange
    const query = "from string s in b'\\xDE\\xAD\\xBE\\xEF' select s.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("4\n", got.out);
}

test "compile+run invalid byte-string escape reports error" {
    // Arrange
    const query = "from string s in b\"\\xZZ\" select s.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid string escape sequence", got.err);
}

test "compile+run plain string keeps backslash path text" {
    // Arrange
    const query = "from string s in 'c:\\Windows' select s.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("10\n", got.out); // c:\Windows
}

test "hash property equals uppercase digest literal case-insensitively" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\where s.md5 == '900150983CD24FB0D6963F7D28E17F72'
        \\select s;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "invalid property span points at property expression" {
    // Arrange
    const query = "from string s in 'abc' select s.nope;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
    // `s.nope` starts after "from string s in 'abc' select " (cols 31–36 inclusive)
    try std.testing.expectEqual(@as(c_int, 1), run_span.first_line);
    try std.testing.expectEqual(@as(c_int, 31), run_span.first_column);
    try std.testing.expectEqual(@as(c_int, 36), run_span.last_column);
}

test "undefined name span is the identifier only" {
    // Arrange — single-char name must not include the trailing `;`
    const query = "from dir a in '/home/egr' from file f in a select d;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("undefined name", got.err);
    try std.testing.expectEqual(@as(c_int, 1), run_span.first_line);
    try std.testing.expectEqual(@as(c_int, 51), run_span.first_column);
    try std.testing.expectEqual(@as(c_int, 51), run_span.last_column);
}

test "undefined name span on continuation line does not include leading space" {
    // Arrange — newline column tracking used to leave last_column=0, shifting
    // every column on the next line by -1 (underline started on the space).
    const query =
        \\from string s in 'abc'
        \\select missing;
    ;

    // Act
    const got = try runQuery(query);

    // Assert — `missing` is columns 8–14 on line 2
    try std.testing.expectEqualStrings("undefined name", got.err);
    try std.testing.expectEqual(@as(c_int, 2), run_span.first_line);
    try std.testing.expectEqual(@as(c_int, 8), run_span.first_column);
    try std.testing.expectEqual(@as(c_int, 14), run_span.last_column);
}

test "compile+run from file in string variable opens as path" {
    // Arrange — String source is a path payload (§3.3), not a Dir listing.
    const query = "from string d in 'abc' from file f in d select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert — 'abc' is not an existing regular file
    try std.testing.expectEqualStrings(
        "I/O failure (missing path or unreadable file/directory): abc",
        got.err,
    );
}

test "compile+run select into continuation rejects outer name" {
    // Arrange — continuation Env has only the into-bound name (§6.8)
    const query = "from string s in 'abc' select s.md5 into h select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("undefined name", got.err);
}

test "compile+run missing file reports io failure" {
    // Arrange
    const query = "from file f in '/definitely-missing-l2h-test-path' select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings(
        "I/O failure (missing path or unreadable file/directory): /definitely-missing-l2h-test-path",
        got.err,
    );
    // Path literal in `from file f in '…'`
    try std.testing.expectEqual(@as(c_int, 1), run_span.first_line);
    try std.testing.expectEqual(@as(c_int, 16), run_span.first_column);
}

test "compile+run file.path projects bound path" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "x.txt");
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select f.path;",
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    const expect = try std.fmt.allocPrint(std.testing.allocator, "{s}\n", .{file_path});
    defer std.testing.allocator.free(expect);
    try std.testing.expectEqualStrings(expect, got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run file.name projects basename only" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "x.txt");
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select f.name;",
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("x.txt\n", got.out);
}

test "compile+run file sfv and checksum ignore declaration order" {
    // Arrange — Digest field first in the object — output order is still fixed by method.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "x.txt");
    defer std.testing.allocator.free(file_path);

    // md5("x") = 9dd4e461268c8034f5c8564e155c67a6
    const sfv_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' let o = {{ f.md5, f.name }} select o.sfv();",
        .{file_path},
    );
    defer std.testing.allocator.free(sfv_q);

    // Act
    const sfv = try runQuery(sfv_q);

    // Assert
    try std.testing.expectEqualStrings("", sfv.err);
    try std.testing.expectEqualStrings("x.txt    9dd4e461268c8034f5c8564e155c67a6\n", sfv.out);

    const sum_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' let o = {{ f.path, f.md5 }} select o.checksum();",
        .{file_path},
    );
    defer std.testing.allocator.free(sum_q);
    const sum = try runQuery(sum_q);
    try std.testing.expectEqualStrings("", sum.err);
    const expect_sum = try std.fmt.allocPrint(
        std.testing.allocator,
        "9dd4e461268c8034f5c8564e155c67a6 {s}\n",
        .{file_path},
    );
    defer std.testing.allocator.free(expect_sum);
    try std.testing.expectEqualStrings(expect_sum, sum.out);
}

test "compile+run record literal method call without let" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "x.txt");
    defer std.testing.allocator.free(file_path);

    const sfv_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select {{ f.md5, f.name }}.sfv();",
        .{file_path},
    );
    defer std.testing.allocator.free(sfv_q);

    // Act
    const sfv = try runQuery(sfv_q);

    // Assert
    try std.testing.expectEqualStrings("", sfv.err);
    try std.testing.expectEqualStrings("x.txt    9dd4e461268c8034f5c8564e155c67a6\n", sfv.out);

    const sum_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select {{ f.path, f.md5 }}.checksum();",
        .{file_path},
    );
    defer std.testing.allocator.free(sum_q);
    const sum = try runQuery(sum_q);
    try std.testing.expectEqualStrings("", sum.err);
    const expect_sum = try std.fmt.allocPrint(
        std.testing.allocator,
        "9dd4e461268c8034f5c8564e155c67a6 {s}\n",
        .{file_path},
    );
    defer std.testing.allocator.free(expect_sum);
    try std.testing.expectEqualStrings(expect_sum, sum.out);
}

test "compile+run file limit and offset window hashes like hc" {
    // Arrange — "0123456789" with offset=2,limit=4 → hash of "2345"
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "part.txt", .data = "0123456789" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "part.txt");
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select f.offset(2).limit(4).md5;",
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("81b073de9370ea873f548e31b8adc081\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run file window via let does not mutate original" {
    // Arrange — windowed hash on w; bare f stays full-file
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "part.txt", .data = "0123456789" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "part.txt");
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from file f in '{s}'
        \\let w = f.offset(2).limit(4)
        \\where w.md5 == '81b073de9370ea873f548e31b8adc081'
        \\select {{ wm = w.md5, fs = f.size, fm = f.md5 }}.json();
    ,
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert — window only on w; f size+md5 are full-file
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        "{\"wm\":\"81b073de9370ea873f548e31b8adc081\",\"fs\":10,\"fm\":\"781e5e245d69b566979b86e28d23f2c7\"}\n",
        got.out,
    );
}

test "compile+run string.limit is invalid property" {
    // Arrange
    const query = "from string s in 'abc' where s.limit == 1 select s.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run string.offset method is invalid receiver" {
    // Arrange

    // Act
    const got = try runQuery("from string s in 'abc' select s.offset(1).md5;");

    // Assert
    try std.testing.expectEqualStrings("invalid method receiver", got.err);
}

test "compile+run file.offset arity errors" {
    // Arrange

    // Act
    const got0 = try runQuery("from file f in 'x' select f.offset().md5;");

    // Assert
    try std.testing.expectEqualStrings("wrong number of method arguments", got0.err);
    const got2 = try runQuery("from file f in 'x' select f.offset(1, 2).md5;");
    try std.testing.expectEqualStrings("wrong number of method arguments", got2.err);
}

test "compile+run file.offset(true) is type mismatch" {
    // Arrange

    // Act
    const got = try runQuery("from file f in 'x' select f.offset(true).md5;");

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run file window property reads after method" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "part.txt", .data = "0123456789" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "part.txt");
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from file f in '{s}'
        \\let w = f.limit(4)
        \\select {{ fo = f.offset, fl = f.limit, wo = w.offset, wl = w.limit }}.json();
    ,
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        "{\"fo\":0,\"fl\":9223372036854775807,\"wo\":0,\"wl\":4}\n",
        got.out,
    );
}

test "compile+run dir.path projects bound path" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' select d.path;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    const expect = try std.fmt.allocPrint(std.testing.allocator, "{s}\n", .{dir_path});
    defer std.testing.allocator.free(expect);
    try std.testing.expectEqualStrings(expect, got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run dir.tree() walks nested files" {
    // Arrange — top-level + one nested file; flat must miss nested
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "top.txt", .data = "t" });
    try tmp.dir.createDir(state.io, "sub", std.Io.Dir.Permissions.default_dir);
    try tmp.dir.writeFile(state.io, .{ .sub_path = "sub/nested.txt", .data = "nn" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const flat_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d orderby f.size select f.size;",
        .{dir_path},
    );
    defer std.testing.allocator.free(flat_q);

    const deep_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree() orderby f.size select f.size;",
        .{dir_path},
    );
    defer std.testing.allocator.free(deep_q);

    // Act
    const flat = try runQuery(flat_q);
    const deep = try runQuery(deep_q);

    // Assert
    try std.testing.expectEqualStrings("1\n", flat.out);
    try std.testing.expectEqualStrings("", flat.err);
    try std.testing.expectEqualStrings("1\n2\n", deep.out);
    try std.testing.expectEqualStrings("", deep.err);
}

test "compile+run dir.tree() does not mutate original dir" {
    // Arrange — bare `d` stays flat after using `d.tree()` elsewhere
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "top.txt", .data = "t" });
    try tmp.dir.createDir(state.io, "sub", std.Io.Dir.Permissions.default_dir);
    try tmp.dir.writeFile(state.io, .{ .sub_path = "sub/nested.txt", .data = "nn" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from dir d in '{s}'
        \\from file f in d.tree()
        \\from file g in d
        \\select g.size;
    ,
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert — only top-level file via `g in d` (flat), once per recursive outer row
    try std.testing.expectEqualStrings("1\n1\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run dir.tree property access is invalid" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' where d.tree == true select d.path;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run dir.tree(true) is type mismatch" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree(true) select f.path;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run dir.tree(0) matches flat listing" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "top.txt", .data = "t" });
    try tmp.dir.createDir(state.io, "sub", std.Io.Dir.Permissions.default_dir);
    try tmp.dir.writeFile(state.io, .{ .sub_path = "sub/nested.txt", .data = "nn" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree(0) orderby f.size select f.size;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert — nested file excluded
    try std.testing.expectEqualStrings("1\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run multi-key orderby sorts by the secondary key on ties" {
    // Arrange — §6.5: same-size files differ only by name; the pair of
    // opposite-direction queries is deterministic iff the secondary key
    // is actually consulted (readdir order cannot pass both).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b.txt", .data = "x" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const asc_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d orderby f.size, f.name select f.name;",
        .{dir_path},
    );
    defer std.testing.allocator.free(asc_q);
    const desc_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d orderby f.size, f.name descending select f.name;",
        .{dir_path},
    );
    defer std.testing.allocator.free(desc_q);

    // Act
    const got_asc = try runQuery(asc_q);
    const got_desc = try runQuery(desc_q);

    // Assert
    try std.testing.expectEqualStrings("", got_asc.err);
    try std.testing.expectEqualStrings("a.txt\nb.txt\n", got_asc.out);
    try std.testing.expectEqualStrings("", got_desc.err);
    try std.testing.expectEqualStrings("b.txt\na.txt\n", got_desc.out);
}

test "compile+run dir.tree(1) stops after one subdirectory level" {
    // Arrange — root / one / two levels deep
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "top.txt", .data = "t" });
    try tmp.dir.createDir(state.io, "sub", std.Io.Dir.Permissions.default_dir);
    try tmp.dir.writeFile(state.io, .{ .sub_path = "sub/mid.txt", .data = "mm" });
    try tmp.dir.createDir(state.io, "sub/deep", std.Io.Dir.Permissions.default_dir);
    try tmp.dir.writeFile(state.io, .{ .sub_path = "sub/deep/bot.txt", .data = "bbb" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const q1 = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree(1) orderby f.size select f.size;",
        .{dir_path},
    );
    defer std.testing.allocator.free(q1);

    const q2 = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree(2) orderby f.size select f.size;",
        .{dir_path},
    );
    defer std.testing.allocator.free(q2);

    // Act
    const got1 = try runQuery(q1);
    const got2 = try runQuery(q2);

    // Assert — tree(1): top(1) + mid(2); tree(2): also bot(3)
    try std.testing.expectEqualStrings("1\n2\n", got1.out);
    try std.testing.expectEqualStrings("", got1.err);
    try std.testing.expectEqualStrings("1\n2\n3\n", got2.out);
    try std.testing.expectEqualStrings("", got2.err);
}

test "compile+run dir.tree(-1) is invalid tree depth" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree(-1) select f.path;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("tree depth must be non-negative", got.err);
}

test "compile+run dir.tree two args is arity error" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree(1, 2) select f.path;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("wrong number of method arguments", got.err);
}

test "compile+run dir.tree() without skipErrors fails on unreadable subdir" {
    if (comptime @import("builtin").os.tag == .windows) return;

    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer {
        tmp.dir.setFilePermissions(state.io, "denied", .fromMode(0o700), .{}) catch {};
        tmp.cleanup();
    }

    try tmp.dir.writeFile(state.io, .{ .sub_path = "ok.txt", .data = "ok" });
    try tmp.dir.createDir(state.io, "denied", .fromMode(0));

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree() select f.path;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expect(std.mem.startsWith(u8, got.err, "I/O failure (missing path or unreadable file/directory):"));
    try std.testing.expect(std.mem.indexOf(u8, got.err, "denied") != null);
}

test "compile+run dir.tree().skipErrors() skips unreadable subdir" {
    if (comptime @import("builtin").os.tag == .windows) return;

    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer {
        tmp.dir.setFilePermissions(state.io, "denied", .fromMode(0o700), .{}) catch {};
        tmp.cleanup();
    }

    try tmp.dir.writeFile(state.io, .{ .sub_path = "ok.txt", .data = "ok" });
    try tmp.dir.createDir(state.io, "denied", .fromMode(0));

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree().skipErrors() select f.name;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("ok.txt\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run skipErrors().tree() composes flags" {
    if (comptime @import("builtin").os.tag == .windows) return;

    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer {
        tmp.dir.setFilePermissions(state.io, "denied", .fromMode(0o700), .{}) catch {};
        tmp.cleanup();
    }

    try tmp.dir.writeFile(state.io, .{ .sub_path = "ok.txt", .data = "ok" });
    try tmp.dir.createDir(state.io, "denied", .fromMode(0));

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.skipErrors().tree() select f.name;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("ok.txt\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run tree stream filters many files without orderby" {
    // Arrange — many siblings; only one name matches. Exercises streaming Dir walk
    // (no full path materialization) plus where-before-select.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var i: usize = 0;
    while (i < 64) : (i += 1) {
        var name_buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "f{d:0>3}.txt", .{i});
        try tmp.dir.writeFile(state.io, .{ .sub_path = name, .data = "x" });
    }
    try tmp.dir.writeFile(state.io, .{ .sub_path = "keep.txt", .data = "keep" });
    try tmp.dir.createDir(state.io, "sub", std.Io.Dir.Permissions.default_dir);
    try tmp.dir.writeFile(state.io, .{ .sub_path = "sub/nested.txt", .data = "n" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree() where f.name == 'keep.txt' select f.name;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("keep.txt\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run from file in Dir via Seq survives row arena reset" {
    // Arrange — Dir from nested query (Seq), then walk files. Regression for
    // use-after-reset of DirVal.path / seq items when FromOp resets row_arena.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b.txt", .data = "bb" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "c.txt", .data = "ccc" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    // Native separators: runtime path.join on Windows emits `\`.
    const a_txt = try std.fs.path.join(std.testing.allocator, &.{ path, "a.txt" });
    defer std.testing.allocator.free(a_txt);
    const b_txt = try std.fs.path.join(std.testing.allocator, &.{ path, "b.txt" });
    defer std.testing.allocator.free(b_txt);
    const c_txt = try std.fs.path.join(std.testing.allocator, &.{ path, "c.txt" });
    defer std.testing.allocator.free(c_txt);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from string s in 'x'
        \\from dir dd in from dir t in '{s}' select t
        \\from file f in dd
        \\orderby f.path
        \\select f.path;
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    const expect = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}\n{s}\n{s}\n",
        .{ a_txt, b_txt, c_txt },
    );
    defer std.testing.allocator.free(expect);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings(expect, got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run orderby f.path restores lex order over tree" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "b.txt", .data = "b" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });
    try tmp.dir.createDir(state.io, "m", std.Io.Dir.Permissions.default_dir);
    try tmp.dir.writeFile(state.io, .{ .sub_path = "m/c.txt", .data = "c" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d.tree() orderby f.path select f.name;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert — lex by full path: a.txt, b.txt, m/c.txt
    try std.testing.expectEqualStrings("a.txt\nb.txt\nc.txt\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run where f.readable filters unreadable files" {
    if (comptime @import("builtin").os.tag == .windows) return;

    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer {
        tmp.dir.setFilePermissions(state.io, "locked.txt", .fromMode(0o644), .{}) catch {};
        tmp.cleanup();
    }

    try tmp.dir.writeFile(state.io, .{ .sub_path = "ok.txt", .data = "ok" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "locked.txt", .data = "secret" });
    try tmp.dir.setFilePermissions(state.io, "locked.txt", .fromMode(0), .{});

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d where f.readable select f.name;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("ok.txt\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run file.readable is a valid bare where predicate" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' from file f in d where f.readable select f.size;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("1\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run file.tree() is invalid method receiver" {
    // Arrange
    const query = "from file f in 'x' select f.tree();";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid method receiver", got.err);
}

test "compile+run boolean literals as values and predicates" {
    // Arrange / Act
    const select_true = try runQuery("from string s in 'a' select true;");
    const select_false = try runQuery("from string s in 'a' select false;");
    const where_true = try runQuery("from string s in 'a' where true select s.size;");
    const where_false = try runQuery("from string s in 'a' where false select s.size;");

    // Assert
    try std.testing.expectEqualStrings("true\n", select_true.out);
    try std.testing.expectEqualStrings("", select_true.err);
    try std.testing.expectEqualStrings("false\n", select_false.out);
    try std.testing.expectEqualStrings("", select_false.err);
    try std.testing.expectEqualStrings("1\n", where_true.out);
    try std.testing.expectEqualStrings("", where_true.err);
    try std.testing.expectEqualStrings("", where_false.out);
    try std.testing.expectEqualStrings("", where_false.err);
}

test "compile+run file.tree is invalid property" {
    // Arrange
    const query = "from file f in 'x' where f.tree == 1 select f.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run dir.size is invalid property" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' select d.size;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run hash digest wrong length for algorithm" {
    // Arrange: MD5 digest (32 hex) cannot be restored as SHA1 (40 hex).
    const query = "from hash h in '202CB962AC59075B964B07152D234B70' select h.sha1;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid hash digest for the selected algorithm", got.err);
    try std.testing.expectEqual(@as(c_int, 1), run_span.first_line);
    try std.testing.expectEqual(@as(c_int, 58), run_span.first_column);
}

test "compile+run into md5 then restore as sha1 reports invalid digest" {
    // Arrange
    const query =
        "from string s in '123' select s.md5 into h123 from hash h in h123 select h.sha1;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid hash digest for the selected algorithm", got.err);
}

test "compile+run hash restore success prints runner output without duplicate digest" {
    // Arrange — §4.4 stdout contract: md5("") restores instantly via the
    // empty-string path; the terminal bare-select sink must not re-print the
    // returned digest (either casing) after the runner output.
    const query = "from hash x in 'D41D8CD98F00B204E9800998ECF8427E' select x.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expect(std.mem.indexOf(u8, got.out, "Initial string is: Empty string") != null);
    try std.testing.expect(std.mem.indexOf(u8, got.out, "D41D8CD98F00B204E9800998ECF8427E") == null);
    try std.testing.expect(std.mem.indexOf(u8, got.out, "d41d8cd98f00b204e9800998ecf8427e") == null);
}

test "compile+run hash restore value preserves input casing off the bare select" {
    // Arrange — the Hash property returns the bound digest as stored; a
    // record projection is not the bare-select form, so the sink prints it.
    // Mixed case proves no re-casing (computed digests are always lowercase).
    const query = "from hash x in 'D41d8cd98F00B204E9800998ECF8427E' select { x.md5 };";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expect(std.mem.indexOf(u8, got.out, "Initial string is: Empty string") != null);
    try std.testing.expect(std.mem.indexOf(u8, got.out, "D41d8cd98F00B204E9800998ECF8427E\n") != null);
}

test "compile+run hash restore knobs via let does not mutate original" {
    // Arrange
    const query =
        \\from hash x in '202CB962AC59075B964B07152D234B70'
        \\let h = x.min(3).max(7).noProbe()
        \\select { xn = x.min, xm = x.max, xnp = x.noProbe, hn = h.min, hm = h.max, hnp = h.noProbe }.json();
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        "{\"xn\":1,\"xm\":10,\"xnp\":false,\"hn\":3,\"hm\":7,\"hnp\":true}\n",
        got.out,
    );
}

test "compile+run hash restore noProbe skips timing probe line" {
    // Arrange — md5("123"); default restore probes unless disabled
    const with_probe =
        "from hash x in '202CB962AC59075B964B07152D234B70' select x.md5;";
    const without_probe =
        "from hash x in '202CB962AC59075B964B07152D234B70' select x.noProbe().md5;";

    // Act
    const probed = try runQuery(with_probe);
    const quiet = try runQuery(without_probe);

    // Assert
    try std.testing.expectEqualStrings("", probed.err);
    try std.testing.expectEqualStrings("", quiet.err);
    try std.testing.expect(std.mem.indexOf(u8, probed.out, "May take approximatelly") != null);
    try std.testing.expect(std.mem.indexOf(u8, quiet.out, "May take approximatelly") == null);
    try std.testing.expect(std.mem.indexOf(u8, quiet.out, "Initial string is: 123") != null);
}

test "compile+run hash.dict is invalid on string" {
    // Arrange
    const query = "from string s in 'abc' select s.dict('x').md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid method receiver", got.err);
}

test "compile+run hash.min is invalid property on string" {
    // Arrange
    const query = "from string s in 'abc' where s.min == 1 select s.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run hash restore honors custom dict and bounds" {
    // Arrange — md5("123") with alphabet restricted to digits and max length 3
    const query =
        "from hash x in '202CB962AC59075B964B07152D234B70' select x.dict('0123456789').max(3).noProbe().md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expect(std.mem.indexOf(u8, got.out, "Initial string is: 123") != null);
}

test "compile+run hash.min(0) is InvalidRestoreBound" {
    // Arrange
    const query = "from hash x in '202CB962AC59075B964B07152D234B70' select x.min(0);";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("restore min/max must be at least 1", got.err);
}

test "compile+run hash.min(-1) is not file-window InvalidWindow" {
    // Arrange
    const query = "from hash x in '202CB962AC59075B964B07152D234B70' select x.min(-1);";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("restore min/max must be at least 1", got.err);
}

test "compile+run hash.min overflow is value out of integer range" {
    // Arrange — i32 max is 2147483647; same ceiling as `hc hash -n`.
    const query = "from hash x in '202CB962AC59075B964B07152D234B70' select x.min(2147483648);";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("value out of integer range", got.err);
}

test "compile+run hash min greater than max is InvalidRestoreRange" {
    // Arrange
    const query = "from hash x in '202CB962AC59075B964B07152D234B70' select x.min(5).max(2);";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("restore min length is greater than max", got.err);
    try std.testing.expect(std.mem.indexOf(u8, got.out, "Minimum password length") == null);
}

test "compile+run invalid group property fails during compilation" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\group s by s.size into g
        \\select g.nope;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run typed record field access works" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let r = { s, s.size }
        \\orderby r.size descending
        \\select r.s;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run explicit record alias and auto-name mix works" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let r = { digest = s.md5, s.size }
        \\select r.digest;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run missing typed record field fails during compilation" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let r = { s }
        \\select r.nope;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run duplicate record field fails during compilation" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let r = { digest = s.md5, digest = s.sha1 }
        \\select r.digest;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("duplicate record field name", got.err);
}

test "compile+run nested query in let produces sequence value" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let items = from string t in s select t.md5
        \\select items;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run nested query in select produces sequence value" {
    // Arrange
    const query =
        "from string s in 'abc' select from string t in s select t;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run nested query in record field works" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let r = { items = from string t in s select t.md5 }
        \\select r.items;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run match operand mismatch fails during compilation" {
    // Arrange — §5.2: both sides of `~` must be String (no stringify)
    const query = "from string s in 'abc' where s.size ~ 'x' select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run match pattern must be string" {
    // Arrange
    const query = "from string s in 'abc' where s ~ 3 select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run from file rejects directory path" {
    // Arrange — §3.3: from file requires a regular file
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select f.path;",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    const expect_err = try std.fmt.allocPrint(
        std.testing.allocator,
        "I/O failure (missing path or unreadable file/directory): {s}",
        .{dir_path},
    );
    defer std.testing.allocator.free(expect_err);
    try std.testing.expectEqualStrings(expect_err, got.err);
}

test "compile+run from file over int source fails during compilation" {
    // Arrange
    const query = "from file f in 1 select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "compile+run equality operand mismatch fails during compilation" {
    // Arrange
    const query = "from string s in 'abc' where s == 1 select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run join key mismatch fails during compilation" {
    // Arrange
    const query =
        \\from string a in 'abc'
        \\join string b in 'x' on a equals b.size
        \\select a;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run orderby key must be comparable" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\group s by s.size into g
        \\orderby g.items
        \\select g.key;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run group items property access stays typed" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\group s by s.size into g
        \\from string item in g.items
        \\select item;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run group by record key fails during compilation" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\group s by { s } into g
        \\select g.key;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run from file in string sequence fails during compilation" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let xs = from string t in s select t
        \\from file f in xs
        \\select f.size;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "compile+run nested query as where exists predicate" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\where from string t in 'ab' where t.size == s.size select t
        \\select s;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.out);
}

test "compile+run nested query where exists keeps matching row" {
    // Arrange
    const query =
        \\from string s in 'ab'
        \\where from string t in 'xy' where t.size == s.size select t
        \\select s;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("ab\n", got.out);
}

test "compile+run nested query in orderby singleton unwrap" {
    // Arrange
    const query =
        \\from string s in 'bb'
        \\from string t in 'a'
        \\orderby from string x in t select x.size
        \\select t;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    // Cartesian product of singletons: one row with t='a', ordered by nested size.
    try std.testing.expectEqualStrings("a\n", got.out);
}

test "compile+run from in nested query sequence" {
    // Arrange
    const query =
        "from string x in from string t in 'abc' select t select x;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run join in nested query sequence" {
    // Arrange
    const query =
        \\from string a in 'ab'
        \\join string b in from string t in 'xy' select t
        \\on a.size equals b.size
        \\select a;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("ab\n", got.out);
}

test "compile+run join key nested query singleton unwrap" {
    // Arrange
    const query =
        \\from string a in 'abc'
        \\join string b in 'xyz'
        \\on a.size equals from string t in b select t.size
        \\select a;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run group by nested query key" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\group s by from string t in s select t.size into g
        \\select g.key;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("3\n", got.out);
}

test "compile+run from in nested query wrong item kind fails during compilation" {
    // Arrange
    const query =
        "from file f in from string t in 'abc' select t select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "compile+run nested query uses outer binding in inner source" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let xs = from string t in s select t
        \\from string x in xs
        \\select x;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run invalid property on nested sequence fails during compilation" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let items = from string t in s select t
        \\select items.size;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run Seq.count() returns cardinality" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b", .data = "b" });
    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);
    const many_q = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from string s in 'x'
        \\let items = from dir d in '{s}' from file f in d select f.path
        \\select items.count();
    ,
        .{path},
    );
    defer std.testing.allocator.free(many_q);

    // Act
    const one = try runQuery(
        \\from string s in 'abc'
        \\let items = from string t in s select t
        \\select items.count();
    );

    const empty = try runQuery(
        \\from string s in 'abc'
        \\let items = from string t in s where false select t
        \\select items.count();
    );

    const many = try runQuery(many_q);

    // Assert
    try std.testing.expectEqualStrings("1\n", one.out);
    try std.testing.expectEqualStrings("", one.err);
    try std.testing.expectEqualStrings("0\n", empty.out);
    try std.testing.expectEqualStrings("", empty.err);
    try std.testing.expectEqualStrings("", many.err);
    try std.testing.expectEqualStrings("2\n", many.out);
}

test "compile+run group items.count() returns group size" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\group s by s.size into g
        \\select g.items.count();
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("1\n", got.out);
}

test "compile+run count() on non-Seq is invalid method receiver" {
    // Arrange
    const query = "from string s in 'abc' select s.count();";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid method receiver", got.err);
}

test "compile+run count() with args is InvalidMethodArity" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let items = from string t in s select t
        \\select items.count(1);
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("wrong number of method arguments", got.err);
}

test "compile+run bare Seq.count is invalid property" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let items = from string t in s select t
        \\select items.count;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run Seq.count() after script into across statements" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b", .data = "b" });
    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);
    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from dir d in '{s}' from file f in d select f into files;
        \\from string _ in 'x' select files.count();
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("2\n", got.out);
}

test "compile+run shallow nested query succeeds within depth limit" {
    // Arrange — A handful of nesting levels is well within MAX_QUERY_DEPTH and must
    // behave as before the guard.
    var buf: [4096]u8 = undefined;
    var fbs = std.Io.Writer.fixed(&buf);
    try fbs.writeAll("from string s in 'abc' ");
    // 5 levels of `let xsN = from string t in xsN-1 select t`
    try fbs.writeAll("let x0 = from string t in s select t ");
    var i: u32 = 1;
    while (i <= 5) : (i += 1) {
        try fbs.print("let x{d} = from string t in x{d} select t ", .{ i, i - 1 });
    }
    try fbs.writeAll("select x5;");

    // Act
    const got = try runQuery(std.Io.Writer.buffered(&fbs));

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run deeply nested query reports QueryTooDeep" {
    // Arrange — Adversarial nesting (a select whose value is itself a query, repeated
    // beyond MAX_QUERY_DEPTH) must surface a clean error instead of crashing
    // the process with a stack overflow. Each `from string t in s select <…>`
    // adds a nesting level the analysis/eval passes descend into.
    var buf: [128 * 1024]u8 = undefined;
    var fbs = std.Io.Writer.fixed(&buf);
    try fbs.writeAll("from string s in 'abc' select ");
    const depth = compile.MAX_QUERY_DEPTH + 4;
    var i: u32 = 0;
    while (i < depth) : (i += 1) {
        try fbs.writeAll("from string t in s select ");
    }
    try fbs.writeAll("t;");

    // Act
    const got = try runQuery(std.Io.Writer.buffered(&fbs));

    // Assert
    try std.testing.expectEqualStrings("query nesting too deep", got.err);
}

test "compile+run record sfv via let" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let o = { name = 'x', digest = s.md5 }
        \\select o.sfv();
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("x    900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run record checksum via into" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\select { path = '/tmp/x', digest = s.md5 } into o
        \\select o.checksum();
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72 /tmp/x\n", got.out);
}

test "compile+run script record field names survive later statements" {
    // Arrange — `into o` stores a Record in script env; plan arena frees after
    // that statement. A churn statement reuses GPA pages; field-name lookups
    // must still work (Value.dupe must own record field names).
    const query =
        \\from string s in 'abc' select { path = '/tmp/x', digest = s.md5 } into o;
        \\from string t in 'churn-allocator-padding-xxxxxxxx' select t;
        \\from string u in 'q' select o.path;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("churn-allocator-padding-xxxxxxxx\n/tmp/x\n", got.out);
}

test "compile+run record json and jsonPretty" {
    // Arrange
    const compact_q =
        \\from string s in 'abc'
        \\let o = { a = 'x', n = s.size }
        \\select o.json();
    ;

    // Act
    const compact = try runQuery(compact_q);

    // Assert
    try std.testing.expectEqualStrings("", compact.err);
    try std.testing.expectEqualStrings("{\"a\":\"x\",\"n\":3}\n", compact.out);

    const pretty_q =
        \\from string s in 'abc'
        \\let o = { a = 'x', n = s.size }
        \\select o.jsonPretty();
    ;
    const pretty = try runQuery(pretty_q);
    try std.testing.expectEqualStrings("", pretty.err);
    try std.testing.expectEqualStrings("{\n  \"a\": \"x\",\n  \"n\": 3\n}\n", pretty.out);
}

test "compile+run jsonPretty allows nested record fields" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let hashes = { digest = s.md5, n = s.size }
        \\select { path = 'x', hashes } into o
        \\select o.json();
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        "{\"path\":\"x\",\"hashes\":{\"digest\":\"900150983cd24fb0d6963f7d28e17f72\",\"n\":3}}\n",
        got.out,
    );
}

test "compile+run record csv spaced tabbed" {
    // Arrange
    const csv_q =
        \\from string s in 'abc'
        \\let o = { a = 'one', b = 'two' }
        \\select o.csv();
    ;

    // Act
    const csv = try runQuery(csv_q);

    // Assert
    try std.testing.expectEqualStrings("", csv.err);
    try std.testing.expectEqualStrings("one,two\n", csv.out);

    const spaced_q =
        \\from string s in 'abc'
        \\let o = { a = 'one', b = 'two' }
        \\select o.spaced();
    ;
    const spaced = try runQuery(spaced_q);
    try std.testing.expectEqualStrings("", spaced.err);
    try std.testing.expectEqualStrings("one two\n", spaced.out);

    const tabbed_q =
        \\from string s in 'abc'
        \\let o = { a = 'one', b = 'two' }
        \\select o.tabbed();
    ;
    const tabbed = try runQuery(tabbed_q);
    try std.testing.expectEqualStrings("", tabbed.err);
    try std.testing.expectEqualStrings("one\ttwo\n", tabbed.out);
}

test "compile+run bare record still prints one line per field" {
    // Arrange
    const query =
        "from string s in 'abc' select { a = '1', b = '2' };";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("1\n2\n", got.out);
}

test "compile+run method on non-record reports InvalidMethodReceiver" {
    // Arrange

    // Act
    const got = try runQuery("from string s in 'abc' select s.sfv();");

    // Assert
    try std.testing.expectEqualStrings("invalid method receiver", got.err);
}

test "compile+run hash-check method on string match and mismatch" {
    // Arrange
    const match_q = "from string s in 'abc' select s.md5('900150983cd24fb0d6963f7d28e17f72');";

    // Act
    const match_got = try runQuery(match_q);

    // Assert
    try std.testing.expectEqualStrings("", match_got.err);
    try std.testing.expectEqualStrings("true\n", match_got.out);

    const mismatch_q = "from string s in 'abc' select s.md5('deadbeef');";
    const mismatch_got = try runQuery(mismatch_q);
    try std.testing.expectEqualStrings("", mismatch_got.err);
    try std.testing.expectEqualStrings("false\n", mismatch_got.out);
}

test "compile+run hash-check unwraps nested query arg" {
    // Arrange
    const query =
        "from string s in 'abc' select s.md5(from string t in 'abc' select t.md5);";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check unwraps let-bound singleton seq arg" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let expected = from string t in 'abc' select t.md5
        \\select s.md5(expected);
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check empty seq arg reports TypeMismatch" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let expected = from string t in 'abc' where false select t.md5
        \\select s.md5(expected);
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch", got.err);
}

test "compile+run hash-check method is case-insensitive" {
    // Arrange
    const query = "from string s in 'abc' select s.md5('900150983CD24FB0D6963F7D28E17F72');";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check method in where filters" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\where s.md5('900150983cd24fb0d6963f7d28e17f72')
        \\select s;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("abc\n", got.out);

    const miss =
        \\from string s in 'abc'
        \\where s.md5('nope')
        \\select s;
    ;
    const miss_got = try runQuery(miss);
    try std.testing.expectEqualStrings("", miss_got.err);
    try std.testing.expectEqualStrings("", miss_got.out);
}

test "compile+run hash-check method with json record" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let valid = s.md5('900150983CD24FB0D6963F7D28E17F72')
        \\let result = { path = 'x', valid }
        \\select result.json();
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("{\"path\":\"x\",\"valid\":true}\n", got.out);
}

test "compile+run hash-check method on file" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "abc" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "x.txt");
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select f.md5('900150983cd24fb0d6963f7d28e17f72');",
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check method respects file window" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "xxabcyy" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try tmpFileQueryPath(std.testing.allocator, dir_path, "x.txt");
    defer std.testing.allocator.free(file_path);

    // window "abc" at offset 2, length 3 — same digest as string 'abc'
    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select f.offset(2).limit(3).md5('900150983cd24fb0d6963f7d28e17f72');",
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check wrong arity reports InvalidMethodArity" {
    // Arrange

    // Act
    const got = try runQuery("from string s in 'abc' select s.md5();");

    // Assert
    try std.testing.expectEqualStrings("wrong number of method arguments", got.err);
}

test "compile+run hash-check on dir reports InvalidMethodReceiver" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from dir d in '{s}' select d.md5('00');",
        .{dir_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid method receiver", got.err);
}

test "compile+run hash-check non-string arg reports TypeMismatch" {
    // Arrange

    // Act
    const got = try runQuery("from string s in 'abc' select s.md5(1);");

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run unknown method reports UnknownMethod" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let o = { a = s, b = s }
        \\select o.nope();
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("unknown method", got.err);
}

test "compile+run json with args reports InvalidMethodArity" {
    // Arrange
    const query =
        \\from string s in 'abc'
        \\let o = { a = s, b = s }
        \\select o.json(true);
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("wrong number of method arguments", got.err);
}

test "compile+run sfv wrong fields reports InvalidMethodFields" {
    // Arrange
    const missing_name =
        \\from string s in 'abc'
        \\let o = { path = '/tmp/x', digest = s.md5 }
        \\select o.sfv();
    ;

    // Act
    const got1 = try runQuery(missing_name);

    // Assert
    try std.testing.expectEqualStrings("record fields do not match method requirements", got1.err);

    const wrong_count =
        \\from string s in 'abc'
        \\let o = { a = s }
        \\select o.sfv();
    ;
    const got2 = try runQuery(wrong_count);
    try std.testing.expectEqualStrings("record fields do not match method requirements", got2.err);
}

test "compile+run bad regex is runtime error" {
    // Arrange
    const query = "from string s in 'abc' where s ~ '[' select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid regular expression", got.err);
}

test "compile+run join outer key cannot see join range" {
    // Arrange — outer key must use outer env only (§6.4)
    const query =
        \\from string a in 'abc'
        \\join string b in 'abc' on b.size equals b.size
        \\select a;
    ;

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("undefined name", got.err);
}

test "compile+run from string in file is invalid source type" {
    // Arrange — no File→String path coercion (§3.3)
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a.txt", .data = "a" });
    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}/a.txt' from string s in f select s;",
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "compile+run offset past EOF on empty file" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "empty", .data = "" });
    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}/empty' select f.offset(1).md5;",
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expect(std.mem.startsWith(u8, got.err, "Offset is greater than file size"));
}

test "compile+run where size filters empty before offset hash" {
    // Arrange — demand-driven: cheap `size` filter skips the window hash (§4.1)
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "empty", .data = "" });
    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}/empty' where f.size > 0 select f.offset(1).md5;",
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("", got.out);
}

test "compile+run terminal group by with Seq items is type mismatch" {
    // Arrange — sinking group Record must not expand `items` (§7)
    const query =
        "from string s in 'abc' group s by s.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch", got.err);
}

test "compile+run join with literal source over multiple outers" {
    // Arrange — stable literal inner must match each outer of size 1
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b", .data = "b" });
    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from dir d in '{s}'
        \\from file f in d
        \\join string s in 'x' on f.size equals s.size
        \\orderby f.path
        \\select f.size;
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("1\n1\n", got.out);
}

test "compile+run join after script into reuses stable source" {
    // Arrange — `files` is script-bound; join source is stable across outers
    var outer = std.testing.tmpDir(.{});
    defer outer.cleanup();
    var inner = std.testing.tmpDir(.{});
    defer inner.cleanup();
    try outer.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try outer.dir.writeFile(state.io, .{ .sub_path = "bb", .data = "bb" });
    try inner.dir.writeFile(state.io, .{ .sub_path = "x", .data = "x" });
    try inner.dir.writeFile(state.io, .{ .sub_path = "yy", .data = "yy" });

    const outer_path = try tmpQueryPath(std.testing.allocator, outer);
    defer std.testing.allocator.free(outer_path);
    const inner_path = try tmpQueryPath(std.testing.allocator, inner);
    defer std.testing.allocator.free(inner_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from dir id in '{s}' from file jf in id select jf into files;
        \\from dir od in '{s}'
        \\from file of in od
        \\join file jf in files on of.size equals jf.size
        \\orderby of.path
        \\select of.size;
    ,
        .{ inner_path, outer_path },
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("1\n2\n", got.out);
}

test "compile+run join source name shadowed by pipeline is not cached" {
    // Arrange — script `into src` must not make pipeline `src` look join-stable.
    // Each outer row binds a different `src`; caching the first inner would drop
    // the second match (one line instead of two).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "bb", .data = "bb" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from string _ in 's' select _ into src;
        \\from dir od in '{s}'
        \\from file of in od
        \\from string src in of.name
        \\join string j in src on of.name equals j
        \\orderby of.path
        \\select of.name;
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("a\nbb\n", got.out);
}

test "compile+run group join env survives let and orderby buffering" {
    // Arrange — `join ... into g` yields the outer row env plus `g`; the `let`
    // write and `orderby` buffering below hit that env across row-arena resets,
    // so its bindings must be parent-owned, not row-arena aliased.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "a", .data = "a" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "b", .data = "b" });
    try tmp.dir.writeFile(state.io, .{ .sub_path = "cc", .data = "cc" });

    const path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(path);

    const a_path = try std.fs.path.join(std.testing.allocator, &.{ path, "a" });
    defer std.testing.allocator.free(a_path);
    const b_path = try std.fs.path.join(std.testing.allocator, &.{ path, "b" });
    defer std.testing.allocator.free(b_path);
    const cc_path = try std.fs.path.join(std.testing.allocator, &.{ path, "cc" });
    defer std.testing.allocator.free(cc_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        \\from dir od in '{s}'
        \\from file of in od
        \\join file jf in od on of.size equals jf.size into g
        \\let n = g.count()
        \\orderby of.path
        \\select {{ of.path, n }};
    ,
        .{path},
    );
    defer std.testing.allocator.free(query);

    const expect = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}\n2\n{s}\n2\n{s}\n1\n",
        .{ a_path, b_path, cc_path },
    );
    defer std.testing.allocator.free(expect);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(expect, got.out);
}
