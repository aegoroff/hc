const std = @import("std");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");
const compile = @import("compile.zig");
const interpret = @import("interpret.zig");
const diag = @import("diag.zig");
const test_stderr = @import("test_stderr.zig");

var out_buf: [4096]u8 = undefined;
var out_writer: std.Io.Writer = undefined;

fn setup() void {
    state.gpa = std.testing.allocator;
    state.io = std.testing.io;
    out_writer = .fixed(&out_buf);
    state.out = &out_writer;
}

const RunResult = struct {
    out: []const u8,
    err: []const u8,
};

fn runQuery(query: []const u8) !RunResult {
    setup();
    state.source_name = "<query>";
    state.source_text = query;
    diag.clearLast();
    out_writer = .fixed(&out_buf);

    const saved_stderr = test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);

    const Callback = struct {
        fn cb(ast: ?*c.fend_node_t) callconv(.c) void {
            const root = ast orelse return;
            if (front.fend_error_count != 0) return;
            var arena = std.heap.ArenaAllocator.init(state.gpa);
            defer arena.deinit();

            const plan_root = compile.compileQuery(arena.allocator(), root) catch |err| {
                diag.report(diag.messageForCompile(err));
                return;
            };
            const ctx: interpret.Ctx = .{
                .allocator = arena.allocator(),
                .io = state.io,
                .out = state.writer(),
            };
            interpret.run(ctx, &plan_root) catch |err| {
                diag.report(diag.messageForRuntime(err));
            };
        }
    };

    front.fend_translation_unit_init(Callback.cb);
    defer front.fend_translation_unit_cleanup();

    front.fend_error_count = 0;
    const z = try state.gpa.dupeSentinel(u8, query, 0);
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
    _ = c.yyparse();
    return .{
        .out = std.Io.Writer.buffered(&out_writer),
        .err = diag.lastMessage(),
    };
}

fn tmpQueryPath(allocator: std.mem.Allocator, tmp: anytype) ![]u8 {
    return try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
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
        "from string s in '123' select s.sha1;\n"
        ++ "from string t in 'abc' select t.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        "40bd001563085fc35165329ea1ff5c5ecbdbbeef\n" ++
            "900150983cd24fb0d6963f7d28e17f72\n",
        got.out,
    );
}

test "compile+run multiple queries reuse range id" {
    // Arrange — each query resets identifier scope
    const query =
        "from string s in '123' select s.sha1;"
        ++ "from string s in 'abc' select s.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        "40bd001563085fc35165329ea1ff5c5ecbdbbeef\n" ++
            "900150983cd24fb0d6963f7d28e17f72\n",
        got.out,
    );
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
        "from string a in 'bb' "
        ++ "join string b in 'a' on a.size equals b.size "
        ++ "into g "
        ++ "from string x in g "
        ++ "orderby x.size descending "
        ++ "select x;";

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
        "from string s in from dir d in '{s}' from file f in d select f.path "
        ++ "orderby s.size "
        ++ "select s;",
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
        "from string s in from dir d in '{s}' from file f in d select f.path "
        ++ "orderby s.size descending "
        ++ "select s;",
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
        "from string s in from dir d in '{s}' from file f in d select f.path "
        ++ "group s by s.size;",
        .{path},
    );
    defer std.testing.allocator.free(query);

    const key1 = a_path.len;
    const key2 = cc_path.len;
    const expect = try std.fmt.allocPrint(
        std.testing.allocator,
        "{d}\n{s}\n{s}\n{d}\n{s}\n",
        .{ key1, a_path, b_path, key2, cc_path },
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
        "from string s in from dir d in '{s}' from file f in d select f.path "
        ++ "group s by s.size into g "
        ++ "select g.key;",
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
        "from dir d in '{s}' "
        ++ "from file f in d "
        ++ "group f by f.size into g "
        ++ "select g.key;",
        .{path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("1\n2\n", got.out);
}

test "compile+run terminal group by over directory" {
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
        "from dir d in '{s}' from file f in d group f by f.size;",
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
        "1\n{s}\n{s}\n2\n{s}\n",
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
        "from dir od in '{s}' "
        ++ "from file of in od "
        ++ "from dir id in '{s}' "
        ++ "join file jf in id on of.size equals jf.size into g "
        ++ "from file mf in g "
        ++ "select mf.size;",
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

test "compile+run undefined select name reports undefined name" {
    const query = "from string s in 'abc' select missing;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("undefined name", got.err);
}

test "compile+run nested query undefined name stays UndefinedName" {
    // Nested query plans are compiled (and typechecked) before eval, so the
    // failure surfaces at compilation.
    const query = "from string s in 'abc' where from string t in missing select t select s;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("undefined name", got.err);
}

test "plain hex-looking strings compare case-sensitively" {
    const query = "from string s in 'ab' where s == 'AB' select s;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run hex escapes hash as binary payload" {
    // Zig source doubles backslashes so the query still contains `\xNN` text.
    const query = "from string s in \"\\x00\\x01\\x02\" select s.md5;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("b95f67f61ebb03619622d798f45fc2d3\n", got.out);
}

test "compile+run hex escapes size is byte count" {
    const query = "from string s in \"\\xDE\\xAD\\xBE\\xEF\" select s.size;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("4\n", got.out);
}

test "compile+run invalid string escape reports error" {
    const query = "from string s in \"\\xZZ\" select s.md5;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("invalid string escape sequence", got.err);
}

test "hash property equals uppercase digest literal case-insensitively" {
    const query =
        "from string s in 'abc' "
        ++ "where s.md5 == '900150983CD24FB0D6963F7D28E17F72' "
        ++ "select s;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "invalid property span points at property expression" {
    const query = "from string s in 'abc' select s.nope;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
    // `s.nope` starts after "from string s in 'abc' select "
    try std.testing.expectEqual(@as(c_int, 1), diag.last_span.first_line);
    try std.testing.expectEqual(@as(c_int, 31), diag.last_span.first_column);
    try std.testing.expectEqual(@as(c_int, 37), diag.last_span.last_column);
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
    try std.testing.expectEqual(@as(c_int, 1), diag.last_span.first_line);
    try std.testing.expectEqual(@as(c_int, 16), diag.last_span.first_column);
}

test "compile+run file.path projects bound path" {
    // Arrange
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "x.txt" });
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
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "x.txt" });
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select f.name;",
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    const got = try runQuery(query);

    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("x.txt\n", got.out);
}

test "compile+run file sfv and checksum ignore declaration order" {
    // Digest field first in the object — output order is still fixed by method.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "x.txt" });
    defer std.testing.allocator.free(file_path);

    // md5("x") = 9dd4e461268c8034f5c8564e155c67a6
    const sfv_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' let o = {{ f.md5, f.name }} select o.sfv();",
        .{file_path},
    );
    defer std.testing.allocator.free(sfv_q);
    const sfv = try runQuery(sfv_q);
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
        "9dd4e461268c8034f5c8564e155c67a6    {s}\n",
        .{file_path},
    );
    defer std.testing.allocator.free(expect_sum);
    try std.testing.expectEqualStrings(expect_sum, sum.out);
}

test "compile+run record literal method call without let" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "x" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "x.txt" });
    defer std.testing.allocator.free(file_path);

    const sfv_q = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select {{ f.md5, f.name }}.sfv();",
        .{file_path},
    );
    defer std.testing.allocator.free(sfv_q);
    const sfv = try runQuery(sfv_q);
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
        "9dd4e461268c8034f5c8564e155c67a6    {s}\n",
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
    const file_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "part.txt" });
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' where f.offset == 2 && f.limit == 4 select f.md5;",
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("81b073de9370ea873f548e31b8adc081\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run file window bind after hash in conjunction" {
    // Arrange — binds under && apply before any hash in the tree (§4.5)
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "part.txt", .data = "0123456789" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "part.txt" });
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' where f.md5 == '81b073de9370ea873f548e31b8adc081' && f.offset == 2 && f.limit == 4 select f.size;",
        .{file_path},
    );
    defer std.testing.allocator.free(query);

    // Act
    const got = try runQuery(query);

    // Assert — full file size; window only affects the hash in where
    try std.testing.expectEqualStrings("10\n", got.out);
    try std.testing.expectEqualStrings("", got.err);
}

test "compile+run string.limit is invalid property" {
    // Arrange
    const query = "from string s in 'abc' where s.limit == 1 select s.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
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
        "from dir d in '{s}' "
        ++ "from file f in d.tree() "
        ++ "from file g in d "
        ++ "select g.size;",
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
    // Arrange / Act — dupe outs: runQuery reuses a shared buffer
    const select_true = try runQuery("from string s in 'a' select true;");
    const true_out = try std.testing.allocator.dupe(u8, select_true.out);
    defer std.testing.allocator.free(true_out);
    try std.testing.expectEqualStrings("", select_true.err);

    const select_false = try runQuery("from string s in 'a' select false;");
    const false_out = try std.testing.allocator.dupe(u8, select_false.out);
    defer std.testing.allocator.free(false_out);
    try std.testing.expectEqualStrings("", select_false.err);

    const where_true = try runQuery("from string s in 'a' where true select s.size;");
    const where_true_out = try std.testing.allocator.dupe(u8, where_true.out);
    defer std.testing.allocator.free(where_true_out);
    try std.testing.expectEqualStrings("", where_true.err);

    const where_false = try runQuery("from string s in 'a' where false select s.size;");
    try std.testing.expectEqualStrings("", where_false.err);
    try std.testing.expectEqualStrings("", where_false.out);

    // Assert
    try std.testing.expectEqualStrings("true\n", true_out);
    try std.testing.expectEqualStrings("false\n", false_out);
    try std.testing.expectEqualStrings("1\n", where_true_out);
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
    try std.testing.expectEqual(@as(c_int, 1), diag.last_span.first_line);
    try std.testing.expectEqual(@as(c_int, 58), diag.last_span.first_column);
}

test "compile+run into md5 then restore as sha1 reports invalid digest" {
    // Arrange
    const query =
        "from string s in '123' select s.md5 into h123 "
        ++ "from hash h in h123 select h.sha1;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid hash digest for the selected algorithm", got.err);
}

test "compile+run invalid group property fails during compilation" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "group s by s.size into g "
        ++ "select g.nope;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run typed record field access works" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let r = { s, s.size } "
        ++ "orderby r.size descending "
        ++ "select r.s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run explicit record alias and auto-name mix works" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let r = { digest = s.md5, s.size } "
        ++ "select r.digest;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run missing typed record field fails during compilation" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let r = { s } "
        ++ "select r.nope;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run duplicate record field fails during compilation" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let r = { digest = s.md5, digest = s.sha1 } "
        ++ "select r.digest;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("duplicate record field name", got.err);
}

test "compile+run nested query in let produces sequence value" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let items = from string t in s select t.md5 "
        ++ "select items;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run nested query in select produces sequence value" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "select from string t in s select t;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run nested query in record field works" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let r = { items = from string t in s select t.md5 } "
        ++ "select r.items;";

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
        "from string a in 'abc' "
        ++ "join string b in 'x' on a equals b.size "
        ++ "select a;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run orderby key must be comparable" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "group s by s.size into g "
        ++ "orderby g.items "
        ++ "select g.key;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run group items property access stays typed" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "group s by s.size into g "
        ++ "from string item in g.items "
        ++ "select item;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run group by record key fails during compilation" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "group s by { s } into g "
        ++ "select g.key;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run from file in string sequence fails during compilation" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let xs = from string t in s select t "
        ++ "from file f in xs "
        ++ "select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "compile+run nested query as where exists predicate" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "where from string t in 'ab' where t.size == s.size select t "
        ++ "select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("", got.out);
}

test "compile+run nested query where exists keeps matching row" {
    // Arrange
    const query =
        "from string s in 'ab' "
        ++ "where from string t in 'xy' where t.size == s.size select t "
        ++ "select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("ab\n", got.out);
}

test "compile+run nested query in orderby singleton unwrap" {
    // Arrange
    const query =
        "from string s in 'bb' "
        ++ "from string t in 'a' "
        ++ "orderby from string x in t select x.size "
        ++ "select t;";

    // Act
    const got = try runQuery(query);

    // Assert
    // Cartesian product of singletons: one row with t='a', ordered by nested size.
    try std.testing.expectEqualStrings("a\n", got.out);
}

test "compile+run from in nested query sequence" {
    // Arrange
    const query =
        "from string x in from string t in 'abc' select t "
        ++ "select x;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run join in nested query sequence" {
    // Arrange
    const query =
        "from string a in 'ab' "
        ++ "join string b in from string t in 'xy' select t "
        ++ "on a.size equals b.size "
        ++ "select a;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("ab\n", got.out);
}

test "compile+run join key nested query singleton unwrap" {
    // Arrange
    const query =
        "from string a in 'abc' "
        ++ "join string b in 'xyz' "
        ++ "on a.size equals from string t in b select t.size "
        ++ "select a;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run group by nested query key" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "group s by from string t in s select t.size into g "
        ++ "select g.key;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("3\n", got.out);
}

test "compile+run from in nested query wrong item kind fails during compilation" {
    // Arrange
    const query =
        "from file f in from string t in 'abc' select t "
        ++ "select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "compile+run nested query uses outer binding in inner source" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let xs = from string t in s select t "
        ++ "from string x in xs "
        ++ "select x;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run invalid property on nested sequence fails during compilation" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "let items = from string t in s select t "
        ++ "select items.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "compile+run shallow nested query succeeds within depth limit" {
    // A handful of nesting levels is well within MAX_QUERY_DEPTH and must
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

    const got = try runQuery(std.Io.Writer.buffered(&fbs));

    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "compile+run deeply nested query reports QueryTooDeep" {
    // Adversarial nesting (a select whose value is itself a query, repeated
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

    const got = try runQuery(std.Io.Writer.buffered(&fbs));

    try std.testing.expectEqualStrings("query nesting too deep", got.err);
}

test "compile+run record sfv via let" {
    const query =
        "from string s in 'abc' "
        ++ "let o = { name = 'x', digest = s.md5 } "
        ++ "select o.sfv();";

    const got = try runQuery(query);

    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("x    900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "compile+run record checksum via into" {
    const query =
        "from string s in 'abc' "
        ++ "select { path = '/tmp/x', digest = s.md5 } into o "
        ++ "select o.checksum();";

    const got = try runQuery(query);

    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72    /tmp/x\n", got.out);
}

test "compile+run record json and jsonPretty" {
    const compact_q =
        "from string s in 'abc' "
        ++ "let o = { a = 'x', n = s.size } "
        ++ "select o.json();";
    const compact = try runQuery(compact_q);
    try std.testing.expectEqualStrings("", compact.err);
    try std.testing.expectEqualStrings("{\"a\":\"x\",\"n\":3}\n", compact.out);

    const pretty_q =
        "from string s in 'abc' "
        ++ "let o = { a = 'x', n = s.size } "
        ++ "select o.jsonPretty();";
    const pretty = try runQuery(pretty_q);
    try std.testing.expectEqualStrings("", pretty.err);
    try std.testing.expectEqualStrings("{\n  \"a\": \"x\",\n  \"n\": 3\n}\n", pretty.out);
}

test "compile+run jsonPretty allows nested record fields" {
    const query =
        "from string s in 'abc' "
        ++ "let hashes = { digest = s.md5, n = s.size } "
        ++ "select { path = 'x', hashes } into o "
        ++ "select o.json();";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings(
        "{\"path\":\"x\",\"hashes\":{\"digest\":\"900150983cd24fb0d6963f7d28e17f72\",\"n\":3}}\n",
        got.out,
    );
}

test "compile+run record csv spaced tabbed" {
    const csv_q =
        "from string s in 'abc' "
        ++ "let o = { a = 'one', b = 'two' } "
        ++ "select o.csv();";
    const csv = try runQuery(csv_q);
    try std.testing.expectEqualStrings("", csv.err);
    try std.testing.expectEqualStrings("one,two\n", csv.out);

    const spaced_q =
        "from string s in 'abc' "
        ++ "let o = { a = 'one', b = 'two' } "
        ++ "select o.spaced();";
    const spaced = try runQuery(spaced_q);
    try std.testing.expectEqualStrings("", spaced.err);
    try std.testing.expectEqualStrings("one two\n", spaced.out);

    const tabbed_q =
        "from string s in 'abc' "
        ++ "let o = { a = 'one', b = 'two' } "
        ++ "select o.tabbed();";
    const tabbed = try runQuery(tabbed_q);
    try std.testing.expectEqualStrings("", tabbed.err);
    try std.testing.expectEqualStrings("one\ttwo\n", tabbed.out);
}

test "compile+run bare record still prints one line per field" {
    const query =
        "from string s in 'abc' "
        ++ "select { a = '1', b = '2' };";

    const got = try runQuery(query);

    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("1\n2\n", got.out);
}

test "compile+run method on non-record reports InvalidMethodReceiver" {
    const got = try runQuery("from string s in 'abc' select s.sfv();");
    try std.testing.expectEqualStrings("invalid method receiver", got.err);
}

test "compile+run hash-check method on string match and mismatch" {
    const match_q = "from string s in 'abc' select s.md5('900150983cd24fb0d6963f7d28e17f72');";
    const match_got = try runQuery(match_q);
    try std.testing.expectEqualStrings("", match_got.err);
    try std.testing.expectEqualStrings("true\n", match_got.out);

    const mismatch_q = "from string s in 'abc' select s.md5('deadbeef');";
    const mismatch_got = try runQuery(mismatch_q);
    try std.testing.expectEqualStrings("", mismatch_got.err);
    try std.testing.expectEqualStrings("false\n", mismatch_got.out);
}

test "compile+run hash-check unwraps nested query arg" {
    const query =
        "from string s in 'abc' "
        ++ "select s.md5(from string t in 'abc' select t.md5);";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check unwraps let-bound singleton seq arg" {
    const query =
        "from string s in 'abc' "
        ++ "let expected = from string t in 'abc' select t.md5 "
        ++ "select s.md5(expected);";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check empty seq arg reports TypeMismatch" {
    const query =
        "from string s in 'abc' "
        ++ "let expected = from string t in 'abc' where false select t.md5 "
        ++ "select s.md5(expected);";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("type mismatch", got.err);
}

test "compile+run hash-check method is case-insensitive" {
    const query = "from string s in 'abc' select s.md5('900150983CD24FB0D6963F7D28E17F72');";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check method in where filters" {
    const query =
        "from string s in 'abc' "
        ++ "where s.md5('900150983cd24fb0d6963f7d28e17f72') "
        ++ "select s;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("abc\n", got.out);

    const miss =
        "from string s in 'abc' "
        ++ "where s.md5('nope') "
        ++ "select s;";
    const miss_got = try runQuery(miss);
    try std.testing.expectEqualStrings("", miss_got.err);
    try std.testing.expectEqualStrings("", miss_got.out);
}

test "compile+run hash-check method with json record" {
    const query =
        "from string s in 'abc' "
        ++ "let valid = s.md5('900150983CD24FB0D6963F7D28E17F72') "
        ++ "let result = { path = 'x', valid } "
        ++ "select result.json();";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("{\"path\":\"x\",\"valid\":true}\n", got.out);
}

test "compile+run hash-check method on file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "abc" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "x.txt" });
    defer std.testing.allocator.free(file_path);

    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' select f.md5('900150983cd24fb0d6963f7d28e17f72');",
        .{file_path},
    );
    defer std.testing.allocator.free(query);
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check method respects file window" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(state.io, .{ .sub_path = "x.txt", .data = "xxabcyy" });

    const dir_path = try tmpQueryPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(dir_path);
    const file_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "x.txt" });
    defer std.testing.allocator.free(file_path);

    // window "abc" at offset 2, length 3 — same digest as string 'abc'
    const query = try std.fmt.allocPrint(
        std.testing.allocator,
        "from file f in '{s}' where f.offset == 2 && f.limit == 3 "
        ++ "select f.md5('900150983cd24fb0d6963f7d28e17f72');",
        .{file_path},
    );
    defer std.testing.allocator.free(query);
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("", got.err);
    try std.testing.expectEqualStrings("true\n", got.out);
}

test "compile+run hash-check wrong arity reports InvalidMethodArity" {
    const got = try runQuery("from string s in 'abc' select s.md5();");
    try std.testing.expectEqualStrings("wrong number of method arguments", got.err);
}

test "compile+run hash-check on dir reports InvalidMethodReceiver" {
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
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("invalid method receiver", got.err);
}

test "compile+run hash-check non-string arg reports TypeMismatch" {
    const got = try runQuery("from string s in 'abc' select s.md5(1);");
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "compile+run unknown method reports UnknownMethod" {
    const query =
        "from string s in 'abc' "
        ++ "let o = { a = s, b = s } "
        ++ "select o.nope();";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("unknown method", got.err);
}

test "compile+run json with args reports InvalidMethodArity" {
    const query =
        "from string s in 'abc' "
        ++ "let o = { a = s, b = s } "
        ++ "select o.json(true);";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("wrong number of method arguments", got.err);
}

test "compile+run sfv wrong fields reports InvalidMethodFields" {
    const missing_name =
        "from string s in 'abc' "
        ++ "let o = { path = '/tmp/x', digest = s.md5 } "
        ++ "select o.sfv();";
    const got1 = try runQuery(missing_name);
    try std.testing.expectEqualStrings("record fields do not match method requirements", got1.err);

    const wrong_count =
        "from string s in 'abc' "
        ++ "let o = { a = s } "
        ++ "select o.sfv();";
    const got2 = try runQuery(wrong_count);
    try std.testing.expectEqualStrings("record fields do not match method requirements", got2.err);
}
