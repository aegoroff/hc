const std = @import("std");
const builtin = @import("builtin");
const c = @import("c");
const state = @import("state.zig");
const front = @import("frontend.zig");
const lower = @import("lower.zig");
const interpret = @import("interpret.zig");
const diag = @import("diag.zig");

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

fn muteStderr() c_int {
    // POSIX-only (std.c.open flag type / fd_t as c_int). On Windows the early
    // return is comptime-taken so the body is not analyzed; intentional parse
    // noise may leak to stderr (cosmetic — tests assert on out/err strings).
    if (builtin.os.tag == .windows) return -1;
    const null_fd = std.c.open("/dev/null", .{ .ACCMODE = .WRONLY });
    if (null_fd < 0) return -1;
    const saved = std.c.dup(std.posix.STDERR_FILENO);
    if (saved < 0) {
        _ = std.c.close(null_fd);
        return -1;
    }
    _ = std.c.dup2(null_fd, std.posix.STDERR_FILENO);
    _ = std.c.close(null_fd);
    return saved;
}

fn restoreStderr(saved: c_int) void {
    if (builtin.os.tag == .windows) return;
    _ = std.c.dup2(saved, std.posix.STDERR_FILENO);
    _ = std.c.close(saved);
}

fn runQuery(query: []const u8) !RunResult {
    setup();
    state.source_name = "<query>";
    state.source_text = query;
    diag.clearLast();
    out_writer = .fixed(&out_buf);

    const saved_stderr = muteStderr();
    defer if (saved_stderr >= 0) restoreStderr(saved_stderr);

    const Callback = struct {
        fn cb(ast: ?*c.fend_node_t) callconv(.c) void {
            const root = ast orelse return;
            if (front.fend_error_count != 0) return;
            var arena = std.heap.ArenaAllocator.init(state.gpa);
            defer arena.deinit();

            const plan_root = lower.lowerQuery(arena.allocator(), root) catch |err| {
                diag.report(diag.messageForLower(err));
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

test "lower+run where/select query string" {
    // Arrange
    const query = "from string s in 'abc' where s.size > 0 select s.md5;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "lower+run multiple top-level queries" {
    // Arrange — semantics §4: several semicolon-separated queries in one unit
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

test "lower+run multiple queries reuse range id" {
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

test "lower+run let/into query string" {
    // Arrange
    const query = "from string s in 'abc' let d = s.md5 select d into h select h;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("900150983cd24fb0d6963f7d28e17f72\n", got.out);
}

test "lower+run join/orderby query string" {
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

test "lower+run regex where query string" {
    // Arrange
    const query = "from string s in 'abc123' where s ~ '[0-9]+' select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc123\n", got.out);
}

test "lower+run dir from file orderby skips symlink" {
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

test "lower+run group by into over directory" {
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

test "lower+run terminal group by over directory" {
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

test "lower+run join into over file sources" {
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

test "lower+run invalid property reports runtime error" {
    // Arrange
    const query = "from string s in 'abc' select s.nope;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("invalid property for this value type", got.err);
}

test "lower+run undefined select name reports undefined name" {
    const query = "from string s in 'abc' select missing;";
    const got = try runQuery(query);
    try std.testing.expectEqualStrings("undefined name", got.err);
}

test "lower+run nested query undefined name is not NotImplemented" {
    // Nested queries are re-lowered at eval; mapping must not collapse this to NotImplemented.
    // Static infer also lowers the nested AST, so the failure surfaces at lowering today.
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

test "lower+run from file in non-dir variable reports type mismatch" {
    // Arrange
    const query = "from string d in 'abc' from file f in d select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "lower+run missing file reports io failure" {
    // Arrange
    const query = "from file f in '/definitely-missing-l2h-test-path' select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("I/O failure (missing path or unreadable file/directory)", got.err);
    // Path literal in `from file f in '…'`
    try std.testing.expectEqual(@as(c_int, 1), diag.last_span.first_line);
    try std.testing.expectEqual(@as(c_int, 16), diag.last_span.first_column);
}

test "lower+run invalid group property fails during lowering" {
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

test "lower+run typed record field access works" {
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

test "lower+run explicit record alias and auto-name mix works" {
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

test "lower+run missing typed record field fails during lowering" {
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

test "lower+run duplicate record field fails during lowering" {
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

test "lower+run nested query in let produces sequence value" {
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

test "lower+run nested query in select produces sequence value" {
    // Arrange
    const query =
        "from string s in 'abc' "
        ++ "select from string t in s select t;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "lower+run nested query in record field works" {
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

test "lower+run match operand mismatch fails during lowering" {
    // Arrange
    const query = "from string s in 'abc' where s.size ~ 'x' select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "lower+run from file over int source fails during lowering" {
    // Arrange
    const query = "from file f in 1 select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "lower+run equality operand mismatch fails during lowering" {
    // Arrange
    const query = "from string s in 'abc' where s == 1 select s;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("type mismatch in expression or clause", got.err);
}

test "lower+run join key mismatch fails during lowering" {
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

test "lower+run orderby key must be comparable" {
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

test "lower+run group items property access stays typed" {
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

test "lower+run group by record key fails during lowering" {
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

test "lower+run from file in string sequence fails during lowering" {
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

test "lower+run nested query as where exists predicate" {
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

test "lower+run nested query where exists keeps matching row" {
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

test "lower+run nested query in orderby singleton unwrap" {
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

test "lower+run from in nested query sequence" {
    // Arrange
    const query =
        "from string x in from string t in 'abc' select t "
        ++ "select x;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("abc\n", got.out);
}

test "lower+run join in nested query sequence" {
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

test "lower+run join key nested query singleton unwrap" {
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

test "lower+run group by nested query key" {
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

test "lower+run from in nested query wrong item kind fails during lowering" {
    // Arrange
    const query =
        "from file f in from string t in 'abc' select t "
        ++ "select f.size;";

    // Act
    const got = try runQuery(query);

    // Assert
    try std.testing.expectEqualStrings("source expression type does not match the declared range kind", got.err);
}

test "lower+run nested query uses outer binding in inner source" {
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

test "lower+run invalid property on nested sequence fails during lowering" {
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
