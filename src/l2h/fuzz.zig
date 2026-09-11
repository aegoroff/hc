//! Fuzz l2h queries via the CLI entry point.
//!
//! Each iteration builds `l2h -q <bytes>` and calls `driver.run`:
//! yazap → parse → compile → interpret. String digests use Zig std via
//! `fuzz_stub/hashes`; file/dir I/O and hash-restore stay stubbed
//! (`fuzz_stub/modes`).
//!
//! Input is a Smith slice (u32 little-endian length + bytes), same in fuzz and
//! smoke-test modes. Corpus entries below are raw query strings wrapped with
//! that length prefix.
//!
//! Invariants:
//!   - panic / abort are not allowed
//!   - memory leak is not allowed (arena + testing allocator)
//!   - parse / compile / I/O errors are expected, not a bug

const std = @import("std");
const builtin = @import("builtin");
const driver = @import("driver.zig");
const test_stderr = @import("test_stderr.zig");

const max_query_len: u32 = 8 * 1024;

fn sliceCorpus(comptime query: []const u8) *const [4 + query.len]u8 {
    const Storage = struct {
        const bytes: [4 + query.len]u8 = blk: {
            var buf: [4 + query.len]u8 = undefined;
            std.mem.writeInt(u32, buf[0..4], @intCast(query.len), .little);
            @memcpy(buf[4..], query);
            break :blk buf;
        };
    };
    return &Storage.bytes;
}

/// Coverage-guided mutation needs a seed that already reaches each grammar
/// production. Byte flips almost never invent keywords (`orderby`, `group`,
/// `equals`, `into`, …), so unique-run growth stalls on a tiny similar corpus.
const corpus = [_][]const u8{
    // empty / trivia
    sliceCorpus(""),
    sliceCorpus("# comment only"),
    sliceCorpus("# lead\r\nfrom string x in '123' select x.md5;"),
    sliceCorpus("from file x in \n'dfg' \nselect x.md5;"),

    // sources
    sliceCorpus("from string s in 'abc' select s.md5;"),
    sliceCorpus("from file x in 'dfg' select x.md5;"),
    sliceCorpus("from dir x in '/tmp' select x.sha1;"),
    sliceCorpus("from hash x in '202CB962AC59075B964B07152D234B70' select x.md5;"),
    sliceCorpus("from dir d in '/tmp' from file f in d select f.sha1;"),
    sliceCorpus("from dir d in '/tmp' from file f in d.tree() select f.sha1;"),
    sliceCorpus("from dir d in '/tmp' from file f in d.tree(1).skipErrors() select f.path;"),

    // clauses
    sliceCorpus("from string s in 'abc' where s.size > 0 select s.md5;"),
    sliceCorpus("from string s in 'abc' let d = s.md5 select d;"),
    sliceCorpus("from string a in 'abc' join string b in 'abc' on a.md5 equals b.md5 select a.md5;"),
    sliceCorpus(
        \\from string a in 'bb'
        \\join string b in 'a' on a.size equals b.size
        \\into g
        \\from string x in g
        \\orderby x.size descending
        \\select x;
    ),
    sliceCorpus("from string s in 'abc' orderby s.size ascending, s.md5 descending select s;"),
    sliceCorpus("from string s in 'abc' group s by s.size;"),
    sliceCorpus("from string s in 'abc' group s by s.size into g;"),
    sliceCorpus("from string s in 'a' select s.md5 into h select h;"),
    sliceCorpus("from string s in 'abc' select s.md5 into h;"),
    sliceCorpus(
        \\from string s in 'abc' select s.md5 into h;
        \\from string t in 'xyz' where t.md5 != h select t;
    ),

    // Real string digests (fuzz_stub/hashes via std) — hit compare / check /
    // group / join paths that zero-digests never distinguished.
    sliceCorpus("from string s in 'abc' where s.md5 == '900150983CD24FB0D6963F7D28E17F72' select s;"),
    sliceCorpus("from string s in 'abc' where s.md5 == '900150983cd24fb0d6963f7d28e17f72' select s;"),
    sliceCorpus("from string s in 'abc' where s.md5 != '00000000000000000000000000000000' select s;"),
    sliceCorpus("from string s in 'abc' where s.md5('900150983CD24FB0D6963F7D28E17F72') select s;"),
    sliceCorpus("from string s in 'abc' where s.md5('00000000000000000000000000000000') select s;"),
    sliceCorpus("from string s in '' select s.md5;"),
    sliceCorpus("from string s in 'abc' select s.sha256;"),
    sliceCorpus("from string s in 'abc' select s.blake3;"),
    sliceCorpus("from string s in 'abc' select s.crc32;"),
    sliceCorpus("from string s in 'abc' select s.xxhash32;"),
    sliceCorpus("from string s in 'abc' select { s.sha1, s.sha256, s.md5 };"),
    sliceCorpus("from string s in 'abc' group s by s.md5 into g select g.key;"),
    sliceCorpus(
        \\from string a in 'abc'
        \\join string b in 'abc' on a.md5 equals b.md5
        \\select a.md5;
    ),
    sliceCorpus(
        \\from string a in 'abc'
        \\join string b in 'xyz' on a.md5 equals b.md5
        \\select a;
    ),
    sliceCorpus(
        \\from string s in 'abc' select s.md5 into h;
        \\from string t in 'abc' where t.md5 == h select t;
    ),
    sliceCorpus(
        \\from string s in 'a'
        \\let d = s.md5
        \\where d == '0CC175B9C0F1B6A831C399E269772661'
        \\select d;
    ),
    sliceCorpus("from string s in 'abc' where s.size > 0 && s.md5('900150983CD24FB0D6963F7D28E17F72') select s;"),
    sliceCorpus("from string s in 'abc' where false && s.md5('00000000000000000000000000000000') select s;"),
    sliceCorpus("from string s in 'abc' select { md5 = s.md5, sha = s.sha256 };"),
    sliceCorpus("from string s in 'abc' orderby s.md5 ascending select s.md5;"),

    // nested queries
    sliceCorpus("from string s in 'abc' let items = from string t in s select t select items.count();"),
    sliceCorpus("from string s in 'abc' where from string t in s where false select t select s;"),

    // expressions / literals
    sliceCorpus("from file x in 'dfg' select { x.md5, x.md2 };"),
    sliceCorpus("from string a in 'abc' select { digest = a.md5, len = a.size };"),
    sliceCorpus("from file f in '/tmp/a' select { f.crc32, f.name }.sfv();"),
    sliceCorpus("from file f in '/tmp/a' let o = { f.path, f.crc32 } select o.checksum();"),
    sliceCorpus("from string s in 'abc' select { path = 'x', valid = true }.json();"),
    sliceCorpus("from file f in 'x' select f.offset(2).limit(4).md5;"),
    sliceCorpus("from hash x in '202CB962AC59075B964B07152D234B70' select x.dict('0123456789').min(1).max(3).noProbe().md5;"),
    sliceCorpus("from file f in 'x' where f.md5('529DF104CA7D7EC2E4B9E4EAB5557CF8') select f.path;"),
    sliceCorpus("from string s in 'abc123' where s ~ '[0-9]+' select s;"),
    sliceCorpus("from string s in 'abc' where s !~ '[0-9]+' select s;"),
    sliceCorpus("from string s in 'abc' where (s.size == 2 || s.size == 3) && !(s ~ 'x') select s;"),
    sliceCorpus("from string s in 'abc' where true || false select s;"),
    sliceCorpus("from string s in 'abc' where s.size >= 0 && s.size <= 10 && s.size != 1 select s;"),
    sliceCorpus("from string s in 'abc' select s.sha-3-256;"),
    sliceCorpus("from string s in 'abc' where s.size > -1 select s;"),
    sliceCorpus("from dir d in '/tmp' from file f in d.tree(-1) select f.path;"),
    sliceCorpus("from string x in b\"\\xDE\\xAD\\xBE\\xEF\" select x.md5;"),
    sliceCorpus("from string x in b'\\xef\\xbb\\xbf' select x.md5;"),
    sliceCorpus("from string s in \"abc\" select s.size;"),
    sliceCorpus("from string s in 'c:\\Windows' select s.size;"),
    sliceCorpus("from file x in 'dfg' select x.m(1, '123');"),
    sliceCorpus("from string s in '" ++ "a" ** 256 ++ "' select s.size;"),

    // syntax / semantic failures (distinct lexer and parser recoveries)
    sliceCorpus("from file x in 'dfg' select x.md5"),
    sliceCorpus("from file x in 'dfg select x.md5;"),
    sliceCorpus("from file x in\n 'dfg'\n select x.md5"),
    sliceCorpus("select;"),
    sliceCorpus("select x.md4 from file x in 'dfg' select x.md5;"),
    sliceCorpus("from file x in 'dfg' select y.md5;"),
    sliceCorpus("from potato x in 'abc' select x.md5;"),
    sliceCorpus("from File x in 'dfg' select x.md5;"),
    sliceCorpus("from string s in 'a' select s.no_such_prop;"),
    sliceCorpus("from string s in"),
    sliceCorpus("from string s in 'abc' select {};"),
    sliceCorpus("from string s in b\"\\xZZ\" select s.md5;"),
    sliceCorpus("from file f in 1 select f.size;"),
    sliceCorpus("from string s in 'abc' where s.size ~ 'x' select s;"),
    sliceCorpus(&.{0x81}),
    sliceCorpus(&.{0xff}),
    sliceCorpus(&.{ 0x9f, 0x03, 0x18, 0x19, 0x0f, '#', ' ', 'c' }),
};

/// Prefer short printable queries; keep high bytes rare (lexer 8-bit paths)
/// and drop NUL so `-q` argv is never a truncated C string.
const query_len_weights = [_]std.testing.Smith.Weight{
    .rangeAtMost(u32, 0, 64, 8),
    .rangeAtMost(u32, 0, 256, 4),
    .rangeAtMost(u32, 0, max_query_len, 1),
};

const query_byte_weights = [_]std.testing.Smith.Weight{
    .rangeAtMost(u8, 1, 255, 1),
    .rangeAtMost(u8, ' ', '~', 16),
    .value(u8, '\n', 8),
    .value(u8, '\r', 4),
    .value(u8, '\t', 4),
};

fn fuzzOne(_: void, smith: *std.testing.Smith) anyerror!void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var query_buf: [max_query_len]u8 = undefined;
    const query_len = smith.sliceWeighted(&query_buf, &query_len_weights, &query_byte_weights);
    const query = query_buf[0..query_len];
    // `-q` argv is a Zig sentinel slice; embedded NULs confuse C-facing CLI
    // parsing and the fuzz runner's stdio. Skip those inputs.
    if (std.mem.indexOfScalar(u8, query, 0) != null) return error.SkipZigTest;

    const query_z = try gpa.dupeSentinel(u8, query, 0);

    var out_buf: [4096]u8 = undefined;
    var out: std.Io.Writer = .fixed(&out_buf);

    // Smoke mode: mute stderr so corpus failures stay quiet.
    // `--fuzz`: do not dup2 STDERR — Zig's FuzzTestRunner pipes it and keeps
    // every byte; diag skips fehler when `builtin.fuzz` instead.
    const saved_stderr = if (builtin.fuzz) @as(c_int, -1) else test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);

    const argv = [_][:0]const u8{ "-q", query_z };
    driver.run(gpa, &out, std.testing.io, &argv) catch {};
}

test "fuzz query via -q" {
    try std.testing.fuzz({}, fuzzOne, .{
        .corpus = &corpus,
    });
}
