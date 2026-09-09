//! Fuzz l2h query syntax via the CLI entry point.
//!
//! Each iteration builds `l2h -n -q <bytes>` and calls `main.run`, so the path
//! matches production: yazap → parseQuery → compileQuery, with interpret skipped
//! by `--syntax-check`.
//!
//! Input is a Smith slice (u32 little-endian length + bytes), same in fuzz and
//! smoke-test modes. Corpus entries below are raw query strings wrapped with
//! that length prefix.
//!
//! Invariants:
//!   - panic / abort are not allowed
//!   - memory leak is not allowed (arena + testing allocator)
//!   - parse / compile errors are expected, not a bug

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

/// Smoke seeds: valid queries, known syntax failures, and a few edge cases.
const corpus = [_][]const u8{
    sliceCorpus(""),
    sliceCorpus("# comment only"),
    sliceCorpus("from string s in 'abc' where s.size > 0 select s.md5;"),
    sliceCorpus("from file x in 'dfg' select x.md5;"),
    sliceCorpus("from file x in 'dfg' select x.md5"),
    sliceCorpus("from file x in 'dfg select x.md5;"),
    sliceCorpus("from string a in 'abc' join string b in 'abc' on a.md5 equals b.md5 select a.md5;"),
    sliceCorpus("from hash x in '202CB962AC59075B964B07152D234B70' select x.md5;"),
    sliceCorpus("from string x in b\"\\xDE\\xAD\\xBE\\xEF\" select x.md5;"),
    sliceCorpus("from string s in 'a' select s.no_such_prop;"),
    sliceCorpus("from dir d in '/tmp' from file f in d.tree() select f.sha1;"),
    sliceCorpus("select;"),
    sliceCorpus("from string s in 'a' select s.md5 into h select h;"),
    sliceCorpus("from string s in '" ++ "a" ** 256 ++ "' select s.size;"),
};

fn fuzzOne(_: void, smith: *std.testing.Smith) anyerror!void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var query_buf: [max_query_len]u8 = undefined;
    const query_len = smith.slice(&query_buf);
    const query = query_buf[0..query_len];
    // `-q` argv is a Zig sentinel slice; embedded NULs confuse C-facing CLI
    // parsing and the fuzz runner's stdio. Skip those inputs.
    if (std.mem.indexOfScalar(u8, query, 0) != null) return error.SkipZigTest;

    const query_z = try gpa.dupeSentinel(u8, query, 0);

    var out_buf: [4096]u8 = undefined;
    var out: std.Io.Writer = .fixed(&out_buf);

    // Do not mute stderr under `--fuzz`: the fuzzer talks to the build runner
    // over stdio; dup2'ing STDERR_FILENO breaks that channel.
    const saved_stderr = if (builtin.fuzz) @as(c_int, -1) else test_stderr.mute();
    defer if (saved_stderr >= 0) test_stderr.restore(saved_stderr);

    const argv = [_][:0]const u8{ "-n", "-q", query_z };
    driver.run(gpa, &out, std.testing.io, &argv) catch {};
}

test "fuzz query syntax-check via -q" {
    try std.testing.fuzz({}, fuzzOne, .{
        .corpus = &corpus,
    });
}
