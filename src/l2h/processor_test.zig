//! GoogleTest ProcessorTest parity: the PCRE2-backed regex match exposed by the
//! l2h processor (processor.matchRe, the port of proc_match_re).

const std = @import("std");
const proc = @import("processor.zig");

test "ProcessorTest MatchSuccess" {
    try std.testing.expect(proc.matchRe("[0-9]+", "123"));
}

test "ProcessorTest MatchFailure" {
    try std.testing.expect(!proc.matchRe("[0-9]+", "num"));
}
