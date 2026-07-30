const std = @import("std");
const re = @import("re");

// PCRE2 match for relational ~ / !~ (port of proc_match_re).
//
// pcre2.h width-suffixed macros do not translate via translate-c; call the _8
// entry points directly.

// Backstop against catastrophic backtracking / stack exhaustion on adversarial
// patterns (ReDoS). Mirrors pcre2grep's defaults so legitimate matches are not
// truncated; a pathological pattern returns no-match instead of hanging.
const PCRE2_MATCH_LIMIT: u32 = 1_000_000;
const PCRE2_DEPTH_LIMIT: u32 = 1000;

pub fn matchRe(pattern: []const u8, subject: []const u8) bool {
    var errnumber: c_int = 0;
    var erroffset: usize = 0;
    const compiled = re.pcre2_compile_8(pattern.ptr, pattern.len, 0, &errnumber, &erroffset, null) orelse return false;
    defer _ = re.pcre2_code_free_8(compiled);

    const match_data = re.pcre2_match_data_create_from_pattern_8(compiled, null) orelse return false;
    defer _ = re.pcre2_match_data_free_8(match_data);

    // Cap backtracking so a hostile pattern/subject cannot exhaust CPU or stack.
    const mctx = re.pcre2_match_context_create_8(null) orelse return false;
    defer _ = re.pcre2_match_context_free_8(mctx);
    _ = re.pcre2_set_match_limit_8(mctx, PCRE2_MATCH_LIMIT);
    _ = re.pcre2_set_depth_limit_8(mctx, PCRE2_DEPTH_LIMIT);

    var flags: u32 = re.PCRE2_NOTEMPTY;
    if (std.mem.indexOfScalar(u8, subject, '^') == null) flags |= re.PCRE2_NOTBOL;
    if (std.mem.indexOfScalar(u8, subject, '$') == null) flags |= re.PCRE2_NOTEOL;

    // rc >= 0 => match;  PCRE2_ERROR_NOMATCH (-1) and the limit errors
    // (-47/-53) are all < 0, so a capped runaway pattern yields false here.
    const rc = re.pcre2_match_8(compiled, subject.ptr, subject.len, 0, flags, match_data, mctx);
    return rc >= 0;
}

test "MatchSuccess" {
    // Arrange
    const pattern = "[0-9]+";
    const subject = "123";

    // Act
    const ok = matchRe(pattern, subject);

    // Assert
    try std.testing.expect(ok);
}

test "MatchFailure" {
    // Arrange
    const pattern = "[0-9]+";
    const subject = "num";

    // Act
    const ok = matchRe(pattern, subject);

    // Assert
    try std.testing.expect(!ok);
}

test "CatastrophicBacktrackingIsCapped" {
    // A classic exponential pattern (a+)+ against a non-matching tail would,
    // without a match limit, backtrack ~2^N times. The match/depth cap must
    // bound it to a prompt false instead of hanging.
    const pattern = "(a+)+$";
    const subject = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"; // trailing 'b' defeats $

    const ok = matchRe(pattern, subject);

    // No hang (this test reaches the assertion) and no match.
    try std.testing.expect(!ok);
}
