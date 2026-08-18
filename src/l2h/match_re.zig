//! PCRE2 match for relational ~ / !~.
//!
//! pcre2.h width-suffixed macros do not translate via translate-c; call the _8
//! entry points directly.

const std = @import("std");
const re = @import("re");

pub const Error = error{BadRegex};

// Backstop against catastrophic backtracking / stack exhaustion on adversarial
// patterns (ReDoS). Mirrors pcre2grep's defaults so legitimate matches are not
// truncated; a pathological pattern returns no-match instead of hanging.
const PCRE2_MATCH_LIMIT: u32 = 1_000_000;
const PCRE2_DEPTH_LIMIT: u32 = 1000;

/// Compile `pattern` and match against `subject`. Invalid patterns raise `BadRegex`.
/// Empty matches succeed (PCRE2 default). Match-limit / depth-limit trips return
/// `false` (no hang), not an error.
pub fn matchRe(pattern: []const u8, subject: []const u8) Error!bool {
    var errnumber: c_int = 0;
    var erroffset: usize = 0;
    const compiled = re.pcre2_compile_8(pattern.ptr, pattern.len, 0, &errnumber, &erroffset, null) orelse
        return error.BadRegex;
    defer _ = re.pcre2_code_free_8(compiled);

    const match_data = re.pcre2_match_data_create_from_pattern_8(compiled, null) orelse return false;
    defer _ = re.pcre2_match_data_free_8(match_data);

    // Cap backtracking so a hostile pattern/subject cannot exhaust CPU or stack.
    const mctx = re.pcre2_match_context_create_8(null) orelse return false;
    defer _ = re.pcre2_match_context_free_8(mctx);
    _ = re.pcre2_set_match_limit_8(mctx, PCRE2_MATCH_LIMIT);
    _ = re.pcre2_set_depth_limit_8(mctx, PCRE2_DEPTH_LIMIT);

    // Default PCRE2 flags: empty matches succeed (`^$`, `''`, `a*` on "").
    // NOTBOL/NOTEOL stay unset so `^`/`$` anchor at subject start/end.
    const flags: u32 = 0;

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
    const ok = try matchRe(pattern, subject);

    // Assert
    try std.testing.expect(ok);
}

test "MatchFailure" {
    // Arrange
    const pattern = "[0-9]+";
    const subject = "num";

    // Act
    const ok = try matchRe(pattern, subject);

    // Assert
    try std.testing.expect(!ok);
}

test "BadPatternIsError" {
    // Arrange / Act / Assert — unclosed character class
    try std.testing.expectError(error.BadRegex, matchRe("[0-9", "1"));
}

test "CatastrophicBacktrackingIsCapped" {
    // A classic exponential pattern (a+)+ against a non-matching tail would,
    // without a match limit, backtrack ~2^N times. The match/depth cap must
    // bound it to a prompt false instead of hanging.
    const pattern = "(a+)+$";
    const subject = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"; // trailing 'b' defeats $

    const ok = try matchRe(pattern, subject);

    // No hang (this test reaches the assertion) and no match.
    try std.testing.expect(!ok);
}

test "anchored start matches at subject start" {
    // Arrange
    const pattern = "^abc";
    const subject = "abcdef";

    // Act
    const ok = try matchRe(pattern, subject);

    // Assert
    try std.testing.expect(ok);
}

test "anchored end matches at subject end" {
    // Arrange
    const pattern = "def$";
    const subject = "abcdef";

    // Act
    const ok = try matchRe(pattern, subject);

    // Assert
    try std.testing.expect(ok);
}

test "anchored pattern rejects non-anchored position" {
    // Arrange — `^def` must NOT match because `def` is mid-subject.
    const pattern = "^def";
    const subject = "abcdef";

    // Act
    const ok = try matchRe(pattern, subject);

    // Assert
    try std.testing.expect(!ok);
}

test "empty-string anchors match empty subject" {
    // Arrange — `^$` is a zero-length match; PCRE2_NOTEMPTY used to reject it.
    const pattern = "^$";
    const subject = "";

    // Act
    const ok = try matchRe(pattern, subject);

    // Assert
    try std.testing.expect(ok);
}

test "empty pattern matches at start of any subject" {
    // Arrange — empty regex matches the empty string at offset 0.

    // Act
    const empty_on_empty = try matchRe("", "");
    const empty_on_text = try matchRe("", "abc");

    // Assert
    try std.testing.expect(empty_on_empty);
    try std.testing.expect(empty_on_text);
}

test "star quantifier matches when the repeat is zero" {
    // Arrange — `x*` matches "" at the start of "abc"; `a*` matches "".

    // Act
    const star_on_abc = try matchRe("x*", "abc");
    const star_on_empty = try matchRe("a*", "");

    // Assert
    try std.testing.expect(star_on_abc);
    try std.testing.expect(star_on_empty);
}

test "empty-string anchors reject non-empty subject" {
    // Arrange
    const ok = try matchRe("^$", "a");

    // Assert
    try std.testing.expect(!ok);
}
