//! Regex match used by relational ~ / !~ (match_re.zig).

const std = @import("std");
const match_re = @import("match_re.zig");

test "ProcessorTest MatchSuccess" {
    // Arrange
    const pattern = "[0-9]+";
    const subject = "123";

    // Act
    const ok = match_re.matchRe(pattern, subject);

    // Assert
    try std.testing.expect(ok);
}

test "ProcessorTest MatchFailure" {
    // Arrange
    const pattern = "[0-9]+";
    const subject = "num";

    // Act
    const ok = match_re.matchRe(pattern, subject);

    // Assert
    try std.testing.expect(!ok);
}
