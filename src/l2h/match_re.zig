const std = @import("std");
const re = @import("re");

// PCRE2 match for relational ~ / !~ (port of proc_match_re).
//
// pcre2.h width-suffixed macros do not translate via translate-c; call the _8
// entry points directly.

comptime {
    _ = re;
}

pub fn matchRe(pattern: []const u8, subject: []const u8) bool {
    var errnumber: c_int = 0;
    var erroffset: usize = 0;
    const compiled = re.pcre2_compile_8(pattern.ptr, pattern.len, 0, &errnumber, &erroffset, null) orelse return false;
    defer _ = re.pcre2_code_free_8(compiled);

    const match_data = re.pcre2_match_data_create_from_pattern_8(compiled, null) orelse return false;
    defer _ = re.pcre2_match_data_free_8(match_data);

    var flags: u32 = re.PCRE2_NOTEMPTY;
    if (std.mem.indexOfScalar(u8, subject, '^') == null) flags |= re.PCRE2_NOTBOL;
    if (std.mem.indexOfScalar(u8, subject, '$') == null) flags |= re.PCRE2_NOTEOL;

    const rc = re.pcre2_match_8(compiled, subject.ptr, subject.len, 0, flags, match_data, null);
    return rc >= 0;
}
