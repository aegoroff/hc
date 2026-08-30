const std = @import("std");
const ErrorReporter = @import("fehler").ErrorReporter;
const Diagnostic = @import("fehler").Diagnostic;
const SourceRange = @import("fehler").SourceRange;
const state = @import("state.zig");
const expr = @import("expr.zig");
const modes = @import("modes");

/// Pending span for the next `report` (set while compiling/evaluating).
var pending_span: ?expr.Span = null;

/// Optional filesystem path attached to the next `IoFailure` message.
var pending_io_path: [512]u8 = undefined;
var pending_io_path_len: usize = 0;

/// Scratch buffer for `messageForRuntime` when it embeds a path.
var runtime_msg_buf: [768]u8 = undefined;

pub const IO_FAILURE_MSG = "I/O failure (missing path or unreadable file/directory)";

/// Plain message + span just emitted to fehler (for callers that need to
/// assert without scraping stderr).
pub const Reported = struct {
    message: []const u8,
    span: expr.Span,
};

/// Opt-in hook for tests that cannot use `report`'s return (`reportParse` path).
/// Production leaves this null.
pub const OnReported = *const fn (Reported) void;
var on_reported: ?OnReported = null;

/// Clear pending span / I/O path before a new compilation unit.
pub fn clearLast() void {
    pending_span = null;
    pending_io_path_len = 0;
}

/// Install or clear the test hook invoked on every `report` / `reportParse`.
pub fn setOnReported(cb: ?OnReported) void {
    on_reported = cb;
}

/// Remember a path to include in the next `IoFailure` diagnostic.
pub fn noteIoPath(path: []const u8) void {
    const n = @min(path.len, pending_io_path.len);
    @memcpy(pending_io_path[0..n], path[0..n]);
    pending_io_path_len = n;
}

/// Remember a source span for the next `report`.
pub fn noteSpan(sp: expr.Span) void {
    if (sp.isSet()) pending_span = sp;
}

/// Remember a parser node's location for the next `report`.
pub fn noteNode(node: anytype) void {
    noteSpan(expr.Span.fromNode(node));
}

fn fileName() []const u8 {
    return state.source_name;
}

/// Convert a Bison/Flex half-open column range `[first, last)` to inclusive
/// `[first, last]` for fehler. No-op when the range is already a single column.
fn inclusiveLastColumn(first_column: c_int, last_column: c_int) c_int {
    if (last_column > first_column) return last_column - 1;
    return last_column;
}

fn reportWithRange(
    message: []const u8,
    first_line: c_int,
    first_column: c_int,
    last_line: c_int,
    last_column: c_int,
) Reported {
    const fl: c_int = if (first_line > 0) first_line else 1;
    const fc: c_int = if (first_column > 0) first_column else 1;
    const ll: c_int = if (last_line > 0) last_line else fl;
    // Parser locs and wholeUnitRange use Bison half-open last_column; fehler
    // underlines inclusive end columns (`end - start + 1` tildes).
    const raw_lc: c_int = if (last_column > 0) last_column else fc;
    const lc: c_int = if (ll == fl)
        inclusiveLastColumn(fc, raw_lc)
    else if (raw_lc > 1)
        raw_lc - 1
    else
        raw_lc;
    const span: expr.Span = .{
        .first_line = fl,
        .first_column = fc,
        .last_line = ll,
        .last_column = lc,
    };
    const reported: Reported = .{ .message = message, .span = span };
    if (on_reported) |cb| cb(reported);

    var reporter = ErrorReporter.init(state.gpa);
    defer reporter.deinit();

    const source = state.source_text;
    if (source.len != 0) {
        reporter.addSource(fileName(), source) catch {};
    }

    const diagnostic = Diagnostic.init(.err, message)
        .withRange(SourceRange.span(
        fileName(),
        @intCast(span.first_line),
        @intCast(span.first_column),
        @intCast(span.last_line),
        @intCast(span.last_column),
    ));
    reporter.report(diagnostic);
    return reported;
}

/// Parser / semantic grammar errors (from `fend_print_error`).
pub fn reportParse(
    first_line: c_int,
    first_column: c_int,
    last_line: c_int,
    last_column: c_int,
    message: []const u8,
) void {
    _ = reportWithRange(message, first_line, first_column, last_line, last_column);
}

/// Half-open `[1, last_col)` covering the whole source unit (same convention as Bison).
fn wholeUnitRange() struct { c_int, c_int, c_int, c_int } {
    const lines: c_int = @intCast(std.mem.count(u8, state.source_text, "\n") + 1);
    const last_col: c_int = if (state.source_text.len == 0) 1 else blk: {
        if (std.mem.lastIndexOfScalar(u8, state.source_text, '\n')) |i| {
            break :blk @intCast(state.source_text.len - i);
        }
        break :blk @intCast(state.source_text.len + 1);
    };
    return .{ 1, 1, lines, last_col };
}

/// Report a failure; uses `pending_span` when set, else the whole unit.
pub fn report(message: []const u8) Reported {
    if (pending_span) |sp| {
        pending_span = null;
        return reportWithRange(message, sp.first_line, sp.first_column, sp.last_line, sp.last_column);
    }
    const r = wholeUnitRange();
    return reportWithRange(message, r[0], r[1], r[2], r[3]);
}

fn sharedMessage(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.InvalidProperty => "invalid property for this value type",
        error.DuplicateField => "duplicate record field name",
        error.InvalidRecordField => "cannot infer a record field name for this expression; use `name = expr`",
        error.UnknownMethod => "unknown method",
        error.InvalidMethodArity => "wrong number of method arguments",
        error.InvalidMethodReceiver => "invalid method receiver",
        error.InvalidMethodFields => "record fields do not match method requirements",
        error.UnsupportedNode => "unsupported syntax in this position",
        error.InvalidAst => "internal error: malformed AST",
        error.UndefinedName => "undefined name",
        error.QueryTooDeep => "query nesting too deep",
        error.InvalidStringEscape => "invalid string escape sequence",
        error.OutOfMemory => "out of memory",
        error.InvalidTreeDepth => "tree depth must be non-negative",
        else => null,
    };
}

/// Human-readable message for a compile/analysis error.
pub fn messageForCompile(err: anyerror) []const u8 {
    if (sharedMessage(err)) |m| return m;
    return switch (err) {
        error.InvalidFromSourceType => "source expression type does not match the declared range kind",
        error.TypeMismatch => "type mismatch in expression or clause",
        else => @errorName(err),
    };
}

/// Human-readable message for a runtime/evaluation error.
pub fn messageForRuntime(err: anyerror) []const u8 {
    if (sharedMessage(err)) |m| return m;
    return switch (err) {
        error.TypeMismatch => "type mismatch",
        error.UnknownProperty => "unknown property",
        error.UnknownHash => "unknown hash algorithm",
        error.InvalidHashDigest => "invalid hash digest for the selected algorithm",
        error.IoFailure => blk: {
            if (pending_io_path_len == 0) break :blk IO_FAILURE_MSG;
            const path = pending_io_path[0..pending_io_path_len];
            pending_io_path_len = 0;
            break :blk std.fmt.bufPrint(&runtime_msg_buf, "{s}: {s}", .{ IO_FAILURE_MSG, path }) catch IO_FAILURE_MSG;
        },
        error.WriteFailed => "write failed",
        error.Overflow => "value out of integer range",
        error.InvalidWindow => "limit/offset must be non-negative",
        error.InvalidRestoreBound => "restore min/max must be at least 1",
        error.InvalidRestoreRange => "restore min length is greater than max",
        error.InvalidRestoreLength => "restore max length is too big",
        error.BadRegex => "invalid regular expression",
        error.InvalidStringPayload => "string payload is not valid UTF-8 for this algorithm",
        error.OffsetTooBig => blk: {
            if (pending_io_path_len == 0) break :blk modes.file.OFFSET_TOO_BIG;
            const path = pending_io_path[0..pending_io_path_len];
            pending_io_path_len = 0;
            break :blk std.fmt.bufPrint(&runtime_msg_buf, "{s}: {s}", .{ modes.file.OFFSET_TOO_BIG, path }) catch modes.file.OFFSET_TOO_BIG;
        },
        else => @errorName(err),
    };
}
