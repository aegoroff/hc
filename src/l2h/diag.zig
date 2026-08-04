const std = @import("std");
const ErrorReporter = @import("fehler").ErrorReporter;
const Diagnostic = @import("fehler").Diagnostic;
const SourceRange = @import("fehler").SourceRange;
const state = @import("state.zig");
const expr = @import("expr.zig");

/// Pending span for the next `report` (set while compiling/evaluating).
var pending_span: ?expr.Span = null;

/// Optional filesystem path attached to the next `IoFailure` message.
var pending_io_path: [512]u8 = undefined;
var pending_io_path_len: usize = 0;

/// Scratch buffer for `messageForRuntime` when it embeds a path.
var runtime_msg_buf: [768]u8 = undefined;

/// Opt-in sink for tests that exercise the C `reportParse` path (void return).
/// Production leaves this null; `compile_test` uses `report`'s returned `Reported`.
var test_msg_buf: ?*[768]u8 = null;
var test_msg_len: ?*usize = null;

pub const IO_FAILURE_MSG = "I/O failure (missing path or unreadable file/directory)";

/// Plain message + span just emitted to fehler (for callers that need to
/// assert without scraping stderr).
pub const Reported = struct {
    message: []const u8,
    span: expr.Span,
};

/// Clear pending span / I/O path before a new compilation unit.
pub fn clearLast() void {
    pending_span = null;
    pending_io_path_len = 0;
}

/// Install a buffer that receives the plain message from the next `report` /
/// `reportParse` (for frontend parse tests). Pass null to uninstall.
pub fn setTestMessageSink(buf: ?*[768]u8, len: ?*usize) void {
    test_msg_buf = buf;
    test_msg_len = len;
    if (len) |l| l.* = 0;
}

/// Remember a path to include in the next `IoFailure` diagnostic.
pub fn noteIoPath(path: []const u8) void {
    const n = @min(path.len, pending_io_path.len);
    @memcpy(pending_io_path[0..n], path[0..n]);
    pending_io_path_len = n;
}

pub fn noteSpan(sp: expr.Span) void {
    if (sp.isSet()) pending_span = sp;
}

pub fn noteNode(node: anytype) void {
    noteSpan(expr.Span.fromNode(node));
}

fn fileName() []const u8 {
    return state.source_name;
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
    const span: expr.Span = .{
        .first_line = fl,
        .first_column = fc,
        .last_line = if (last_line > 0) last_line else fl,
        .last_column = if (last_column > 0) last_column else fc,
    };

    if (test_msg_buf) |buf| {
        const n = @min(message.len, buf.len);
        @memcpy(buf[0..n], message[0..n]);
        if (test_msg_len) |len| len.* = n;
    }

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
    return .{ .message = message, .span = span };
}

/// Parser / semantic grammar errors (called from C via `fend_print_error`).
pub fn reportParse(
    first_line: c_int,
    first_column: c_int,
    last_line: c_int,
    last_column: c_int,
    message: []const u8,
) void {
    _ = reportWithRange(message, first_line, first_column, last_line, last_column);
}

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

pub fn messageForCompile(err: anyerror) []const u8 {
    if (sharedMessage(err)) |m| return m;
    return switch (err) {
        error.InvalidFromSourceType => "source expression type does not match the declared range kind",
        error.TypeMismatch => "type mismatch in expression or clause",
        else => @errorName(err),
    };
}

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
        else => @errorName(err),
    };
}
