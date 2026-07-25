//! Command line interface for the `hc` executable.
//!
//! Port of src/hc/configuration.c: argument parsing, validation and dispatch
//! to the modes run functions (string/hash/file/dir). The argument layout
//! matches the C binary exactly: the first positional is the hash algorithm,
//! the second positional is the command, followed by options. Parsing is done
//! with yazap; all options are registered on the root command and then
//! validated against the per-mode allow-list (mirroring the four argtable3
//! tables in configuration.c).

const std = @import("std");
const builtin = @import("builtin");
const yazap = @import("yazap");
const lib = @import("lib");
const hashes = @import("hashes");
const modes = @import("modes");

const build_options = @import("build_options");

const App = yazap.App;
const Arg = yazap.Arg;
const ArgMatches = yazap.ArgMatches;

pub const PROGRAM_NAME = "hc";

// --- Option name constants (mirror OPT_* macros in configuration.c) --------

pub const opt_algorithm = "algorithm";
pub const opt_command = "command";
pub const opt_source = "source";
pub const opt_hash = "hash";
pub const opt_limit = "limit";
pub const opt_offset = "offset";
pub const opt_threads = "threads";
pub const opt_base64 = "base64";
pub const opt_sfv = "sfv";
pub const opt_noprobe = "noprobe";
pub const opt_noerroronfind = "noerroronfind";
pub const opt_save = "save";
pub const opt_lower = "lower";
pub const opt_time = "time";
pub const opt_checksumfile = "checksumfile";
pub const opt_recursively = "recursively";
pub const opt_performance = "performance";
pub const opt_dict = "dict";
pub const opt_min = "min";
pub const opt_max = "max";
pub const opt_include = "include";
pub const opt_exclude = "exclude";
pub const opt_search = "search";

const STRING_CMD = "string";
const HASH_CMD = "hash";
const FILE_CMD = "file";
const DIR_CMD = "dir";

pub const Mode = enum {
    none,
    string,
    hash,
    file,
    dir,
};

/// Active mode, mirrored from C's `hc_mode_t g_mode`. Published before
/// dispatch so the SIGINT handler in main.zig can decide whether to print
/// brute-force timings.
pub var active_mode: Mode = .none;

/// Every option registered on the root command (used for the foreign-option
/// allow-list validation).
const all_options = [_][]const u8{
    opt_source,        opt_hash,      opt_limit,          opt_offset,
    opt_threads,       opt_base64,    opt_sfv,            opt_noprobe,
    opt_noerroronfind, opt_save,      opt_lower,          opt_time,
    opt_checksumfile,  opt_recursively, opt_performance,  opt_dict,
    opt_min,           opt_max,       opt_include,        opt_exclude,
    opt_search,
};

fn allowedOptions(mode: Mode) []const []const u8 {
    return switch (mode) {
        .string => &[_][]const u8{ opt_source, opt_base64, opt_lower },
        .hash => &[_][]const u8{
            opt_source, opt_base64, opt_dict, opt_min, opt_max,
            opt_performance, opt_noprobe, opt_threads, opt_lower,
        },
        .file => &[_][]const u8{
            opt_source, opt_hash, opt_limit, opt_offset, opt_checksumfile,
            opt_save, opt_time, opt_sfv, opt_lower, opt_base64,
        },
        .dir => &[_][]const u8{
            opt_source, opt_hash, opt_exclude, opt_include, opt_limit,
            opt_offset, opt_search, opt_recursively, opt_checksumfile,
            opt_save, opt_time, opt_sfv, opt_lower, opt_base64,
            opt_noerroronfind,
        },
        .none => &[_][]const u8{},
    };
}

fn isAllowed(mode: Mode, name: []const u8) bool {
    for (allowedOptions(mode)) |n| {
        if (std.mem.eql(u8, n, name)) return true;
    }
    return false;
}

pub fn detectMode(cmd: []const u8) Mode {
    if (std.mem.eql(u8, cmd, STRING_CMD)) return .string;
    if (std.mem.eql(u8, cmd, HASH_CMD)) return .hash;
    if (std.mem.eql(u8, cmd, FILE_CMD)) return .file;
    if (std.mem.eql(u8, cmd, DIR_CMD)) return .dir;
    return .none;
}

// --- Number parsing (mirrors prconf_read_offset_parameter / sscanf %lli) ---

pub const NumberError = error{InvalidCharacter};

/// Parses a signed 64-bit number mirroring `sscanf("%lli", ...)`: out of range
/// values clamp to the signed extremum based on their sign rather than erroring.
pub fn parseBigNumber(s: []const u8) NumberError!i64 {
    const trimmed = std.mem.trim(u8, s, &std.ascii.whitespace);
    if (trimmed.len == 0) return error.InvalidCharacter;
    return std.fmt.parseInt(i64, trimmed, 10) catch |err| switch (err) {
        error.Overflow => blk: {
            const negative = trimmed[0] == '-';
            break :blk if (negative) std.math.minInt(i64) else std.math.maxInt(i64);
        },
        error.InvalidCharacter => return error.InvalidCharacter,
    };
}

/// Returns the architecture suffix used in the copyright banner. The C binary
/// hardcodes "x64"; we keep that on x86_64 and extend sensibly elsewhere.
pub fn archSuffix() []const u8 {
    return switch (builtin.cpu.arch) {
        .x86_64 => "x64",
        .aarch64 => "arm64",
        .x86 => "x86",
        else => "native",
    };
}

pub fn productVersion() []const u8 {
    return build_options.version;
}

pub fn appName() []const u8 {
    return "Hash Calculator";
}

/// Prints the copyright banner exactly like hc_print_copyright():
/// "\n<APP_NAME> <version> <arch>\nCopyright ...\n\n"
pub fn printCopyright(out: *std.Io.Writer) !void {
    try out.print(
        "\n{s} {s} {s}\nCopyright (C) 2009-2026 Alexander Egorov. All rights reserved.\n\n",
        .{ appName(), productVersion(), archSuffix() },
    );
}

/// Builds a comma separated list of all supported hash names for the help text
/// (replaces hsh_print_hashes in the C version).
pub fn buildHashList(allocator: std.mem.Allocator) ![]u8 {
    var total: usize = 0;
    for (hashes.hashes, 0..) |h, i| {
        if (i != 0) total += 2; // ", "
        total += h.name.len;
    }

    const buf = try allocator.alloc(u8, total);
    var pos: usize = 0;
    for (hashes.hashes, 0..) |h, i| {
        if (i != 0) {
            buf[pos] = ',';
            buf[pos + 1] = ' ';
            pos += 2;
        }
        @memcpy(buf[pos .. pos + h.name.len], h.name);
        pos += h.name.len;
    }
    return buf;
}

// --- Threads resolution (mirrors prconf_get_threads_count) ----------------

pub fn resolveThreads(out: *std.Io.Writer, provided: ?[]const u8) i32 {
    const processors: u32 = lib.getProcessorCount();
    const processors_i32: i32 = @intCast(processors);

    var num: i32 = if (provided) |v|
        std.fmt.parseInt(i32, v, 10) catch 0
    else blk: {
        break :blk if (processors == 1) 1 else @intCast(@min(processors, processors / 2));
    };

    if (num < 1 or num > processors_i32) {
        const def: i32 = if (processors == 1) 1 else @intCast(processors / 2);
        out.print(
            "Threads number must be between 1 and {d} but it was set to {d}. Reset to default {d}\n",
            .{ processors, @as(u32, @bitCast(num)), @as(u32, @bitCast(def)) },
        ) catch {};
        num = def;
    }
    return num;
}

/// Reads and validates a limit/offset parameter. Prints the appropriate error
/// (no copyright for non-numeric, copyright + message for negative) and returns
/// false if the caller must abort.
fn readNumberParam(
    out: *std.Io.Writer,
    provided: ?[]const u8,
    option_name: []const u8,
) !bool {
    if (provided) |v| {
        const n = parseBigNumber(v) catch {
            try out.print("Invalid parameter --{s} {s}. Must be number\n", .{ option_name, v });
            return false;
        };
        if (n < 0) {
            try printCopyright(out);
            try out.print("Invalid {s} option must be positive but was {d}\n", .{ option_name, n });
            return false;
        }
    }
    return true;
}

// --- App construction ------------------------------------------------------

fn createApp(allocator: std.mem.Allocator) !*App {
    const app = try allocator.create(App);
    errdefer allocator.destroy(app);

    const descr = try std.fmt.allocPrint(
        allocator,
        "{s} {s} {s}\nCopyright (C) 2009-2026 Alexander Egorov. All rights reserved.",
        .{ appName(), productVersion(), archSuffix() },
    );
    app.* = App.init(allocator, PROGRAM_NAME, descr);

    var root = app.rootCommand();
    root.setProperty(.help_on_empty_args);

    // Positional arguments: <algorithm> then <command>. The full hash list is
    // intentionally NOT inlined here: yazap's HelpMessageWriter uses a fixed
    // 4096-byte buffer and the algorithm names overflow it.
    var algorithm_arg = Arg.positional(opt_algorithm, "hash algorithm. See all possible values below", null);
    algorithm_arg.setValuePlaceholder("<algorithm>");
    try root.addArg(algorithm_arg);

    try root.addArg(Arg.positional(opt_command, "must be string, hash, file or dir", null));

    // Options (registered on root, validated per-mode after parsing). Each
    // value option allows an empty value so `-s ""` (normalized to `-s=`) is
    // accepted, matching argtable3 semantics.
    var a: Arg = undefined;

    a = Arg.singleValueOption(opt_source, 's', "string to calculate hash sum for");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_hash, 'm', "hash to validate file");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_limit, 'z', "set the limit in bytes of the part of the file to calculate hash for. The whole file by default will be applied");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_offset, 'q', "set start position within file to calculate hash from. Zero by default");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_threads, 'T', "the number of threads to crack hash. The half of system processors by default");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_save, 'o', "save files' hashes into the file specified besides console output");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_dict, 'a', "initial string's dictionary. All digits, upper and lower case latin symbols by default");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_min, 'n', "set minimum length of the string to restore. 1 by default");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_max, 'x', "set maximum length of the string to restore. 10 by default");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_include, 'i', "include only files that match the pattern specified");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_exclude, 'e', "exclude files that match the pattern specified");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);
    a = Arg.singleValueOption(opt_search, 'H', "hash to search a file that matches it");
    a.setProperty(.allow_empty_value);
    try root.addArg(a);

    try root.addArg(Arg.booleanOption(opt_base64, 'b', "output hash as Base64"));
    try root.addArg(Arg.booleanOption(opt_lower, 'l', "output hash using low case (false by default)"));
    try root.addArg(Arg.booleanOption(opt_time, 't', "show calculation time (false by default)"));
    try root.addArg(Arg.booleanOption(opt_checksumfile, 'c', "output hash in file checksum format"));
    try root.addArg(Arg.booleanOption(opt_recursively, 'r', "scan directory recursively"));
    try root.addArg(Arg.booleanOption(opt_performance, 'p', "test performance by cracking 12345 string hash"));
    try root.addArg(Arg.booleanOption(opt_noprobe, null, "Disable hash crack time probing"));
    try root.addArg(Arg.booleanOption(opt_noerroronfind, null, "Disable error output while search files. False by default"));
    try root.addArg(Arg.booleanOption(opt_sfv, null, "output hash in the SFV (Simple File Verification) format. Only for CRC32 or CRC32C."));

    return app;
}

/// Short names of options that take a value.
const value_option_short = [_]u8{ 's', 'm', 'z', 'q', 'T', 'o', 'a', 'n', 'x', 'i', 'e', 'H' };

/// Long names of options that take a value.
const value_option_long = [_][]const u8{
    "source", "hash", "limit", "offset", "threads", "save",
    "dict", "min", "max", "include", "exclude", "search",
};

/// Short names of options that take a numeric value (limit/offset/min/max/threads).
const numeric_option_short = [_]u8{ 'z', 'q', 'n', 'x', 'T' };

/// Long names of options that take a numeric value.
const numeric_option_long = [_][]const u8{ "limit", "offset", "min", "max", "threads" };

/// True when `tok` is a bare value-expecting option token with no attached
/// value (e.g. `-s` or `--source` but not `-s=x` or `-sx`).
pub fn isBareValueOption(tok: []const u8) bool {
    if (tok.len == 2 and tok[0] == '-') {
        for (value_option_short) |c| if (tok[1] == c) return true;
        return false;
    }
    if (std.mem.startsWith(u8, tok, "--") and std.mem.indexOfScalar(u8, tok, '=') == null) {
        const name = tok[2..];
        for (value_option_long) |n| if (std.mem.eql(u8, name, n)) return true;
        return false;
    }
    return false;
}

/// True when `tok` is a bare numeric-value option (limit/offset/min/max/threads).
pub fn isNumericValueOption(tok: []const u8) bool {
    if (tok.len == 2 and tok[0] == '-') {
        for (numeric_option_short) |c| if (tok[1] == c) return true;
        return false;
    }
    if (std.mem.startsWith(u8, tok, "--") and std.mem.indexOfScalar(u8, tok, '=') == null) {
        const name = tok[2..];
        for (numeric_option_long) |n| if (std.mem.eql(u8, name, n)) return true;
        return false;
    }
    return false;
}

/// True when `tok` is a negative decimal integer (e.g. "-10"). Tokens that
/// look like options ("--limit", "-l") are rejected.
pub fn isNegativeNumber(tok: []const u8) bool {
    if (tok.len < 2 or tok[0] != '-') return false;
    for (tok[1..]) |c| if (c < '0' or c > '9') return false;
    return true;
}

/// yazap's tokenizer skips empty argv elements and treats `-10` as a short
/// option group, so a value passed as a separate token is lost in two cases:
///   * `-s ""`   -> empty value (argtable3 accepts it; C# EmptyStringHash)
///   * `-z -10`  -> negative numeric value (C# InvalidNumbericOptionsNegativeTest)
/// This rewrites such a bare value-option followed by the problematic token
/// into the attached form (`-s=` / `-z=-10`), which yazap captures. Mirrors
/// argtable3, which consumes the next token as a string regardless of `-`.
fn shouldAttach(opt_tok: []const u8, next_tok: []const u8) bool {
    if (!isBareValueOption(opt_tok)) return false;
    if (next_tok.len == 0) return true;
    if (isNumericValueOption(opt_tok) and isNegativeNumber(next_tok)) return true;
    return false;
}

fn attachValue(
    allocator: std.mem.Allocator,
    opt_tok: []const u8,
    val_tok: []const u8,
) ![:0]const u8 {
    const buf = try allocator.allocSentinel(u8, opt_tok.len + 1 + val_tok.len, 0);
    @memcpy(buf[0..opt_tok.len], opt_tok);
    buf[opt_tok.len] = '=';
    @memcpy(buf[opt_tok.len + 1 ..][0..val_tok.len], val_tok);
    return buf;
}

/// Rewrites argv so empty and negative-numeric option values are attached to
/// their option (see `shouldAttach`). Returns the original slice unchanged when
/// no rewrite is needed.
pub fn normalizeArgv(
    allocator: std.mem.Allocator,
    argv: []const [:0]const u8,
) ![]const [:0]const u8 {
    var merged: usize = 0;
    {
        var i: usize = 0;
        while (i < argv.len) {
            if (i + 1 < argv.len and shouldAttach(argv[i], argv[i + 1])) {
                merged += 1;
                i += 2;
            } else i += 1;
        }
    }
    if (merged == 0) return argv;

    const out = try allocator.alloc([:0]const u8, argv.len - merged);
    errdefer allocator.free(out);

    var oi: usize = 0;
    var i: usize = 0;
    while (i < argv.len) {
        if (i + 1 < argv.len and shouldAttach(argv[i], argv[i + 1])) {
            out[oi] = try attachValue(allocator, argv[i], argv[i + 1]);
            oi += 1;
            i += 2;
        } else {
            out[oi] = argv[i];
            oi += 1;
            i += 1;
        }
    }
    return out;
}

// --- Dispatch helpers ------------------------------------------------------

fn foreignOptionPresent(matches: ArgMatches, mode: Mode) ?[]const u8 {
    for (all_options) |name| {
        if (!isAllowed(mode, name) and matches.containsArg(name)) return name;
    }
    return null;
}

fn runString(
    matches: ArgMatches,
    bctx: *const modes.BuiltinCtx,
    env: modes.RunEnv,
) !void {
    const source = matches.getSingleValue(opt_source) orelse {
        try env.out.print("--{s} option is required\n", .{opt_source});
        return;
    };
    var sctx: modes.StringCtx = .{
        .builtin = bctx,
        .string = source,
        .is_base64 = matches.containsArg(opt_base64),
    };
    try modes.builtinRun(modes.StringCtx, bctx, &sctx, modes.strRun, env);
}

fn runHash(
    matches: ArgMatches,
    bctx: *const modes.BuiltinCtx,
    env: modes.RunEnv,
    app: *App,
    io: std.Io,
) !void {
    const performance = matches.containsArg(opt_performance);
    const source = matches.getSingleValue(opt_source);
    if (!performance and (source == null or source.?.len == 0)) {
        try env.out.print(
            "--{s} option is required to restore hash. Use -p to run performance test without it\n",
            .{opt_source},
        );
        try app.displayHelp(io);
        return;
    }

    const threads = resolveThreads(env.out, matches.getSingleValue(opt_threads));
    // resolveThreads writes to the Zig-side stdout buffer, but bf_crack_hash
    // (C) prints via lib_printf which fflush-es libc stdout immediately. Flush
    // our buffer first so the thread-count warning precedes the crack output
    // (mirrors C prconf_get_threads_count running during config parsing).
    env.out.flush() catch {};

    var hctx: modes.HashCtx = .{
        .builtin = bctx,
        .hash = source,
        .is_base64 = matches.containsArg(opt_base64),
        .no_probe = matches.containsArg(opt_noprobe),
        .performance = performance,
        .threads = threads,
    };
    if (matches.getSingleValue(opt_dict)) |d| hctx.dictionary = d;
    if (matches.getSingleValue(opt_min)) |m| hctx.min = std.fmt.parseInt(i32, m, 10) catch 0;
    if (matches.getSingleValue(opt_max)) |m| hctx.max = std.fmt.parseInt(i32, m, 10) catch 0;

    try modes.builtinRun(modes.HashCtx, bctx, &hctx, modes.hashRun, env);
}

fn runFile(
    matches: ArgMatches,
    bctx: *const modes.BuiltinCtx,
    env: modes.RunEnv,
) !void {
    if (!try readNumberParam(env.out, matches.getSingleValue(opt_limit), opt_limit)) return;
    if (!try readNumberParam(env.out, matches.getSingleValue(opt_offset), opt_offset)) return;

    const file_path = matches.getSingleValue(opt_source) orelse {
        try env.out.print("--{s} option is required\n", .{opt_source});
        return;
    };

    const limit_value: i64 = if (matches.getSingleValue(opt_limit)) |v| (parseBigNumber(v) catch 0) else std.math.maxInt(i64);
    const offset_value: i64 = if (matches.getSingleValue(opt_offset)) |v| (parseBigNumber(v) catch 0) else 0;

    var fctx: modes.FileCtx = .{
        .builtin = bctx,
        .file_path = file_path,
        .limit = limit_value,
        .offset = offset_value,
        .show_time = matches.containsArg(opt_time),
        .is_verify = matches.containsArg(opt_checksumfile),
        .result_in_sfv = matches.containsArg(opt_sfv),
        .is_base64 = matches.containsArg(opt_base64),
    };
    if (matches.getSingleValue(opt_hash)) |h| fctx.hash = h;
    if (matches.getSingleValue(opt_save)) |s| fctx.save_result_path = s;

    try modes.builtinRun(modes.FileCtx, bctx, &fctx, modes.fileRun, env);
}

fn runDir(
    matches: ArgMatches,
    bctx: *const modes.BuiltinCtx,
    env: modes.RunEnv,
) !void {
    if (!try readNumberParam(env.out, matches.getSingleValue(opt_limit), opt_limit)) return;
    if (!try readNumberParam(env.out, matches.getSingleValue(opt_offset), opt_offset)) return;

    const dir_path = matches.getSingleValue(opt_source) orelse {
        try env.out.print("--{s} option is required\n", .{opt_source});
        return;
    };

    const limit_value: i64 = if (matches.getSingleValue(opt_limit)) |v| (parseBigNumber(v) catch 0) else std.math.maxInt(i64);
    const offset_value: i64 = if (matches.getSingleValue(opt_offset)) |v| (parseBigNumber(v) catch 0) else 0;

    var dctx: modes.DirCtx = .{
        .builtin = bctx,
        .dir_path = dir_path,
        .limit = limit_value,
        .offset = offset_value,
        .show_time = matches.containsArg(opt_time),
        .is_verify = matches.containsArg(opt_checksumfile),
        .result_in_sfv = matches.containsArg(opt_sfv),
        .recursively = matches.containsArg(opt_recursively),
        .no_error_on_find = matches.containsArg(opt_noerroronfind),
        .is_base64 = matches.containsArg(opt_base64),
    };
    if (matches.getSingleValue(opt_hash)) |h| dctx.hash = h;
    if (matches.getSingleValue(opt_search)) |s| dctx.search_hash = s;
    if (matches.getSingleValue(opt_include)) |i| dctx.include_pattern = i;
    if (matches.getSingleValue(opt_exclude)) |e| dctx.exclude_pattern = e;
    if (matches.getSingleValue(opt_save)) |s| dctx.save_result_path = s;

    try modes.builtinRun(modes.DirCtx, bctx, &dctx, modes.dirRun, env);
}

// --- Entry point -----------------------------------------------------------

pub const Outcome = enum { ok, invalid_command, invalid_options };

/// Parses argv and dispatches to the matching mode. Mirrors conf_run_app().
/// Returns the outcome so the caller can map it to a process exit code.
pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    argv: []const [:0]const u8,
) !Outcome {
    const app = try createApp(allocator);
    defer {
        app.deinit();
        allocator.destroy(app);
    }

    // Rewrite `-s ""` (empty value) and `-z -10` (negative numeric value)
    // into attached form so yazap captures them (it otherwise skips empty
    // argv elements and treats `-10` as a short option group).
    const argv_norm = try normalizeArgv(allocator, argv);

    const matches = try app.parseFrom(io, argv_norm);

    // No arguments at all -> full syntax (help_on_empty_args also covers this).
    if (!matches.containsArgs()) {
        try app.displayHelp(io);
        return .ok;
    }

    const cmd = matches.getSingleValue(opt_command);
    const mode = if (cmd) |c| detectMode(c) else .none;
    if (mode == .none) {
        try out.print(
            "Invalid command one of: {s}, {s}, {s} or {s} expected",
            .{ STRING_CMD, HASH_CMD, FILE_CMD, DIR_CMD },
        );
        return .invalid_command;
    }

    const algorithm = matches.getSingleValue(opt_algorithm) orelse {
        try app.displayHelp(io);
        return .invalid_options;
    };

    // Reject options that do not belong to the selected command (each C
    // argtable only declares its own options; a foreign option is a syntax
    // error for that command).
    if (foreignOptionPresent(matches, mode)) |_| {
        try app.displayHelp(io);
        return .invalid_options;
    }

    const bctx: modes.BuiltinCtx = .{
        .hash_algorithm = algorithm,
        .is_print_low_case = matches.containsArg(opt_lower),
    };

    const env: modes.RunEnv = .{
        .io = io,
        .allocator = allocator,
        .out = out,
    };

    active_mode = mode;

    switch (mode) {
        .string => try runString(matches, &bctx, env),
        .hash => try runHash(matches, &bctx, env, app, io),
        .file => try runFile(matches, &bctx, env),
        .dir => try runDir(matches, &bctx, env),
        .none => unreachable,
    }

    return .ok;
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

test "detectMode maps commands" {
    try std.testing.expectEqual(Mode.string, detectMode("string"));
    try std.testing.expectEqual(Mode.hash, detectMode("hash"));
    try std.testing.expectEqual(Mode.file, detectMode("file"));
    try std.testing.expectEqual(Mode.dir, detectMode("dir"));
    try std.testing.expectEqual(Mode.none, detectMode("bogus"));
}

test "parseBigNumber valid values" {
    try std.testing.expectEqual(@as(i64, 0), try parseBigNumber("0"));
    try std.testing.expectEqual(@as(i64, 42), try parseBigNumber("42"));
    try std.testing.expectEqual(@as(i64, -10), try parseBigNumber("-10"));
    try std.testing.expectEqual(@as(i64, 1024), try parseBigNumber(" 1024 "));
}

test "parseBigNumber clamps overflow to signed extremum" {
    try std.testing.expectEqual(std.math.maxInt(i64), try parseBigNumber("18446744073709551615"));
    try std.testing.expectEqual(std.math.minInt(i64), try parseBigNumber("-10223372036854775808"));
}

test "parseBigNumber rejects non-numeric" {
    try std.testing.expectError(error.InvalidCharacter, parseBigNumber("a"));
    try std.testing.expectError(error.InvalidCharacter, parseBigNumber(""));
}

test "isAllowed per-mode option sets" {
    try std.testing.expect(isAllowed(.string, "source"));
    try std.testing.expect(isAllowed(.string, "base64"));
    try std.testing.expect(!isAllowed(.string, "noprobe"));
    try std.testing.expect(isAllowed(.hash, "noprobe"));
    try std.testing.expect(isAllowed(.file, "limit"));
    try std.testing.expect(isAllowed(.dir, "search"));
    try std.testing.expect(!isAllowed(.hash, "search"));
}

test "isNegativeNumber distinguishes values from options" {
    try std.testing.expect(isNegativeNumber("-10"));
    try std.testing.expect(isNegativeNumber("-10223372036854775808"));
    try std.testing.expect(!isNegativeNumber("10"));
    try std.testing.expect(!isNegativeNumber("--limit"));
    try std.testing.expect(!isNegativeNumber("-l"));
    try std.testing.expect(!isNegativeNumber("-"));
}

test "normalizeArgv attaches empty and negative values" {
    const allocator = std.testing.allocator;
    {
        const argv = [_][:0]const u8{ "-s", "" };
        const out = try normalizeArgv(allocator, &argv);
        defer if (out.ptr != argv.ptr) allocator.free(out);
        try std.testing.expectEqual(@as(usize, 1), out.len);
        try std.testing.expectEqualStrings("-s=", out[0]);
    }
    {
        const argv = [_][:0]const u8{ "-z", "-10" };
        const out = try normalizeArgv(allocator, &argv);
        defer if (out.ptr != argv.ptr) allocator.free(out);
        try std.testing.expectEqual(@as(usize, 1), out.len);
        try std.testing.expectEqualStrings("-z=-10", out[0]);
    }
    {
        // Positive numbers and normal tokens are untouched.
        const argv = [_][:0]const u8{ "-z", "10", "-s", "abc" };
        const out = try normalizeArgv(allocator, &argv);
        try std.testing.expect(out.ptr == argv.ptr);
    }
}

test "buildHashList includes known algorithms" {
    const list = try buildHashList(std.testing.allocator);
    defer std.testing.allocator.free(list);
    try std.testing.expect(std.mem.indexOf(u8, list, "tiger") != null);
    try std.testing.expect(std.mem.indexOf(u8, list, "md5") != null);
    try std.testing.expect(std.mem.indexOf(u8, list, ", ") != null);
}

test "string mode dispatch produces hash output" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var writer = std.Io.Writer.Allocating.init(arena.allocator());
    const out = &writer.writer;

    const argv = [_][:0]const u8{ "tiger", "string", "-s", "abc" };
    const outcome = try run(arena.allocator(), std.testing.io, out, &argv);

    try std.testing.expectEqual(Outcome.ok, outcome);
    const got = writer.written();
    // tiger of "abc" -> known digest, uppercase hex + trailing newline.
    try std.testing.expect(std.mem.indexOf(u8, got, "\n") != null);
    try std.testing.expect(got.len >= 24 * 2);
}

test "unknown command reports invalid command" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var writer = std.Io.Writer.Allocating.init(arena.allocator());
    const out = &writer.writer;

    const argv = [_][:0]const u8{ "tiger", "bogus", "-s", "abc" };
    const outcome = try run(arena.allocator(), std.testing.io, out, &argv);

    try std.testing.expectEqual(Outcome.invalid_command, outcome);
    const got = writer.written();
    try std.testing.expect(std.mem.indexOf(u8, got, "Invalid command") != null);
}
