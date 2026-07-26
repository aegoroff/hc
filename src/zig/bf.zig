//! Brute-force hash cracker — thin Zig wrapper around src/srclib/bf.c.

const std = @import("std");
const builtin = @import("builtin");
const lib = @import("lib");
const hashes = @import("hashes");
const gpu = @import("gpu");

const c = @import("c");

pub const MAX_DEFAULT: u32 = 10;

pub const CrackResult = struct {
    password: ?[]u8,
    attempts: u64,
};

var apr_ready: bool = false;

fn ensureApr() void {
    if (apr_ready) return;
    if (c.apr_initialize() != c.APR_SUCCESS) return;
    apr_ready = true;
}

fn cDigest(
    digest: [*c]c.apr_byte_t,
    string: ?*const anyopaque,
    input_len: c.apr_size_t,
) callconv(.c) void {
    const ctx: *const hashes.HashDefinition = @ptrCast(@alignCast(c_digest_hash_def));
    ctx.digest(@ptrCast(digest), @ptrCast(string), input_len);
}

var c_digest_hash_def: ?*const hashes.HashDefinition = null;

/// Live attempts for SIGINT (reads bf.c `g_attempts`).
pub fn getAttempts() u64 {
    return c.bf_get_attempts();
}

pub fn outputTimings(writer: *std.Io.Writer, attempts: u64, time: lib.Time) !void {
    const speed: f64 = if (time.total_seconds > 0)
        @as(f64, @floatFromInt(attempts)) / time.total_seconds
    else
        0;

    var abuf: [64]u8 = undefined;
    var sbuf: [64]u8 = undefined;
    const attempts_s = formatCommify(&abuf, attempts);
    const speed_s = formatCommifyF(&sbuf, speed);

    try writer.writeAll("\n");
    try writer.print("Attempts: {s} Time {d:0>2}:{d:0>2}:{d:.3} Speed: {s} attempts/second\n", .{
        attempts_s,
        time.hours,
        time.minutes,
        time.seconds,
        speed_s,
    });
}

fn formatCommify(buf: []u8, value: u64) []const u8 {
    var tmp: [32]u8 = undefined;
    const raw = std.fmt.bufPrint(&tmp, "{d}", .{value}) catch return "";
    var out_i: usize = 0;
    const digits = raw.len;
    for (raw, 0..) |ch, i| {
        if (i != 0 and (digits - i) % 3 == 0) {
            if (out_i < buf.len) {
                buf[out_i] = ' ';
                out_i += 1;
            }
        }
        if (out_i < buf.len) {
            buf[out_i] = ch;
            out_i += 1;
        }
    }
    return buf[0..out_i];
}

fn formatCommifyF(buf: []u8, value: f64) []const u8 {
    if (!std.math.isFinite(value) or value < 0) return formatCommify(buf, 0);
    return formatCommify(buf, @intFromFloat(@round(value)));
}

/// Full crack path via C `bf_crack_hash` (probe, CPU/GPU, timings, result line).
pub fn crackHash(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    dict: []const u8,
    hash: []const u8,
    passmin: u32,
    passmax_in: u32,
    hash_def: *const hashes.HashDefinition,
    no_probe: bool,
    num_threads: u32,
    use_wide: bool,
    has_gpu: bool,
) !CrackResult {
    _ = writer;

    ensureApr();

    const passmax: u32 = if (passmax_in == 0) MAX_DEFAULT else passmax_in;
    var threads = if (num_threads == 0) lib.getProcessorCount() / 2 else num_threads;
    if (threads == 0) threads = 1;

    c_digest_hash_def = hash_def;
    c.bf_shim_set(cDigest, hash_def.hash_length);

    var pool: ?*c.apr_pool_t = null;
    const st = c.apr_pool_create_ex(&pool, null, @as(c.apr_abortfunc_t, null), null);
    if (st != c.APR_SUCCESS or pool == null) {
        return .{ .password = null, .attempts = 0 };
    }
    defer _ = c.apr_pool_destroy(pool);

    const dict_z = try std.heap.c_allocator.dupeZ(u8, dict);
    defer std.heap.c_allocator.free(dict_z);
    const hash_z = try std.heap.c_allocator.dupeZ(u8, hash);
    defer std.heap.c_allocator.free(hash_z);

    var gpu_ctx_storage: gpu.GpuContext = .{};
    var gpu_ptr: ?*c.gpu_context_t = null;
    const want_gpu = has_gpu and hash_def.has_gpu_implementation;
    if (want_gpu) {
        if (gpu.contextFor(hash_def.name)) |gc| {
            gpu_ctx_storage = gc;
            gpu_ptr = @ptrCast(&gpu_ctx_storage);
        }
    }

    // During tests the C brute-force path prints probe/timings/result to stdout
    // (lib_printf -> vfprintf(stdout)). zig's --listen=- test IPC rides the same
    // fd 1, so the C output must be muted or it desyncs the protocol. On POSIX
    // the runner isolates its IPC fd, so redirecting fd 1 to /dev/null is enough
    // (unchanged from the original port). On Windows the runner writes IPC on
    // fd 1 itself, so an fd redirect would clobber those writes (WriteFailed) —
    // instead suppress lib_printf at the source via g_lib_output_suspended.
    const muted_stdout: ?c_int = if (builtin.is_test and builtin.os.tag != .windows) blk: {
        const null_fd = std.c.open("/dev/null", .{ .ACCMODE = .WRONLY });
        if (null_fd < 0) break :blk null;
        const saved = std.c.dup(std.posix.STDOUT_FILENO);
        if (saved < 0) {
            _ = std.c.close(null_fd);
            break :blk null;
        }
        if (std.c.dup2(null_fd, std.posix.STDOUT_FILENO) < 0) {
            _ = std.c.close(saved);
            _ = std.c.close(null_fd);
            break :blk null;
        }
        _ = std.c.close(null_fd);
        break :blk saved;
    } else null;
    const suspend_output = builtin.is_test and builtin.os.tag == .windows;
    if (suspend_output) c.bf_shim_set_output_suspended(1);
    defer {
        if (muted_stdout) |fd| {
            _ = std.c.dup2(fd, std.posix.STDOUT_FILENO);
            _ = std.c.close(fd);
        }
        if (suspend_output) c.bf_shim_set_output_suspended(0);
    }

    c.bf_crack_hash(
        dict_z.ptr,
        hash_z.ptr,
        passmin,
        passmax,
        hash_def.hash_length,
        cDigest,
        no_probe,
        threads,
        use_wide,
        want_gpu and gpu_ptr != null,
        gpu_ptr,
        pool,
    );

    const attempts = c.bf_get_attempts();
    if (c.bf_get_found_password()) |found| {
        const password = try allocator.dupe(u8, std.mem.span(found));
        return .{ .password = password, .attempts = attempts };
    }
    return .{ .password = null, .attempts = attempts };
}
