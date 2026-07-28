//! Brute-force hash cracker — Zig orchestration + C hot loops (bf_core), no APR.

const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const gpu = @import("gpu");
const bf_dict = @import("bf_dict.zig");

const c = @import("c");

pub const MAX_DEFAULT: u32 = 10;
pub const ansiToWide = bf_dict.ansiToWide;

pub const CrackResult = struct {
    password: ?[]u8,
    attempts: u64,
};

fn cDigest(
    digest: [*c]u8,
    string: ?*const anyopaque,
    input_len: usize,
) callconv(.c) void {
    const ctx: *const hashes.HashDefinition = @ptrCast(@alignCast(c_digest_hash_def));
    ctx.digest(@ptrCast(digest), @ptrCast(string), input_len);
}

var c_digest_hash_def: ?*const hashes.HashDefinition = null;

/// Live attempts for SIGINT (reads bf_core attempt counter).
pub fn getAttempts() u64 {
    return c.bf_core_get_attempts();
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
    // Clamp to a u64-safe bound: max_attempts = pow(dictlen, passmax) for long
    // passwords exceeds maxInt(u64), and @floatFromInt(maxInt(u64)) rounds above
    // it, so @intFromFloat would trap in Debug/ReleaseSafe / be UB in ReleaseFast.
    // 2^63 is exactly representable and far beyond any displayable attempt count.
    const CLAMP: f64 = @floatFromInt(@as(u64, 1) << 63);
    return formatCommify(buf, @intFromFloat(@round(@min(value, CLAMP))));
}

fn digestToHexUpper(digest: []const u8, out: []u8) []const u8 {
    for (digest, 0..) |b, i| {
        _ = std.fmt.bufPrint(out[i * 2 ..][0..2], "{X:0>2}", .{b}) catch unreachable;
    }
    return out[0 .. digest.len * 2];
}

/// Full crack path: probe, CPU/GPU workers, timings, result (no APR).
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
    const passmax: u32 = if (passmax_in == 0) MAX_DEFAULT else passmax_in;
    var threads = if (num_threads == 0) lib.getProcessorCount() / 2 else num_threads;
    if (threads == 0) threads = 1;

    c_digest_hash_def = hash_def;
    c.bf_shim_set(cDigest, hash_def.hash_length);

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const hash_z = try arena.dupeZ(u8, hash);

    var digest_buf: [64]u8 = undefined;
    const digest = digest_buf[0..hash_def.hash_length];
    @memset(digest, 0);

    // Empty string validation — same order as classic bf_crack_hash:
    // timings first, then "Initial string is: Empty string".
    hash_def.digest(digest.ptr, "".ptr, 0);
    if (c.bf_compare_hash(digest.ptr, hash_z.ptr) != 0) {
        lib.startTimer();
        const attempts = c.bf_core_get_attempts();
        try printTimings(writer, attempts);
        try printResult(writer, "Empty string");
        return .{ .password = try allocator.dupe(u8, ""), .attempts = attempts };
    }

    var gpu_ctx_storage: gpu.GpuContext = .{};
    var gpu_ptr: ?*c.gpu_context_t = null;
    const want_gpu = has_gpu and hash_def.has_gpu_implementation;
    if (want_gpu) {
        if (gpu.contextFor(hash_def.name)) |gc| {
            gpu_ctx_storage = gc;
            gpu_ptr = @ptrCast(&gpu_ctx_storage);
        }
    }

    if (!no_probe) {
        const probe = "123";
        if (use_wide) {
            var wide = [_]u16{ '1', '2', '3' };
            const wide_bytes = std.mem.sliceAsBytes(wide[0..]);
            hash_def.digest(digest.ptr, wide_bytes.ptr, wide_bytes.len);
        } else {
            hash_def.digest(digest.ptr, probe.ptr, probe.len);
        }
        var hexbuf: [128]u8 = undefined;
        const hex = digestToHexUpper(digest, &hexbuf);

        lib.startTimer();
        _ = try runBruteForce(
            arena,
            writer,
            bf_dict.DEFAULT_ALPHABET,
            hex,
            1,
            MAX_DEFAULT,
            threads,
            use_wide,
            false,
            null,
        );
        lib.stopTimer();
        const probe_time = lib.readElapsedTime();
        const probe_attempts = c.bf_core_get_attempts();
        const ratio = if (probe_time.total_seconds > 0)
            @as(f64, @floatFromInt(probe_attempts)) / probe_time.total_seconds
        else
            0;

        const prepared = try bf_dict.prepareDictionary(arena, dict);
        const max_attempts = std.math.pow(f64, @floatFromInt(prepared.len), @floatFromInt(passmax));
        const max_time = lib.normalizeTime(if (ratio > 0) max_attempts / ratio else 0);
        var time_msg: [64]u8 = undefined;
        var tw: std.Io.Writer = .fixed(&time_msg);
        lib.formatTime(max_time, &tw) catch {};
        const time_s = std.Io.Writer.buffered(&tw);
        var max_buf: [64]u8 = undefined;
        const max_s = formatCommifyF(&max_buf, max_attempts);
        // No trailing newline: bf_output_timings historically starts with
        // lib_new_line(), which both ends this line and separates Attempts.
        try writer.print("May take approximatelly: {s} ({s} attempts)", .{ time_s, max_s });
        try writer.flush();
    }

    lib.startTimer();
    const found = try runBruteForce(
        arena,
        writer,
        dict,
        hash,
        passmin,
        passmax,
        threads,
        use_wide,
        want_gpu and gpu_ptr != null,
        if (gpu_ptr != null) &gpu_ctx_storage else null,
    );
    const attempts = c.bf_core_get_attempts();
    try printTimings(writer, attempts);

    if (found) |pw| {
        try printResult(writer, pw);
        return .{ .password = try allocator.dupe(u8, pw), .attempts = attempts };
    }
    try printResult(writer, null);
    return .{ .password = null, .attempts = attempts };
}

fn printTimings(writer: *std.Io.Writer, attempts: u64) !void {
    lib.stopTimer();
    try outputTimings(writer, attempts, lib.readElapsedTime());
}

fn printResult(writer: *std.Io.Writer, password: ?[]const u8) !void {
    if (password) |pw| {
        try writer.print("Initial string is: {s}\n", .{pw});
    } else {
        try writer.writeAll("Nothing found\n");
    }
}

fn cpuWorkerEntry(ctx: *c.bf_cpu_ctx_t) void {
    c.bf_core_cpu_worker(ctx);
}

fn gpuWorkerEntry(ctx: *c.gpu_tread_ctx_t) void {
    c.bf_core_gpu_worker(ctx);
}

/// Max password length that fits in a GPU attempt slot (trailing NUL included
/// in `GPU_ATTEMPT_SIZE`). Longer lengths stay on the CPU path.
fn gpuMaxPasswordLen() u32 {
    return @intCast(gpu.GPU_ATTEMPT_SIZE - 1);
}

/// Whether to signal CPU workers to stop after GPU threads join.
/// Only a GPU hit should stop CPU; a miss must not (short lengths / lengths
/// beyond the GPU slot still need the CPU path).
fn shouldStopCpuAfterGpu(gpu_found: bool) bool {
    return gpu_found;
}

test "gpuMaxPasswordLen leaves room for trailing NUL" {
    try std.testing.expectEqual(@as(u32, @intCast(gpu.GPU_ATTEMPT_SIZE - 1)), gpuMaxPasswordLen());
    try std.testing.expect(gpuMaxPasswordLen() >= 3);
}

test "formatCommifyF does not trap on overflow attempt counts" {
    // pow(dictlen, passmax) for -x 13+ exceeds maxInt(u64); previously this
    // trapped @intFromFloat. It must clamp and format a large number instead.
    var buf: [64]u8 = undefined;
    const s = formatCommifyF(&buf, @as(f64, 2.0e23));
    try std.testing.expect(s.len > 0);
    // Still contains only digits and the space separator, no panic.
    for (s) |ch| try std.testing.expect((ch >= '0' and ch <= '9') or ch == ' ');
}

test "shouldStopCpuAfterGpu only on hit" {
    try std.testing.expect(!shouldStopCpuAfterGpu(false));
    try std.testing.expect(shouldStopCpuAfterGpu(true));
}

fn runBruteForce(
    arena: std.mem.Allocator,
    writer: *std.Io.Writer,
    dict: []const u8,
    hash_hex: []const u8,
    passmin: u32,
    passmax: u32,
    num_threads_in: u32,
    use_wide: bool,
    has_gpu_in: bool,
    gpu_context: ?*gpu.GpuContext,
) !?[]u8 {
    if (passmax > std.math.maxInt(c_int) / @sizeOf(c_int)) {
        try writer.print("Max string length is too big: {d}\n", .{passmax});
        return null;
    }

    // GPU stays on for passmax > 3 (classic), but its attempt_/variants_ slots
    // are fixed at GPU_ATTEMPT_SIZE (trailing NUL included), so the GPU worker
    // only searches up to that length. CPU keeps the full passmax.
    var has_gpu = has_gpu_in and passmax > 3;
    if (has_gpu and !c.gpu_can_use_gpu()) {
        has_gpu = false;
    }
    const gpu_max_len: u32 = gpuMaxPasswordLen();

    var num_threads = num_threads_in;
    if (has_gpu) num_threads = 1;

    const prepared = try bf_dict.prepareDictionary(arena, dict);
    if (prepared.len <= num_threads) {
        num_threads = @intCast(@max(prepared.len, 1));
    }

    const hash_z = try arena.dupeZ(u8, hash_hex);
    const hash_bytes = try arena.alloc(u8, c.bf_shim_hash_len());
    c.bf_shim_hash_to_bytes(hash_z.ptr, hash_bytes.ptr);

    c.bf_core_reset();
    c.bf_core_set_context(prepared.ptr, prepared.len, hash_bytes.ptr, c.bf_compare_hash_attempt);

    const cpu_ctxs = try arena.alloc(c.bf_cpu_ctx_t, num_threads);
    var cpu_threads = try arena.alloc(?std.Thread, num_threads);

    for (cpu_ctxs, 0..) |*ctx, i| {
        ctx.* = std.mem.zeroes(c.bf_cpu_ctx_t);
        ctx.passmin_ = passmin;
        ctx.passmax_ = passmax;
        ctx.work_thread_ = 1;
        ctx.thread_num_ = i + 1;
        ctx.pass_ = (try arena.alloc(u8, passmax + 1)).ptr;
        @memset(ctx.pass_[0 .. passmax + 1], 0);
        const wide_buf = try arena.alloc(c.bf_wide_char_t, passmax + 1);
        @memset(std.mem.sliceAsBytes(wide_buf), 0);
        ctx.wide_pass_ = wide_buf.ptr;
        ctx.chars_indexes_ = (try arena.alloc(usize, passmax)).ptr;
        ctx.pass_length_ = passmin;
        ctx.num_of_threads = @intCast(num_threads);
        ctx.use_wide_pass_ = use_wide;
        ctx.found_in_the_thread_ = false;
        cpu_threads[i] = try std.Thread.spawn(.{ .allocator = arena }, cpuWorkerEntry, .{ctx});
    }

    var found_pass: ?[]u8 = null;

    if (has_gpu and gpu_context != null) {
        var props: c.device_props_t = std.mem.zeroes(c.device_props_t);
        c.gpu_get_props(&props);
        if (props.device_count > 0) {
            const n_gpu: usize = @intCast(props.device_count);
            const gpu_ctxs = try arena.alloc(c.gpu_tread_ctx_t, n_gpu);
            var gpu_threads = try arena.alloc(?std.Thread, n_gpu);
            const gpu_passmax = @min(passmax, gpu_max_len);
            var gpu_found = false;

            for (gpu_ctxs, 0..) |*gctx, i| {
                gctx.* = std.mem.zeroes(c.gpu_tread_ctx_t);
                gctx.passmin_ = passmin;
                gctx.passmax_ = gpu_passmax;
                gctx.attempt_ = (try arena.alloc(u8, gpu.GPU_ATTEMPT_SIZE)).ptr;
                @memset(gctx.attempt_[0..gpu.GPU_ATTEMPT_SIZE], 0);
                gctx.result_ = (try arena.alloc(u8, gpu.GPU_ATTEMPT_SIZE)).ptr;
                @memset(gctx.result_[0..gpu.GPU_ATTEMPT_SIZE], 0);
                gctx.pass_length_ = passmin;
                gctx.max_gpu_blocks_number_ = props.max_blocks_number;
                gctx.multiprocessor_count_ = props.multiprocessor_count;
                const dec = gpu_context.?.max_threads_decrease_factor_;
                gctx.max_threads_per_block_ = @divTrunc(props.max_threads_per_block, if (dec == 0) 1 else dec);
                gctx.device_ix_ = @intCast(i);
                gctx.gpu_context_ = @ptrCast(gpu_context.?);
                gctx.use_wide_pass_ = use_wide;
                gctx.max_threads_decrease_factor_ = dec;
                gctx.comparisons_per_iteration_ = gpu_context.?.comparisons_per_iteration_;
                gctx.pool_ = null;

                const variants_count: usize = @as(usize, @intCast(gctx.max_gpu_blocks_number_)) *
                    @as(usize, @intCast(gctx.max_threads_per_block_));
                const variants_size = variants_count * gpu.GPU_ATTEMPT_SIZE;
                const variants = try arena.alloc(u8, variants_size);
                @memset(variants, 0);
                gctx.variants_ = variants.ptr;
                gctx.variants_count_ = variants_count;
                gctx.variants_size_ = variants_size;

                gpu_threads[i] = try std.Thread.spawn(.{ .allocator = arena }, gpuWorkerEntry, .{gctx});
            }

            for (gpu_threads, 0..) |th, i| {
                if (th) |t| t.join();
                const gctx = &gpu_ctxs[i];
                if (gctx.found_in_the_thread_ and gctx.result_ != null) {
                    const len = std.mem.len(gctx.result_);
                    found_pass = try arena.dupe(u8, gctx.result_[0..len]);
                    gpu_found = true;
                }
            }
            // Classic always set found after GPU join (even on a miss), which
            // can abort CPU before it finishes shorter lengths. Only stop CPU
            // when GPU actually found a password; on a miss CPU keeps going
            // (also covers passmax beyond the GPU slot size — see #10).
            if (shouldStopCpuAfterGpu(gpu_found)) {
                c.bf_core_set_found(true);
            }
        }
    }

    for (cpu_threads, 0..) |th, i| {
        if (th) |t| t.join();
        const ctx = &cpu_ctxs[i];
        c.bf_core_add_attempts(ctx.num_of_attempts_);

        if (use_wide) {
            if (ctx.wide_pass_ != null) {
                var len: usize = 0;
                while (ctx.wide_pass_[len] != 0) : (len += 1) {}
                if (len > 0) {
                    const wide: []const u16 = @ptrCast(ctx.wide_pass_[0..len]);
                    found_pass = try bf_dict.wideToAnsi(arena, wide);
                }
            }
        } else if (ctx.pass_ != null) {
            const len = std.mem.len(ctx.pass_);
            if (len > 0) found_pass = try arena.dupe(u8, ctx.pass_[0..len]);
        }
    }

    return found_pass;
}
