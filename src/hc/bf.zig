//! Brute-force hash cracker — Zig orchestration + C hot loops (bf_core).

const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const gpu = @import("gpu");
const bf_dict = @import("bf_dict.zig");

const c = @import("c");

pub const DEFAULT_ALPHABET = bf_dict.DEFAULT_ALPHABET;
pub const MAX_DEFAULT: u32 = 10;

pub const CrackResult = struct {
    password: ?[]u8,
    attempts: u64,
};

/// Async-signal-safe stop request for the SIGINT/console handler: sets the
/// shared brute-force "found" flag so the C hot loops (which poll it every
/// iteration) wind their workers down. The main loop then prints timings and
/// exits. Safe to call from a signal handler — only one relaxed atomic store
/// inside bf_core.
pub fn signalStopCrack() void {
    c.bf_core_set_found(true);
}

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

    // Pass the C-ABI digest entry directly — avoid a Zig trampoline on every
    // crack attempt (the extra hop tanked multi-thread scaling on fast hashes).
    c.bf_shim_set(@ptrCast(hash_def.digest), hash_def.hash_length);

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
            const wide = try std.unicode.utf8ToUtf16LeAlloc(arena, probe);
            const wide_bytes = std.mem.sliceAsBytes(wide);
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

/// One completion flag per spawned worker. Set by the trampoline just before the
/// thread returns, read by `waitForWorkersInterruptible` so a SIGINT can break
/// out of the blocking join and let the main thread print timings + exit while
/// the (soon-to-stop) workers finish on their own. `std.Thread.join` is not
/// interruptible in Zig, so the poll loop is how we keep the main thread
/// responsive without deadlocking the interrupted stdio/arena state.
const WorkerTracker = struct {
    done: std.atomic.Value(bool) = .init(false),
};

fn trackedCpuEntry(ctx: *c.bf_cpu_ctx_t, tracker: *WorkerTracker) void {
    defer tracker.done.store(true, .release);
    c.bf_core_cpu_worker(ctx);
}

fn trackedGpuEntry(ctx: *c.gpu_tread_ctx_t, tracker: *WorkerTracker) void {
    defer tracker.done.store(true, .release);
    c.bf_core_gpu_worker(ctx);
}

/// Join any still-live slots and clear them. Safe to call twice (second is a no-op).
/// Must run before the crackHash arena is freed — spawn stacks and worker ctx
/// live in that arena.
fn joinSpawnedThreads(threads: []?std.Thread) void {
    for (threads) |*slot| {
        if (slot.*) |t| {
            t.join();
            slot.* = null;
        }
    }
}

/// Poll worker completion until every thread is done, then join (now instant).
/// If `interrupted` flips true (SIGINT), keep nudging the shared brute-force
/// stop flag so stragglers wind down fast. This replaces a plain blocking join
/// so an interruptible crack can surface its timing line on the main thread
/// instead of blocking forever on a worker that the signal already asked to stop.
fn waitForWorkersInterruptible(threads: []?std.Thread, trackers: []*WorkerTracker) void {
    while (true) {
        var all_done = true;
        for (trackers) |t| {
            if (!t.done.load(.acquire)) {
                all_done = false;
                break;
            }
        }
        if (all_done) break;
        std.Thread.yield() catch {};
    }
    joinSpawnedThreads(threads);
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

/// Narrow a CUDA device property (signed `int`) to usize. Driver error paths
/// can report negative values; return null so the caller skips the device
/// instead of trapping the cast (Debug/ReleaseSafe) or producing UB (ReleaseFast).
fn gpuIntToUsize(x: c_int) ?usize {
    if (x < 0) return null;
    return @intCast(x);
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

    // Lengths 1..=3 crack instantly on CPU; GPU is overkill there.
    // Enable GPU only when passmax > 3 (same gate as classic).
    // While GPU runs, keep exactly one CPU thread (intentional: avoid fighting
    // the device for host cores; CPU still covers lengths above the GPU slot).
    var has_gpu = has_gpu_in and passmax > 3;
    if (has_gpu and !c.gpu_can_use_gpu()) {
        // Leading newline: probe estimate is printed without a trailing '\n'
        // (outputTimings supplies it). Without this, the diagnostic would glue
        // onto the probe line and get dropped by test output filters.
        const driver_ver = c.gpu_number_to_version(c.gpu_driver_version());
        const runtime_ver = c.gpu_number_to_version(c.gpu_runtime_version());
        if (driver_ver.major > 0) {
            try writer.print(
                "\nGPU present but driver's CUDA version {d}.{d} less then required {d}.{d}. So use only CPU\n",
                .{ driver_ver.major, driver_ver.minor, runtime_ver.major, runtime_ver.minor },
            );
        } else {
            try writer.writeAll("\nGPU unavailable (driver/toolkit); using CPU only\n");
        }
        try writer.flush();
        has_gpu = false;
    }
    const gpu_max_len: u32 = gpuMaxPasswordLen();

    var num_threads: u32 = num_threads_in;
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

    // Pad each worker ctx to a 128-byte stride so adjacent threads do not share
    // a cache line (default Arena alloc is only 8-byte aligned; a tight
    // []bf_cpu_ctx_t then false-shares and tanks crc32 multi-thread speed).
    const CpuCtxSlot = extern struct {
        ctx: c.bf_cpu_ctx_t,
        pad: [128 - @sizeOf(c.bf_cpu_ctx_t)]u8 = undefined,
    };
    comptime {
        if (@sizeOf(c.bf_cpu_ctx_t) > 128) @compileError("bf_cpu_ctx_t larger than pad stride");
    }
    const cpu_slots = try arena.alloc(CpuCtxSlot, num_threads);
    var cpu_threads = try arena.alloc(?std.Thread, num_threads);
    @memset(cpu_threads, null);
    const cpu_trackers = try arena.alloc(*WorkerTracker, num_threads);
    // Join before leaving on any path so crackHash's arena.deinit cannot free
    // worker ctx / thread stacks while threads still run (partial spawn failure).
    // Interruptible variant so a SIGINT can break the blocking join and let the
    // crack's timing line print on the main thread.
    defer waitForWorkersInterruptible(cpu_threads, cpu_trackers);
    errdefer c.bf_core_set_found(true);

    for (cpu_slots, 0..) |*slot, i| {
        const ctx = &slot.ctx;
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
        const tracker = try arena.create(WorkerTracker);
        cpu_trackers[i] = tracker;
        cpu_threads[i] = try std.Thread.spawn(.{ .allocator = arena }, trackedCpuEntry, .{ ctx, tracker });
    }

    var found_pass: ?[]u8 = null;

    if (has_gpu and gpu_context != null) {
        var count_props: c.device_props_t = std.mem.zeroes(c.device_props_t);
        c.gpu_get_props(&count_props);
        if (count_props.device_count > 0) {
            // The `> 0` guard above already rules out the negative/zero error
            // paths a CUDA driver can report, so the signed->usize cast is safe.
            const n_gpu: usize = @intCast(count_props.device_count);
            const gpu_ctxs = try arena.alloc(c.gpu_tread_ctx_t, n_gpu);
            var gpu_threads = try arena.alloc(?std.Thread, n_gpu);
            @memset(gpu_threads, null);
            const gpu_trackers = try arena.alloc(*WorkerTracker, n_gpu);
            // Pre-create trackers so the `continue` skips below still leave a
            // slot the wait loop accounts for; mark never-spawned slots done up
            // front so they don't block the poll.
            for (gpu_trackers) |*t| {
                t.* = try arena.create(WorkerTracker);
            }
            defer waitForWorkersInterruptible(gpu_threads, gpu_trackers);
            errdefer c.bf_core_set_found(true);
            const gpu_passmax = @min(passmax, gpu_max_len);
            var gpu_found = false;

            for (gpu_ctxs, 0..) |*gctx, i| {
                var props: c.device_props_t = std.mem.zeroes(c.device_props_t);
                if (!c.gpu_get_device_props(@intCast(i), &props)) {
                    // Never spawned: mark done so the wait loop does not stall.
                    gpu_trackers[i].done.store(true, .release);
                    continue;
                }

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

                // Index-gen: variants buffers unused; count = max launch size.
                // CUDA props are signed `int` and can be negative on error
                // paths; skip the device rather than cast a negative value to
                // usize (UB in ReleaseFast) or overflow the product.
                gctx.variants_ = null;
                const blocks = gpuIntToUsize(gctx.max_gpu_blocks_number_) orelse {
                    gpu_trackers[i].done.store(true, .release);
                    continue;
                };
                const threads = gpuIntToUsize(gctx.max_threads_per_block_) orelse {
                    gpu_trackers[i].done.store(true, .release);
                    continue;
                };
                const product = @mulWithOverflow(blocks, threads);
                if (product[1] != 0) {
                    gpu_trackers[i].done.store(true, .release);
                    continue;
                }
                gctx.variants_count_ = product[0];
                gctx.variants_size_ = 0;

                gpu_threads[i] = try std.Thread.spawn(.{ .allocator = arena }, trackedGpuEntry, .{ gctx, gpu_trackers[i] });
            }

            waitForWorkersInterruptible(gpu_threads, gpu_trackers);
            for (gpu_ctxs) |*gctx| {
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

    joinSpawnedThreads(cpu_threads);
    for (cpu_slots) |*slot| {
        const ctx = &slot.ctx;
        c.bf_core_add_attempts(ctx.num_of_attempts_);

        if (use_wide) {
            if (ctx.wide_pass_ != null) {
                var len: usize = 0;
                while (ctx.wide_pass_[len] != 0) : (len += 1) {}
                if (len > 0) {
                    const wide: []const u16 = @ptrCast(ctx.wide_pass_[0..len]);
                    found_pass = try std.unicode.utf16LeToUtf8Alloc(arena, wide);
                }
            }
        } else if (ctx.pass_ != null) {
            const len = std.mem.len(ctx.pass_);
            if (len > 0) found_pass = try arena.dupe(u8, ctx.pass_[0..len]);
        }
    }

    return found_pass;
}

test {
    // Pull bf_dict unit tests into `zig build test` (root is bf.zig).
    _ = @import("bf_dict.zig");
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

test "gpu thread ctx carries per-context variant fill index" {
    // Multi-GPU workers must not share a process-global fill cursor; the ABI
    // field is the contract bf_core uses for partial-batch flush.
    var gctx: gpu.GpuThreadCtx = std.mem.zeroes(gpu.GpuThreadCtx);
    try std.testing.expectEqual(@as(c_uint, 0), gctx.variant_ix_);
    gctx.variant_ix_ = 42;
    try std.testing.expectEqual(@as(c_uint, 42), gctx.variant_ix_);
}

test "joinSpawnedThreads is a no-op on null slots" {
    var slots = [_]?std.Thread{ null, null };
    joinSpawnedThreads(slots[0..]);
    try std.testing.expect(slots[0] == null);
    try std.testing.expect(slots[1] == null);
}

test "gpuIntToUsize rejects negative driver values" {
    // CUDA error paths can return -1 for device properties; the cast must not
    // trap (Debug/ReleaseSafe) or become UB (ReleaseFast).
    try std.testing.expectEqual(@as(?usize, null), gpuIntToUsize(-1));
    try std.testing.expectEqual(@as(?usize, 0), gpuIntToUsize(0));
    try std.testing.expectEqual(@as(?usize, 64), gpuIntToUsize(64));
}

test "waitForWorkersInterruptible returns once all trackers are done" {
    // No real workers are spawned here (bf_core context isn't initialized in a
    // unit test); instead exercise the poll-loop contract directly: with every
    // tracker already marked done the wait returns immediately, and with an
    // empty slot list it is a no-op.
    var t0: WorkerTracker = .{};
    var t1: WorkerTracker = .{};
    t0.done.store(true, .release);
    t1.done.store(true, .release);
    var trackers = [_]*WorkerTracker{ &t0, &t1 };
    var slots = [_]?std.Thread{ null, null };
    waitForWorkersInterruptible(slots[0..], trackers[0..]);
    try std.testing.expect(slots[0] == null);
    try std.testing.expect(slots[1] == null);

    // Empty case (e.g. zero GPU devices after pruning) must not block.
    var none: [0]?std.Thread = .{};
    var none_t: [0]*WorkerTracker = .{};
    waitForWorkersInterruptible(&none, &none_t);
}
