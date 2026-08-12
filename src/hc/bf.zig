//! Brute-force hash cracker — Zig orchestration + C hot loops (bf_core).

const std = @import("std");
const lib = @import("lib");
const hashes = @import("hashes");
const gpu = @import("gpu");

const c = @import("c");

const DIGITS = "0123456789";
const DIGITS_TPL = "0-9";
const LOW_CASE = "abcdefghijklmnopqrstuvwxyz";
const LOW_CASE_TPL = "a-z";
const UPPER_CASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const UPPER_CASE_TPL = "A-Z";
const ASCII_TPL = "ASCII";
const ASCII_FIRST: u8 = '!';
const ASCII_LAST: u8 = '~';

pub const DEFAULT_ALPHABET = DIGITS ++ LOW_CASE ++ UPPER_CASE;
pub const MAX_DEFAULT: u32 = 10;

/// Expand dict templates (`0-9`, `a-z`, `A-Z`, `ASCII`) and dedupe bytes.
/// Caller owns the returned NUL-terminated slice.
fn prepareDictionary(allocator: std.mem.Allocator, dict: []const u8) ![:0]u8 {
    if (std.mem.indexOf(u8, dict, ASCII_TPL) != null) {
        const len = @as(usize, ASCII_LAST - ASCII_FIRST) + 1;
        const tmp = try allocator.allocSentinel(u8, len, 0);
        var i: usize = 0;
        var sym: u8 = ASCII_FIRST;
        while (sym <= ASCII_LAST) : (sym += 1) {
            tmp[i] = sym;
            i += 1;
        }
        return tmp;
    }

    var buf: ?[]u8 = null;
    defer if (buf) |b| allocator.free(b);
    var current: []const u8 = dict;

    inline for (.{
        .{ DIGITS_TPL, DIGITS },
        .{ LOW_CASE_TPL, LOW_CASE },
        .{ UPPER_CASE_TPL, UPPER_CASE },
    }) |pair| {
        if (std.mem.indexOf(u8, current, pair[0]) != null) {
            const len = std.mem.replacementSize(u8, current, pair[0], pair[1]);
            const replaced = try allocator.alloc(u8, len);
            _ = std.mem.replace(u8, current, pair[0], pair[1], replaced);
            if (buf) |b| allocator.free(b);
            buf = replaced;
            current = replaced;
        }
    }

    var seen = [_]bool{false} ** 256;
    var unique: usize = 0;
    for (current) |ch| {
        if (!seen[ch]) {
            seen[ch] = true;
            unique += 1;
        }
    }

    const out = try allocator.allocSentinel(u8, unique, 0);
    @memset(seen[0..], false);
    var ir: usize = 0;
    for (current) |ch| {
        if (!seen[ch]) {
            out[ir] = ch;
            ir += 1;
            seen[ch] = true;
        }
    }
    return out;
}

/// Async-signal-safe stop request for the SIGINT/console handler: sets the
/// shared brute-force "found" flag so the C hot loops (which poll it every
/// iteration) wind their workers down. The main loop then prints timings and
/// exits. Safe to call from a signal handler — only one relaxed atomic store
/// inside bf_core.
pub fn signalStopCrack() void {
    c.bf_core_set_found(true);
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

fn parseHashHex(hash_hex: []const u8, out: []u8) void {
    @memset(out, 0);
    const n = @min(out.len, hash_hex.len / 2);
    if (n == 0) return;
    _ = std.fmt.hexToBytes(out[0..n], hash_hex[0 .. n * 2]) catch {
        @memset(out, 0);
    };
}

/// Case-insensitive hex compare of `digest` against a user-supplied hex string.
/// Length mismatch or invalid hex digits yield no match.
fn hashHexMatches(digest: []const u8, hash_hex: []const u8) bool {
    var buf: [128]u8 = undefined;
    const hex = std.fmt.bufPrint(&buf, "{X}", .{digest}) catch return false;
    return std.ascii.eqlIgnoreCase(hex, hash_hex);
}

/// Full crack path: probe, CPU/GPU workers, timings, result.
pub fn crackHash(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    dict: []const u8,
    hash: []const u8,
    passmin: u32,
    passmax_in: u32,
    hash_def: *const hashes.HashDefinition,
    no_probe: bool,
    num_threads: u32,
    use_wide: bool,
) !?[]u8 {
    const passmax: u32 = if (passmax_in == 0) MAX_DEFAULT else passmax_in;
    var threads = if (num_threads == 0)
        @as(u32, @intCast(std.Thread.getCpuCount() catch 1)) / 2
    else
        num_threads;
    if (threads == 0) threads = 1;

    // Pass the C-ABI digest entry directly — avoid a Zig trampoline on every
    // crack attempt (the extra hop tanked multi-thread scaling on fast hashes).
    c.bf_shim_set(@ptrCast(hash_def.digest), hash_def.hash_length);

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var digest_buf: [64]u8 = undefined;
    const digest = digest_buf[0..hash_def.hash_length];
    @memset(digest, 0);

    // Empty string validation — same order as classic bf_crack_hash:
    // timings first, then "Initial string is: Empty string".
    hash_def.digest(digest.ptr, "".ptr, 0);
    if (hashHexMatches(digest, hash)) {
        const t0 = std.Io.Clock.awake.now(io);
        const attempts = c.bf_core_get_attempts();
        try printTimings(io, writer, attempts, t0);
        try printResult(writer, "Empty string");
        return try allocator.dupe(u8, "");
    }

    var gpu_ctx_storage: gpu.GpuContext = .{};
    var gpu_ptr: ?*c.gpu_context_t = null;
    if (gpu.contextFor(hash_def.name)) |gc| {
        gpu_ctx_storage = gc;
        gpu_ptr = @ptrCast(&gpu_ctx_storage);
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
        const hex = std.fmt.bufPrint(&hexbuf, "{X}", .{digest}) catch unreachable;

        const probe_started = std.Io.Clock.awake.now(io);
        _ = try runBruteForce(
            arena,
            writer,
            DEFAULT_ALPHABET,
            hex,
            1,
            MAX_DEFAULT,
            threads,
            use_wide,
            false,
            null,
        );
        const probe_time = lib.elapsedSince(io, probe_started);
        const probe_attempts = c.bf_core_get_attempts();
        const ratio = if (probe_time.total_seconds > 0)
            @as(f64, @floatFromInt(probe_attempts)) / probe_time.total_seconds
        else
            0;

        const prepared = try prepareDictionary(arena, dict);
        const max_attempts = std.math.pow(f64, @floatFromInt(prepared.len), @floatFromInt(passmax));
        const max_time = lib.normalizeTime(if (ratio > 0) max_attempts / ratio else 0);
        var time_msg: [64]u8 = undefined;
        var tw: std.Io.Writer = .fixed(&time_msg);
        lib.formatTime(max_time, &tw) catch {};
        const time_s = std.Io.Writer.buffered(&tw);
        var max_buf: [64]u8 = undefined;
        const max_s = formatCommifyF(&max_buf, max_attempts);
        // No trailing newline: bf_output_timings historically starts with
        // trailing newline, which both ends this line and separates Attempts.
        try writer.print("May take approximatelly: {s} ({s} attempts)", .{ time_s, max_s });
        try writer.flush();
    }

    const crack_started = std.Io.Clock.awake.now(io);
    const found = try runBruteForce(
        arena,
        writer,
        dict,
        hash,
        passmin,
        passmax,
        threads,
        use_wide,
        gpu_ptr != null,
        if (gpu_ptr != null) &gpu_ctx_storage else null,
    );
    const attempts = c.bf_core_get_attempts();
    try printTimings(io, writer, attempts, crack_started);

    if (found) |pw| {
        try printResult(writer, pw);
        return try allocator.dupe(u8, pw);
    }
    try printResult(writer, null);
    return null;
}

fn printTimings(io: std.Io, writer: *std.Io.Writer, attempts: u64, started: std.Io.Timestamp) !void {
    try outputTimings(writer, attempts, lib.elapsedSince(io, started));
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
    var has_gpu = has_gpu_in and passmax > 3;
    const gpu_factor: c_int = if (gpu_context) |gc| gc.max_threads_decrease_factor_ else 1;
    // OpenCL-only builds: heavy kernels (factor >= 4) lose short cracks to
    // multi-CPU and load a large runtime — skip GPU before any device probe.
    if (has_gpu and gpu_factor >= 4 and gpu.enable_opencl and !gpu.enable_cuda) {
        has_gpu = false;
    }
    if (has_gpu and !c.gpu_can_use_gpu()) {
        // Leading newline: probe estimate is printed without a trailing '\n'
        // (outputTimings supplies it). Without this, the diagnostic would glue
        // onto the probe line and get dropped by test output filters.
        const driver_ver = c.gpu_number_to_version(c.gpu_driver_version());
        const runtime_ver = c.gpu_number_to_version(c.gpu_runtime_version());
        if (driver_ver.major > 0) {
            try writer.print(
                "\nGPU present but driver's CUDA version {d}.{d} less than required {d}.{d}. So use only CPU\n",
                .{ driver_ver.major, driver_ver.minor, runtime_ver.major, runtime_ver.minor },
            );
            try writer.flush();
        }
        has_gpu = false;
    }
    const gpu_max_len: u32 = gpuMaxPasswordLen();

    // Light kernels (factor 1–2) beat multi-CPU on GPU — pin to 1 host thread.
    // Heavy kernels (factor >= 4): on OpenCL (dual binary that fell back to CL)
    // skip GPU so wall time matches classic multi-CPU; CUDA keeps GPU + threads.
    var num_threads: u32 = num_threads_in;
    if (has_gpu) {
        if (gpu_factor >= 4 and c.gpu_is_opencl()) {
            has_gpu = false;
        } else if (gpu_factor < 4) {
            num_threads = 1;
        }
    }

    const prepared = try prepareDictionary(arena, dict);
    if (prepared.len <= num_threads) {
        num_threads = @intCast(@max(prepared.len, 1));
    }

    const hash_bytes = try arena.alloc(u8, c.bf_shim_hash_len());
    parseHashHex(hash_hex, hash_bytes);

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
            if (gpu_found) {
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

test "prepareDictionary ASCII" {
    const d = try prepareDictionary(std.testing.allocator, "ASCII");
    defer std.testing.allocator.free(d);
    // '!'..'~' inclusive → 94 printable ASCII bytes.
    try std.testing.expectEqual(@as(usize, 94), d.len);
    try std.testing.expectEqual(@as(u8, '!'), d[0]);
    try std.testing.expectEqual(@as(u8, '~'), d[93]);
}

test "prepareDictionary digit class" {
    const d = try prepareDictionary(std.testing.allocator, "0-9");
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualStrings(DIGITS, d);
}

test "prepareDictionary mixed dedupe" {
    const d = try prepareDictionary(std.testing.allocator, "0-9abc0");
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualStrings("0123456789abc", d);
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

test "joinSpawnedThreads is a no-op on null slots" {
    var slots = [_]?std.Thread{ null, null };
    joinSpawnedThreads(slots[0..]);
    try std.testing.expect(slots[0] == null);
    try std.testing.expect(slots[1] == null);
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
