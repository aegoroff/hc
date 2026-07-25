//! Brute force hash cracker (Zig 0.16 port of src/srclib/bf.c).
//! Crypto stays in C; this module only drives CPU enumeration and
//! delegates hashing to the `hashes` module's digest callbacks.
//!
//! GPU path is intentionally a no-op stub (CUDA is Task 9); bruteForce
//! always runs the CPU search regardless of `has_gpu`.

const std = @import("std");
const builtin = @import("builtin");
const lib = @import("lib");
const hashes = @import("hashes");

pub const DIGITS = "0123456789";
pub const LOW_CASE = "abcdefghijklmnopqrstuvwxyz";
pub const UPPER_CASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const DIGITS_TPL = "0-9";
const LOW_CASE_TPL = "a-z";
const UPPER_CASE_TPL = "A-Z";
const ASCII_TPL = "ASCII";

pub const MAX_DEFAULT: u32 = 10;
const CPU_FLUSH: u64 = 200_000;

const k_ascii_first: u8 = '!';
const k_ascii_last: u8 = '~';

pub const default_alphabet: []const u8 = DIGITS ++ LOW_CASE ++ UPPER_CASE;

pub const CrackOptions = struct {
    passmin: u32,
    passmax: u32 = 0,
    dict: []const u8,
    hash: []const u8,
    hash_def: *const hashes.HashDefinition,
    num_threads: u32 = 0,
    use_wide_pass: bool = false,
    has_gpu: bool = false,
};

pub const CrackResult = struct {
    password: ?[]u8,
    attempts: u64,
};

const Shared = struct {
    dict: []const u8,
    target: []const u8,
    hash_len: usize,
    digest_fn: hashes.DigestFn,
    passmin: u32,
    passmax: u32,
    use_wide: bool,
    found: std.atomic.Value(bool),
    attempts: std.atomic.Value(u64),
    result_buf: []u8,
    result_len: usize,
    result_set: bool,
};

const WorkerScratch = struct {
    indices: []usize,
    bytes: []u8,
    digest: [64]u8 align(8) = undefined,
    attempts: u64 = 0,
};

fn allocBytes(allocator: std.mem.Allocator, passmax: u32, use_wide: bool) ![]u8 {
    const len: usize = if (use_wide) @as(usize, passmax) * 2 else passmax;
    return allocator.alloc(u8, @max(len, 1));
}

fn spaceSize(base: usize, len: u32) u128 {
    if (base == 0) return 0;
    var result: u128 = 1;
    var i: u32 = 0;
    while (i < len) : (i += 1) {
        const res = @mulWithOverflow(result, @as(u128, base));
        if (res[1] != 0) return std.math.maxInt(u128);
        result = res[0];
    }
    return result;
}

fn flushAttempts(sh: *Shared, scratch: *WorkerScratch) void {
    if (scratch.attempts != 0) {
        _ = sh.attempts.fetchAdd(scratch.attempts, .monotonic);
        scratch.attempts = 0;
    }
}

fn worker(sh: *Shared, stride: u32, id: u32, scratch: *WorkerScratch) void {
    const base = sh.dict.len;
    var L: u32 = sh.passmin;
    while (L <= sh.passmax) : (L += 1) {
        const space = spaceSize(base, L);
        var pos: u128 = id;
        while (pos < space) : (pos += stride) {
            if (sh.found.load(.acquire)) {
                flushAttempts(sh, scratch);
                return;
            }

            var rem: u128 = pos;
            var k: usize = L;
            while (k > 0) {
                k -= 1;
                scratch.indices[k] = @intCast(rem % base);
                rem /= base;
            }

            const b = scratch.bytes;
            const in_len: usize = if (sh.use_wide) blk: {
                var j: usize = 0;
                while (j < L) : (j += 1) {
                    b[j * 2] = sh.dict[scratch.indices[j]];
                    b[j * 2 + 1] = 0;
                }
                break :blk @as(usize, L) * 2;
            } else blk: {
                var j: usize = 0;
                while (j < L) : (j += 1) b[j] = sh.dict[scratch.indices[j]];
                break :blk L;
            };

            sh.digest_fn(&scratch.digest, b.ptr, in_len);
            scratch.attempts += 1;
            if (scratch.attempts >= CPU_FLUSH) flushAttempts(sh, scratch);

            if (std.mem.eql(u8, scratch.digest[0..sh.hash_len], sh.target[0..sh.hash_len])) {
                flushAttempts(sh, scratch);
                if (sh.found.cmpxchgStrong(false, true, .acq_rel, .acquire) == null) {
                    var j: usize = 0;
                    while (j < L) : (j += 1) sh.result_buf[j] = sh.dict[scratch.indices[j]];
                    sh.result_len = L;
                    sh.result_set = true;
                }
                return;
            }
        }
    }
    flushAttempts(sh, scratch);
}

pub fn prepareDictionary(allocator: std.mem.Allocator, spec: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, spec, ASCII_TPL) != null) {
        const count: usize = @as(usize, k_ascii_last) - @as(usize, k_ascii_first) + 1;
        const buf = try allocator.alloc(u8, count);
        var i: usize = 0;
        while (i < count) : (i += 1) buf[i] = k_ascii_first + @as(u8, @intCast(i));
        return buf;
    }

    var cur = try allocator.dupe(u8, spec);
    cur = try replaceOwned(allocator, cur, DIGITS_TPL, DIGITS);
    cur = try replaceOwned(allocator, cur, LOW_CASE_TPL, LOW_CASE);
    cur = try replaceOwned(allocator, cur, UPPER_CASE_TPL, UPPER_CASE);

    var seen = [_]bool{false} ** 256;
    const out = try allocator.alloc(u8, cur.len);
    var w: usize = 0;
    for (cur) |c| {
        if (!seen[c]) {
            seen[c] = true;
            out[w] = c;
            w += 1;
        }
    }
    return out[0..w];
}

fn replaceOwned(allocator: std.mem.Allocator, haystack: []const u8, needle: []const u8, replacement: []const u8) ![]u8 {
    if (needle.len == 0) return allocator.dupe(u8, haystack);

    var count: usize = 0;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) {
        if (std.mem.eql(u8, haystack[i..][0..needle.len], needle)) {
            count += 1;
            i += needle.len;
        } else {
            i += 1;
        }
    }

    const removed = count * needle.len;
    const added = count * replacement.len;
    const total = haystack.len - removed + added;
    var out = try allocator.alloc(u8, total);

    var w: usize = 0;
    i = 0;
    while (i < haystack.len) {
        if (i + needle.len <= haystack.len and std.mem.eql(u8, haystack[i..][0..needle.len], needle)) {
            @memcpy(out[w..][0..replacement.len], replacement);
            w += replacement.len;
            i += needle.len;
        } else {
            out[w] = haystack[i];
            w += 1;
            i += 1;
        }
    }
    return out;
}

pub fn createDigest(allocator: std.mem.Allocator, hex: []const u8) ![]u8 {
    const n = hex.len / 2;
    const buf = try allocator.alloc(u8, n);
    lib.hexToBytes(hex, buf);
    return buf;
}

pub fn compareDigestHex(digest: []const u8, hex: []const u8) bool {
    var tmp: [64]u8 = undefined;
    const n = @min(@min(digest.len, hex.len / 2), tmp.len);
    lib.hexToBytes(hex, tmp[0..n]);
    return std.mem.eql(u8, digest[0..n], tmp[0..n]);
}

pub fn bytesToHex(bytes: []const u8, out: []u8) []u8 {
    const hexchars = "0123456789abcdef";
    var i: usize = 0;
    while (i < bytes.len) : (i += 1) {
        out[i * 2] = hexchars[bytes[i] >> 4];
        out[i * 2 + 1] = hexchars[bytes[i] & 0xf];
    }
    return out[0 .. bytes.len * 2];
}

fn finishResult(allocator: std.mem.Allocator, sh: *Shared) CrackResult {
    const attempts = sh.attempts.load(.acquire);
    if (!sh.result_set) return .{ .password = null, .attempts = attempts };
    const dup = allocator.dupe(u8, sh.result_buf[0..sh.result_len]) catch
        return .{ .password = null, .attempts = attempts };
    return .{ .password = dup, .attempts = attempts };
}

pub fn bruteForce(allocator: std.mem.Allocator, opts: CrackOptions) !CrackResult {
    const passmax: u32 = if (opts.passmax == 0) MAX_DEFAULT else opts.passmax;
    if (passmax == 0 or passmax < opts.passmin) return .{ .password = null, .attempts = 0 };

    const alphabet = try prepareDictionary(allocator, opts.dict);
    if (alphabet.len == 0) return .{ .password = null, .attempts = 0 };

    var num_threads = if (opts.num_threads == 0) lib.getProcessorCount() else opts.num_threads;
    if (num_threads == 0) num_threads = 1;
    if (alphabet.len < num_threads) num_threads = @intCast(alphabet.len);
    if (num_threads == 0) num_threads = 1;

    const target = try createDigest(allocator, opts.hash);
    const result_buf = try allocator.alloc(u8, passmax);

    var sh: Shared = .{
        .dict = alphabet,
        .target = target,
        .hash_len = @min(opts.hash_def.hash_length, target.len),
        .digest_fn = opts.hash_def.digest,
        .passmin = opts.passmin,
        .passmax = passmax,
        .use_wide = opts.use_wide_pass,
        .found = std.atomic.Value(bool).init(false),
        .attempts = std.atomic.Value(u64).init(0),
        .result_buf = result_buf,
        .result_len = 0,
        .result_set = false,
    };

    var main_scratch: WorkerScratch = .{
        .indices = try allocator.alloc(usize, passmax),
        .bytes = try allocBytes(allocator, passmax, opts.use_wide_pass),
    };

    if (num_threads == 1 or builtin.single_threaded) {
        worker(&sh, 1, 0, &main_scratch);
        return finishResult(allocator, &sh);
    }

    const threads = try allocator.alloc(std.Thread, num_threads);
    const scratches = try allocator.alloc(WorkerScratch, num_threads);
    for (scratches) |*s| {
        s.* = .{
            .indices = try allocator.alloc(usize, passmax),
            .bytes = try allocBytes(allocator, passmax, opts.use_wide_pass),
        };
    }

    var spawned: u32 = 0;
    var spawn_failed = false;
    for (0..num_threads) |i| {
        threads[i] = std.Thread.spawn(.{}, worker, .{ &sh, num_threads, @as(u32, @intCast(i)), &scratches[i] }) catch {
            spawn_failed = true;
            break;
        };
        spawned += 1;
    }

    if (spawn_failed) sh.found.store(true, .release);
    {
        var j: u32 = 0;
        while (j < spawned) : (j += 1) threads[j].join();
    }
    if (spawn_failed) {
        sh.found.store(false, .release);
        worker(&sh, 1, 0, &main_scratch);
    }

    return finishResult(allocator, &sh);
}

pub fn outputTimings(writer: *std.Io.Writer, attempts: u64, time: lib.Time) !void {
    const speed: f64 = if (time.total_seconds > 0)
        @as(f64, @floatFromInt(attempts)) / time.total_seconds
    else
        0;

    var tbuf: [96]u8 = undefined;
    var tw: std.Io.Writer = .fixed(&tbuf);
    try lib.formatTime(time, &tw);

    try writer.writeAll("\n");
    try writer.print("Attempts: {d} Time {s} Speed: {d:.0} attempts/second\n", .{
        attempts,
        std.Io.Writer.buffered(&tw),
        speed,
    });
}

/// High-level orchestration mirroring C's bf_crack_hash: optional probe
/// (estimate), main run, timings and result line. GPU is a no-op stub.
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
    _ = has_gpu; // GPU stub: CPU-only for now (TODO Task 9 / CUDA)

    const passmax: u32 = if (passmax_in == 0) MAX_DEFAULT else passmax_in;
    const digest_buf = try allocator.alloc(u8, hash_def.hash_length);

    hash_def.digest(digest_buf.ptr, "".ptr, 0);
    if (compareDigestHex(digest_buf, hash)) {
        lib.startTimer();
        lib.stopTimer();
        try writer.print("Initial string is: Empty string\n", .{});
        return .{ .password = try allocator.dupe(u8, ""), .attempts = 0 };
    }

    if (!no_probe) {
        const probe_str = "123";
        hash_def.digest(digest_buf.ptr, probe_str.ptr, probe_str.len);
        var hexbuf: [128]u8 = undefined;
        const probe_hex = bytesToHex(digest_buf[0..hash_def.hash_length], &hexbuf);

        lib.startTimer();
        const probe_res = try bruteForce(allocator, .{
            .passmin = 1,
            .passmax = MAX_DEFAULT,
            .dict = default_alphabet,
            .hash = probe_hex,
            .hash_def = hash_def,
            .num_threads = num_threads,
            .use_wide_pass = use_wide,
        });
        lib.stopTimer();
        const probe_time = lib.readElapsedTime();

        const prepared = try prepareDictionary(allocator, dict);
        const ratio: f64 = if (probe_time.total_seconds > 0)
            @as(f64, @floatFromInt(probe_res.attempts)) / probe_time.total_seconds
        else
            0;
        const max_attempts = std.math.pow(f64, @as(f64, @floatFromInt(prepared.len)), @as(f64, @floatFromInt(passmax)));
        const est = if (ratio > 0) lib.normalizeTime(max_attempts / ratio) else lib.normalizeTime(0);
        var ebuf: [96]u8 = undefined;
        var ew: std.Io.Writer = .fixed(&ebuf);
        try lib.formatTime(est, &ew);
        try writer.print("May take approximately: {s} ({d:.0} attempts)\n", .{ std.Io.Writer.buffered(&ew), max_attempts });
    }

    lib.startTimer();
    const res = try bruteForce(allocator, .{
        .passmin = passmin,
        .passmax = passmax,
        .dict = dict,
        .hash = hash,
        .hash_def = hash_def,
        .num_threads = num_threads,
        .use_wide_pass = use_wide,
    });
    lib.stopTimer();

    try outputTimings(writer, res.attempts, lib.readElapsedTime());
    if (res.password) |p| {
        try writer.print("Initial string is: {s}\n", .{p});
    } else {
        try writer.print("Nothing found\n", .{});
    }
    return res;
}

test "prepareDictionary templates" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    try std.testing.expectEqualStrings("0123456789", try prepareDictionary(a, "0-9"));
    try std.testing.expectEqualStrings(LOW_CASE, try prepareDictionary(a, "a-z"));
    try std.testing.expectEqualStrings(UPPER_CASE, try prepareDictionary(a, "A-Z"));

    const combo = try prepareDictionary(a, "0-9a-zA-Z");
    try std.testing.expectEqual(@as(usize, 62), combo.len);

    const asc = try prepareDictionary(a, "ASCII");
    try std.testing.expectEqual(@as(usize, 94), asc.len);
    try std.testing.expectEqual(@as(u8, '!'), asc[0]);
    try std.testing.expectEqual(@as(u8, '~'), asc[asc.len - 1]);

    try std.testing.expectEqualStrings("abc", try prepareDictionary(a, "abcaabbc"));
}

test "createDigest and compareDigestHex" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const bytes = try createDigest(a, "deadbeef");
    try std.testing.expectEqualSlices(u8, &.{ 0xde, 0xad, 0xbe, 0xef }, bytes);
    try std.testing.expect(compareDigestHex(&.{ 0xde, 0xad, 0xbe, 0xef }, "deadbeef"));
    try std.testing.expect(!compareDigestHex(&.{ 0xde, 0xad, 0xbe, 0xee }, "deadbeef"));
}

test "bruteForce tiger recovers abc (single thread)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tiger = hashes.getHash("tiger").?;
    var target: [24]u8 align(8) = undefined;
    hashes.compute(tiger, "abc", &target);
    var hexbuf: [48]u8 = undefined;
    const hex = bytesToHex(&target, &hexbuf);

    const res = try bruteForce(a, .{
        .passmin = 3,
        .passmax = 3,
        .dict = "abcd",
        .hash = hex,
        .hash_def = tiger,
        .num_threads = 1,
    });
    try std.testing.expect(res.password != null);
    try std.testing.expectEqualStrings("abc", res.password.?);
}

test "bruteForce tiger recovers abc (multi thread, strided)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tiger = hashes.getHash("tiger").?;
    var target: [24]u8 align(8) = undefined;
    hashes.compute(tiger, "abc", &target);
    var hexbuf: [48]u8 = undefined;
    const hex = bytesToHex(&target, &hexbuf);

    const res = try bruteForce(a, .{
        .passmin = 3,
        .passmax = 3,
        .dict = "abcd",
        .hash = hex,
        .hash_def = tiger,
        .num_threads = 4,
    });
    try std.testing.expectEqualStrings("abc", res.password orelse "");
}

test "bruteForce digits multi-length recovers 1234" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const rmd = hashes.getHash("ripemd160").?;
    var target: [20]u8 align(8) = undefined;
    hashes.compute(rmd, "1234", &target);
    var hexbuf: [40]u8 = undefined;
    const hex = bytesToHex(&target, &hexbuf);

    const res = try bruteForce(a, .{
        .passmin = 1,
        .passmax = 4,
        .dict = "0-9",
        .hash = hex,
        .hash_def = rmd,
        .num_threads = 0,
    });
    try std.testing.expectEqualStrings("1234", res.password orelse "");
}

test "bruteForce not found returns null" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tiger = hashes.getHash("tiger").?;
    const res = try bruteForce(a, .{
        .passmin = 1,
        .passmax = 2,
        .dict = "ab",
        .hash = "000000000000000000000000000000000000000000000000",
        .hash_def = tiger,
        .num_threads = 2,
    });
    try std.testing.expect(res.password == null);
}

test "bruteForce wide (UTF-16) recovers abc" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tiger = hashes.getHash("tiger").?;
    const wide = [_]u8{ 'a', 0, 'b', 0, 'c', 0 };
    var target: [24]u8 align(8) = undefined;
    hashes.compute(tiger, &wide, &target);
    var hexbuf: [48]u8 = undefined;
    const hex = bytesToHex(&target, &hexbuf);

    const res = try bruteForce(a, .{
        .passmin = 3,
        .passmax = 3,
        .dict = "abcd",
        .hash = hex,
        .hash_def = tiger,
        .num_threads = 2,
        .use_wide_pass = true,
    });
    try std.testing.expectEqualStrings("abc", res.password orelse "");
}
