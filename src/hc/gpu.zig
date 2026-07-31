//! GPU brute-force wrapper.
//!
//! When `nvcc` is available the build links the CUDA static library and
//! exposes the C ABI from gpu.cu / *.cu. Without a toolkit, stubs report that
//! the GPU is unavailable so callers fall back to CPU. Either way there is a
//! single `hc` binary with all hash algorithms; `gpu_algos` /
//! `contextFor` and runtime `gpu_can_use_gpu()` choose GPU vs CPU.
//!
//! The structs (GpuThreadCtx / GpuContext), `GPU_ATTEMPT_SIZE`, and the
//! per-algorithm extern entry points are imported from `c` — the translate-c
//! rendering of the canonical `src/abi/gpu_abi.h` (+ per-algo headers).
//! This keeps a single C definition across the CUDA / stub / Zig domains
//! instead of a hand-maintained third copy here.

const std = @import("std");
const c = @import("c");
const build_options = @import("build_options");

/// Layout and field names mirror `hc_gpu_thread_ctx_t` in gpu_abi.h.
pub const GpuThreadCtx = c.hc_gpu_thread_ctx_t;

/// Layout and field names mirror `hc_gpu_context_t` in gpu_abi.h.
pub const GpuContext = c.hc_gpu_context_t;

pub const GPU_ATTEMPT_SIZE: usize = @intCast(c.GPU_ATTEMPT_SIZE);

/// Convenience aliases for the run/prepare callback shapes declared in
/// gpu_abi.h. Kept as `*const fn (...)` (non-optional) so callers pass real
/// functions; assignment into `GpuContext.pfn_*_` coerces to the optional form.
pub const GpuRunFn = *const fn (
    context: ?*anyopaque,
    dict_len: usize,
    variants: [*c]u8,
    variants_size: usize,
) callconv(.c) void;

pub const GpuPrepareFn = *const fn (
    device_ix: c_int,
    dict: [*c]const u8,
    dict_len: usize,
    hash: [*c]const u8,
    ctx: [*c]GpuThreadCtx,
) callconv(.c) void;

pub const enable_cuda = build_options.enable_cuda;

/// Creates a GpuContext pointing at the given prepare/run pair.
pub fn makeContext(
    run: GpuRunFn,
    prepare: GpuPrepareFn,
    max_threads_decrease_factor: c_int,
    comparisons_per_iteration: c_int,
) GpuContext {
    return .{
        .pfn_run_ = run,
        .pfn_prepare_ = prepare,
        .max_threads_decrease_factor_ = max_threads_decrease_factor,
        .comparisons_per_iteration_ = comparisons_per_iteration,
    };
}

pub const GpuAlgo = struct {
    name: []const u8,
    run: GpuRunFn,
    prepare: GpuPrepareFn,
    max_threads_decrease_factor: c_int,
    comparisons_per_iteration: c_int,
};

/// Algorithms that ship with a CUDA implementation (mirrors hashes.c GPU table).
pub const gpu_algos = [_]GpuAlgo{
    .{ .name = "md5", .run = @ptrCast(&c.md5_run_on_gpu), .prepare = @ptrCast(&c.md5_on_gpu_prepare), .max_threads_decrease_factor = 1, .comparisons_per_iteration = 2 },
    .{ .name = "sha1", .run = @ptrCast(&c.sha1_run_on_gpu), .prepare = @ptrCast(&c.sha1_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 2 },
    .{ .name = "sha256", .run = @ptrCast(&c.sha256_run_on_gpu), .prepare = @ptrCast(&c.sha256_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "sha224", .run = @ptrCast(&c.sha224_run_on_gpu), .prepare = @ptrCast(&c.sha224_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "sha384", .run = @ptrCast(&c.sha384_run_on_gpu), .prepare = @ptrCast(&c.sha384_on_gpu_prepare), .max_threads_decrease_factor = 4, .comparisons_per_iteration = 1 },
    .{ .name = "sha512", .run = @ptrCast(&c.sha512_run_on_gpu), .prepare = @ptrCast(&c.sha512_on_gpu_prepare), .max_threads_decrease_factor = 4, .comparisons_per_iteration = 1 },
    .{ .name = "md2", .run = @ptrCast(&c.md2_run_on_gpu), .prepare = @ptrCast(&c.md2_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "md4", .run = @ptrCast(&c.md4_run_on_gpu), .prepare = @ptrCast(&c.md4_on_gpu_prepare), .max_threads_decrease_factor = 1, .comparisons_per_iteration = 2 },
    .{ .name = "ntlm", .run = @ptrCast(&c.md4_run_on_gpu), .prepare = @ptrCast(&c.md4_on_gpu_prepare), .max_threads_decrease_factor = 1, .comparisons_per_iteration = 2 },
    .{ .name = "ripemd160", .run = @ptrCast(&c.rmd160_run_on_gpu), .prepare = @ptrCast(&c.rmd160_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "whirlpool", .run = @ptrCast(&c.whirl_run_on_gpu), .prepare = @ptrCast(&c.whirl_on_gpu_prepare), .max_threads_decrease_factor = 4, .comparisons_per_iteration = 1 },
    .{ .name = "crc32", .run = @ptrCast(&c.crc32_run_on_gpu), .prepare = @ptrCast(&c.crc32_on_gpu_prepare), .max_threads_decrease_factor = 1, .comparisons_per_iteration = 2 },
};

pub fn lookupAlgo(name: []const u8) ?GpuAlgo {
    for (gpu_algos) |a| {
        if (std.ascii.eqlIgnoreCase(a.name, name)) return a;
    }
    return null;
}

pub fn contextFor(name: []const u8) ?GpuContext {
    const a = lookupAlgo(name) orelse return null;
    return makeContext(a.run, a.prepare, a.max_threads_decrease_factor, a.comparisons_per_iteration);
}

test "gpu stubs report unavailable without driver" {
    // Without a live NVIDIA driver (or with CPU stubs), gpu_can_use_gpu is false.
    // When CUDA is linked but the driver is missing, the real gpu.cu path
    // also returns false — so this assertion holds in both configurations.
    try std.testing.expect(!c.gpu_can_use_gpu() or enable_cuda);
}

test "contextFor known algorithms" {
    const md5 = contextFor("md5").?;
    try std.testing.expect(md5.pfn_run_ != null);
    try std.testing.expect(md5.pfn_prepare_ != null);
    try std.testing.expectEqual(@as(c_int, 1), md5.max_threads_decrease_factor_);
    try std.testing.expect(contextFor("nope") == null);
}
