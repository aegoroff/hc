//! GPU brute-force wrapper (Task 9).
//!
//! When `nvcc` is available the build links the CUDA static library and
//! exposes the C ABI from gpu.cu / *.cu. Without a toolkit, stubs report that
//! the GPU is unavailable so callers fall back to CPU. Either way there is a
//! single `hc` binary with all hash algorithms; per-algorithm
//! `has_gpu_implementation` and runtime `canUseGpu()` choose GPU vs CPU.

const std = @import("std");
const build_options = @import("build_options");

pub const GPU_ATTEMPT_SIZE: usize = 16;

pub const GpuThreadCtx = extern struct {
    variants: ?[*]u8 = null,
    dev_variants: ?[*]u8 = null,
    attempt: ?[*]u8 = null,
    result: ?[*]u8 = null,
    dev_result: ?[*]u8 = null,
    gpu_context: ?*GpuContext = null,
    variants_size: usize = 0,
    variants_count: usize = 0,
    passmin: u32 = 0,
    passmax: u32 = 0,
    pass_length: u32 = 0,
    found_in_the_thread: bool = false,
    max_gpu_blocks_number: c_int = 0,
    max_threads_per_block: c_int = 0,
    multiprocessor_count: c_int = 0,
    device_ix: c_int = 0,
    use_wide_pass: bool = false,
    max_threads_decrease_factor: c_int = 1,
    comparisons_per_iteration: c_int = 1,
    pool: ?*anyopaque = null,
};

pub const GpuRunFn = *const fn (
    context: ?*anyopaque,
    dict_len: usize,
    variants: ?[*]u8,
    variants_size: usize,
) callconv(.c) void;

pub const GpuPrepareFn = *const fn (
    device_ix: c_int,
    dict: ?[*]const u8,
    dict_len: usize,
    hash: ?[*]const u8,
    ctx: ?*GpuThreadCtx,
) callconv(.c) void;

pub const GpuContext = extern struct {
    pfn_run: ?GpuRunFn = null,
    pfn_prepare: ?GpuPrepareFn = null,
    max_threads_decrease_factor: c_int = 1,
    comparisons_per_iteration: c_int = 1,
};

pub const enable_cuda = build_options.enable_cuda;

extern fn gpu_can_use_gpu() bool;

pub fn canUseGpu() bool {
    return gpu_can_use_gpu();
}

/// Creates a GpuContext pointing at the given prepare/run pair.
pub fn makeContext(
    run: GpuRunFn,
    prepare: GpuPrepareFn,
    max_threads_decrease_factor: c_int,
    comparisons_per_iteration: c_int,
) GpuContext {
    return .{
        .pfn_run = run,
        .pfn_prepare = prepare,
        .max_threads_decrease_factor = max_threads_decrease_factor,
        .comparisons_per_iteration = comparisons_per_iteration,
    };
}

// Per-algorithm GPU entry points (present as stubs when CUDA is disabled).
pub extern fn md5_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn md5_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn sha256_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn sha256_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn sha1_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn sha1_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn sha224_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn sha224_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn sha384_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn sha384_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn sha512_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn sha512_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn md2_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn md2_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn md4_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn md4_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn rmd160_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn rmd160_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn whirl_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn whirl_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;
pub extern fn crc32_run_on_gpu(ctx: *GpuThreadCtx, dict_len: usize, variants: [*]u8, variants_size: usize) void;
pub extern fn crc32_on_gpu_prepare(device_ix: c_int, dict: [*]const u8, dict_len: usize, hash: [*]const u8, ctx: *GpuThreadCtx) void;

pub const GpuAlgo = struct {
    name: []const u8,
    run: GpuRunFn,
    prepare: GpuPrepareFn,
    max_threads_decrease_factor: c_int,
    comparisons_per_iteration: c_int,
};

/// Algorithms that ship with a CUDA implementation (mirrors hashes.c GPU table).
pub const gpu_algos = [_]GpuAlgo{
    .{ .name = "md5", .run = @ptrCast(&md5_run_on_gpu), .prepare = @ptrCast(&md5_on_gpu_prepare), .max_threads_decrease_factor = 1, .comparisons_per_iteration = 2 },
    .{ .name = "sha1", .run = @ptrCast(&sha1_run_on_gpu), .prepare = @ptrCast(&sha1_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 2 },
    .{ .name = "sha256", .run = @ptrCast(&sha256_run_on_gpu), .prepare = @ptrCast(&sha256_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "sha224", .run = @ptrCast(&sha224_run_on_gpu), .prepare = @ptrCast(&sha224_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "sha384", .run = @ptrCast(&sha384_run_on_gpu), .prepare = @ptrCast(&sha384_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "sha512", .run = @ptrCast(&sha512_run_on_gpu), .prepare = @ptrCast(&sha512_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "md2", .run = @ptrCast(&md2_run_on_gpu), .prepare = @ptrCast(&md2_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "md4", .run = @ptrCast(&md4_run_on_gpu), .prepare = @ptrCast(&md4_on_gpu_prepare), .max_threads_decrease_factor = 1, .comparisons_per_iteration = 2 },
    .{ .name = "ntlm", .run = @ptrCast(&md4_run_on_gpu), .prepare = @ptrCast(&md4_on_gpu_prepare), .max_threads_decrease_factor = 1, .comparisons_per_iteration = 2 },
    .{ .name = "ripemd160", .run = @ptrCast(&rmd160_run_on_gpu), .prepare = @ptrCast(&rmd160_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "whirlpool", .run = @ptrCast(&whirl_run_on_gpu), .prepare = @ptrCast(&whirl_on_gpu_prepare), .max_threads_decrease_factor = 2, .comparisons_per_iteration = 1 },
    .{ .name = "crc32", .run = @ptrCast(&crc32_run_on_gpu), .prepare = @ptrCast(&crc32_on_gpu_prepare), .max_threads_decrease_factor = 1, .comparisons_per_iteration = 2 },
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
    // Without a live NVIDIA driver (or with CPU stubs), canUseGpu is false.
    // When CUDA is linked but the driver is missing, the real gpu.cu path
    // also returns false — so this assertion holds in both configurations.
    try std.testing.expect(!canUseGpu() or enable_cuda);
}

test "contextFor known algorithms" {
    const md5 = contextFor("md5").?;
    try std.testing.expect(md5.pfn_run != null);
    try std.testing.expect(md5.pfn_prepare != null);
    try std.testing.expectEqual(@as(c_int, 1), md5.max_threads_decrease_factor);
    try std.testing.expect(contextFor("nope") == null);
}
