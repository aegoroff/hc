//! GPU brute-force wrapper.
//!
//! When `nvcc` is available the build links the CUDA static library and
//! exposes the C ABI from gpu.cu / *.cu. Without a toolkit, stubs report that
//! the GPU is unavailable so callers fall back to CPU. Either way there is a
//! single `hc` binary with all hash algorithms; `contextFor` and
//! runtime `gpu_can_use_gpu()` choose GPU vs CPU.
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

/// Callback shapes from gpu_abi.h. Non-optional `*const fn` so table entries
/// pass real functions; assignment into `GpuContext.pfn_*_` coerces to optional.
const GpuRunFn = *const fn (
    context: ?*anyopaque,
    dict_len: usize,
) callconv(.c) void;

const GpuPrepareFn = *const fn (
    device_ix: c_int,
    dict: [*c]const u8,
    dict_len: usize,
    hash: [*c]const u8,
    ctx: [*c]GpuThreadCtx,
) callconv(.c) void;

pub const enable_cuda = build_options.enable_cuda;
pub const enable_opencl = build_options.enable_opencl;

const GpuAlgoEntry = struct {
    name: []const u8,
    ctx: GpuContext,
};

fn gpuEntry(
    name: []const u8,
    run: GpuRunFn,
    prepare: GpuPrepareFn,
    max_threads_decrease_factor: c_int,
) GpuAlgoEntry {
    return .{
        .name = name,
        .ctx = .{
            .pfn_run_ = run,
            .pfn_prepare_ = prepare,
            .max_threads_decrease_factor_ = max_threads_decrease_factor,
        },
    };
}

/// Algorithms that ship with a CUDA / OpenCL implementation.
const gpu_algos = [_]GpuAlgoEntry{
    // Kernels expand 2 suffix chars per prefix (plen=3 covers lengths 4 and 5).
    // `factor` lowers max threads/block for register-heavy algos (CUDA-safe).
    gpuEntry("md5", @ptrCast(&c.md5_run_on_gpu), @ptrCast(&c.md5_on_gpu_prepare), 1),
    gpuEntry("sha1", @ptrCast(&c.sha1_run_on_gpu), @ptrCast(&c.sha1_on_gpu_prepare), 1),
    // factor 2: CUDA sha224/256 hit LaunchOutOfResources at factor 1 (full WG).
    // OpenCL short-pack kernels could use factor 1; shared table stays CUDA-safe.
    gpuEntry("sha256", @ptrCast(&c.sha256_run_on_gpu), @ptrCast(&c.sha256_on_gpu_prepare), 2),
    gpuEntry("sha224", @ptrCast(&c.sha224_run_on_gpu), @ptrCast(&c.sha224_on_gpu_prepare), 2),
    gpuEntry("sha-3-224", @ptrCast(&c.sha3_224_run_on_gpu), @ptrCast(&c.sha3_224_on_gpu_prepare), 4),
    gpuEntry("sha-3-256", @ptrCast(&c.sha3_256_run_on_gpu), @ptrCast(&c.sha3_256_on_gpu_prepare), 4),
    gpuEntry("sha-3-384", @ptrCast(&c.sha3_384_run_on_gpu), @ptrCast(&c.sha3_384_on_gpu_prepare), 4),
    gpuEntry("sha-3-512", @ptrCast(&c.sha3_512_run_on_gpu), @ptrCast(&c.sha3_512_on_gpu_prepare), 4),
    gpuEntry("sha-3k-224", @ptrCast(&c.keccak_224_run_on_gpu), @ptrCast(&c.keccak_224_on_gpu_prepare), 4),
    gpuEntry("sha-3k-256", @ptrCast(&c.keccak_256_run_on_gpu), @ptrCast(&c.keccak_256_on_gpu_prepare), 4),
    gpuEntry("sha-3k-384", @ptrCast(&c.keccak_384_run_on_gpu), @ptrCast(&c.keccak_384_on_gpu_prepare), 4),
    gpuEntry("sha-3k-512", @ptrCast(&c.keccak_512_run_on_gpu), @ptrCast(&c.keccak_512_on_gpu_prepare), 4),
    // factor 4: CUDA sha512 OOMs registers at factor 2; sha384 kept in lockstep.
    gpuEntry("sha384", @ptrCast(&c.sha384_run_on_gpu), @ptrCast(&c.sha384_on_gpu_prepare), 4),
    gpuEntry("sha512", @ptrCast(&c.sha512_run_on_gpu), @ptrCast(&c.sha512_on_gpu_prepare), 4),
    gpuEntry("md2", @ptrCast(&c.md2_run_on_gpu), @ptrCast(&c.md2_on_gpu_prepare), 4),
    gpuEntry("md4", @ptrCast(&c.md4_run_on_gpu), @ptrCast(&c.md4_on_gpu_prepare), 1),
    gpuEntry("ntlm", @ptrCast(&c.md4_run_on_gpu), @ptrCast(&c.md4_on_gpu_prepare), 1),
    gpuEntry("ripemd128", @ptrCast(&c.rmd128_run_on_gpu), @ptrCast(&c.rmd128_on_gpu_prepare), 4),
    gpuEntry("ripemd160", @ptrCast(&c.rmd160_run_on_gpu), @ptrCast(&c.rmd160_on_gpu_prepare), 4),
    gpuEntry("ripemd256", @ptrCast(&c.rmd256_run_on_gpu), @ptrCast(&c.rmd256_on_gpu_prepare), 4),
    gpuEntry("ripemd320", @ptrCast(&c.rmd320_run_on_gpu), @ptrCast(&c.rmd320_on_gpu_prepare), 4),
    gpuEntry("blake2s", @ptrCast(&c.blake2s_run_on_gpu), @ptrCast(&c.blake2s_on_gpu_prepare), 4),
    gpuEntry("blake2b", @ptrCast(&c.blake2b_run_on_gpu), @ptrCast(&c.blake2b_on_gpu_prepare), 4),
    gpuEntry("blake3", @ptrCast(&c.blake3_run_on_gpu), @ptrCast(&c.blake3_on_gpu_prepare), 4),
    // S-boxes in __local/shared; 2-char expand amortizes the table load.
    gpuEntry("tiger", @ptrCast(&c.tiger_run_on_gpu), @ptrCast(&c.tiger_on_gpu_prepare), 2),
    gpuEntry("tiger2", @ptrCast(&c.tiger2_run_on_gpu), @ptrCast(&c.tiger2_on_gpu_prepare), 2),
    gpuEntry("whirlpool", @ptrCast(&c.whirl_run_on_gpu), @ptrCast(&c.whirl_on_gpu_prepare), 2),
    gpuEntry("crc32", @ptrCast(&c.crc32_run_on_gpu), @ptrCast(&c.crc32_on_gpu_prepare), 1),
};

pub fn contextFor(name: []const u8) ?GpuContext {
    for (gpu_algos) |a| {
        if (std.ascii.eqlIgnoreCase(a.name, name)) return a.ctx;
    }
    return null;
}

test "gpu stubs report unavailable without driver" {
    // Without a live GPU runtime (or with CPU stubs), gpu_can_use_gpu is false.
    // CUDA / OpenCL builds may still report true when a device is present.
    try std.testing.expect(!c.gpu_can_use_gpu() or enable_cuda or enable_opencl);
}

test "contextFor known algorithms" {
    const md5 = contextFor("md5").?;
    try std.testing.expect(md5.pfn_run_ != null);
    try std.testing.expect(md5.pfn_prepare_ != null);
    try std.testing.expectEqual(@as(c_int, 1), md5.max_threads_decrease_factor_);
    try std.testing.expect(contextFor("ripemd128") != null);
    try std.testing.expectEqual(@as(c_int, 4), contextFor("ripemd128").?.max_threads_decrease_factor_);
    try std.testing.expect(contextFor("ripemd256") != null);
    try std.testing.expect(contextFor("ripemd320") != null);
    try std.testing.expect(contextFor("blake2s") != null);
    try std.testing.expect(contextFor("blake2b") != null);
    try std.testing.expectEqual(@as(c_int, 4), contextFor("blake2b").?.max_threads_decrease_factor_);
    try std.testing.expect(contextFor("blake3") != null);
    try std.testing.expectEqual(@as(c_int, 4), contextFor("blake3").?.max_threads_decrease_factor_);
    try std.testing.expect(contextFor("sha-3-256") != null);
    try std.testing.expectEqual(@as(c_int, 4), contextFor("sha-3-256").?.max_threads_decrease_factor_);
    try std.testing.expect(contextFor("sha-3k-256") != null);
    try std.testing.expect(contextFor("tiger") != null);
    try std.testing.expectEqual(@as(c_int, 2), contextFor("tiger").?.max_threads_decrease_factor_);
    try std.testing.expect(contextFor("whirlpool") != null);
}
