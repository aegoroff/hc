const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = resolveTarget(b);
    const optimize = b.standardOptimizeOption(.{});
    const strip = optimize != .Debug;

    const crypto_lib = addCryptoLib(b, target, optimize);
    const bf_lib = addBfLib(b, target, optimize);

    const yazap = b.dependency("yazap", .{});

    const version_opt = b.option([]const u8, "version", "Application version") orelse "6.0.0";
    // CUDA / OpenCL: native Windows and Linux gnu only. musl / macOS / cross-arch
    // use the CPU stub — toolkit artefacts match the host, not a cross triple.
    const cuda_opt = b.option(bool, "cuda", "Link CUDA when nvcc is available (native Windows / Linux gnu only)");
    const opencl_opt = b.option(bool, "opencl", "Enable OpenCL GPU backend (Linux gnu / Windows; may combine with CUDA)");
    const gpu_eligible = targetSupportsGpuBackend(target);
    warnGpuFlagIfIgnored(target, cuda_opt, gpu_eligible, "cuda");
    warnGpuFlagIfIgnored(target, opencl_opt, gpu_eligible, "opencl");
    // Defaults: on when eligible. -Dcuda=false / -Dopencl=false opts out.
    const want_cuda = gpuBackendWanted(cuda_opt, gpu_eligible);
    const want_opencl = gpuBackendWanted(opencl_opt, gpu_eligible);
    const enable_opencl = want_opencl;
    const enable_cuda = want_cuda and nvccAvailable(b);
    // Missing nvcc: Windows hard-fails only when OpenCL is also off (release
    // binaries should keep a GPU path). Elsewhere warn and fall back.
    if (want_cuda and !enable_cuda) reportMissingNvcc(target, enable_opencl);

    const options = b.addOptions();
    options.addOption([]const u8, "version", version_opt);
    options.addOption(bool, "enable_cuda", enable_cuda);
    options.addOption(bool, "enable_opencl", enable_opencl);
    const build_options_mod = options.createModule();

    const gpu_lib = addGpuLib(b, target, optimize, enable_cuda, enable_opencl);

    // C headers → Zig modules via addTranslateC (replaces deprecated @cImport).
    // Pattern mirrors grok / l2h: umbrella .h + include paths + defineCMacro.
    const translate_hashes = b.addTranslateC(.{
        .root_source_file = b.path("src/hc/hashes_c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_hashes.addIncludePath(b.path("src/srclib"));
    translate_hashes.addIncludePath(b.path(opensslPath(b, target, "include")));
    translate_hashes.defineCMacro("OPENSSL_API_COMPAT", "0x10100000L");
    const hashes_c_mod = translate_hashes.createModule();

    const translate_ltc = b.addTranslateC(.{
        .root_source_file = b.path("src/hc/ltc_c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_ltc.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    const ltc_c_mod = translate_ltc.createModule();

    const translate_bf = b.addTranslateC(.{
        .root_source_file = b.path("src/hc/bf_c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_bf.addIncludePath(b.path("src/hc"));
    translate_bf.addIncludePath(b.path("src/abi"));
    const bf_c_mod = translate_bf.createModule();

    // Canonical GPU ABI + per-algorithm CUDA/stub entry points, surfaced to
    // gpu.zig so the Zig-side structs/externs mirror a single C source
    // (src/abi/gpu_abi.h) instead of a hand-maintained third copy.
    const translate_gpu = b.addTranslateC(.{
        .root_source_file = b.path("src/hc/gpu_c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_gpu.addIncludePath(b.path("src/abi"));
    translate_gpu.addIncludePath(b.path("src/cuda_include"));
    const gpu_c_mod = translate_gpu.createModule();

    const lib_mod = b.addModule("lib", .{
        .root_source_file = b.path("src/hc/lib.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
    });
    lib_mod.addImport("build_options", build_options_mod);

    const gpu_mod = b.createModule(.{
        .root_source_file = b.path("src/hc/gpu.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    gpu_mod.linkLibrary(gpu_lib);
    gpu_mod.addImport("c", gpu_c_mod);
    gpu_mod.addImport("build_options", build_options_mod);
    if (enable_cuda) attachCudaArchive(b, gpu_mod);
    if (enable_opencl) attachOpenclRuntime(gpu_mod);

    const hashes_mod = b.createModule(.{
        .root_source_file = b.path("src/hc/hashes.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    hashes_mod.linkLibrary(crypto_lib);
    hashes_mod.linkLibrary(gpu_lib);
    linkOpenSslCrypto(b, hashes_mod, target);
    hashes_mod.addImport("c", hashes_c_mod);
    hashes_mod.addImport("ltc", ltc_c_mod);
    hashes_mod.addImport("lib", lib_mod);
    hashes_mod.addImport("gpu", gpu_mod);

    const hashes_tests = b.addTest(.{ .name = "hashes_tests", .root_module = hashes_mod });
    const run_hashes_tests = b.addRunArtifact(hashes_tests);

    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/hc/lib.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    // Reusable bf module so hc and tests can @import("bf") without re-deriving wiring.
    const bf_mod = b.createModule(.{
        .root_source_file = b.path("src/hc/bf.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    bf_mod.linkLibrary(crypto_lib);
    bf_mod.linkLibrary(bf_lib);
    bf_mod.addImport("c", bf_c_mod);
    bf_mod.addImport("lib", lib_mod);
    bf_mod.addImport("hashes", hashes_mod);
    bf_mod.addImport("gpu", gpu_mod);
    bf_mod.linkLibrary(gpu_lib);
    linkUnixLibs(bf_mod, target);

    const bf_tests = b.addTest(.{ .name = "bf_tests", .root_module = bf_mod });
    const run_bf_tests = b.addRunArtifact(bf_tests);

    const modes_mod = b.createModule(.{
        .root_source_file = b.path("src/hc/modes.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    modes_mod.linkLibrary(crypto_lib);
    modes_mod.linkLibrary(gpu_lib);
    modes_mod.addImport("lib", lib_mod);
    modes_mod.addImport("hashes", hashes_mod);
    modes_mod.addImport("bf", bf_mod);

    const modes_tests = b.addTest(.{ .name = "modes_tests", .root_module = modes_mod });
    const run_modes_tests = b.addRunArtifact(modes_tests);

    const test_step = b.step("test", "Run unit tests");

    buildHc(
        b,
        target,
        optimize,
        strip,
        crypto_lib,
        bf_lib,
        gpu_lib,
        lib_mod,
        hashes_mod,
        modes_mod,
        bf_mod,
        gpu_mod,
        yazap,
        build_options_mod,
        test_step,
        enable_cuda,
        enable_opencl,
    );
    buildL2h(b, target, optimize, lib_mod, hashes_mod, modes_mod, yazap, build_options_mod, test_step, enable_cuda);

    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_hashes_tests.step);
    test_step.dependOn(&run_bf_tests.step);
    test_step.dependOn(&run_modes_tests.step);
    const gpu_tests = b.addTest(.{ .root_module = gpu_mod, .name = "gpu_tests" });
    const run_gpu_tests = b.addRunArtifact(gpu_tests);
    test_step.dependOn(&run_gpu_tests.step);

    const hash_gtest_mod = b.createModule(.{
        .root_source_file = b.path("src/tests/hash_test.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    hash_gtest_mod.linkLibrary(crypto_lib);
    hash_gtest_mod.linkLibrary(gpu_lib);
    hash_gtest_mod.addImport("lib", lib_mod);
    hash_gtest_mod.addImport("hashes", hashes_mod);
    hash_gtest_mod.addImport("gpu", gpu_mod);
    const hash_gtest = b.addTest(.{ .name = "hash_gtest", .root_module = hash_gtest_mod });
    const run_hash_gtest = b.addRunArtifact(hash_gtest);
    test_step.dependOn(&run_hash_gtest.step);

    // Brute-force crack matrix (src/tests/brute_force_test.zig).
    // Links bf_core + lib helpers and imports the reusable bf module so its
    // lib/hashes/gpu deps resolve.
    const bf_gtest_mod = b.createModule(.{
        .root_source_file = b.path("src/tests/brute_force_test.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    bf_gtest_mod.linkLibrary(crypto_lib);
    bf_gtest_mod.linkLibrary(bf_lib);
    bf_gtest_mod.linkLibrary(gpu_lib);
    bf_gtest_mod.addImport("lib", lib_mod);
    bf_gtest_mod.addImport("hashes", hashes_mod);
    bf_gtest_mod.addImport("gpu", gpu_mod);
    bf_gtest_mod.addImport("bf", bf_mod);
    linkUnixLibs(bf_gtest_mod, target);
    const bf_gtest = b.addTest(.{ .name = "bf_gtest", .root_module = bf_gtest_mod });
    const run_bf_gtest = b.addRunArtifact(bf_gtest);
    test_step.dependOn(&run_bf_gtest.step);
}

// External C dependency layouts differ by target:
//   Unix: scripts/build_external_libs.sh installs per triple under
//         external_lib/lib/openssl-${arch}-${os}-${abi}/ (lib/libcrypto.a)
//   Windows: scripts/build_external_libs.ps1 -> external_lib/openssl/...

/// Path under the OpenSSL install root for this target. `sub` is a relative
/// suffix (e.g. "include", "lib", "lib/libcrypto.a") joined onto the root.
fn opensslPath(b: *std.Build, target: std.Build.ResolvedTarget, sub: []const u8) []const u8 {
    const t = target.result;
    const root = if (t.os.tag == .windows)
        "external_lib/openssl"
    else blk: {
        const os = switch (t.os.tag) {
            .linux => "linux",
            .macos => "macos",
            .freebsd => "freebsd",
            else => @tagName(t.os.tag),
        };
        break :blk b.fmt("external_lib/lib/openssl-{s}-{s}-{s}", .{
            @tagName(t.cpu.arch),
            os,
            @tagName(t.abi),
        });
    };
    return b.pathJoin(&.{ root, sub });
}

/// pthread/dl/m — needed by bf/hc on non-Windows (OpenSSL asm, math, dlopen).
fn linkUnixLibs(mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    switch (target.result.os.tag) {
        .windows => {},
        .macos => {
            // libSystem provides pthread / dl / m.
        },
        .freebsd => {
            // dl* lives in libc on FreeBSD; no separate -ldl.
            mod.linkSystemLibrary("pthread", .{});
            mod.linkSystemLibrary("m", .{});
        },
        else => {
            mod.linkSystemLibrary("pthread", .{});
            mod.linkSystemLibrary("dl", .{});
            mod.linkSystemLibrary("m", .{});
        },
    }
}

fn linkOpenSslCrypto(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    mod.addLibraryPath(b.path(opensslPath(b, target, "lib")));
    // Prefer the explicit archive so Zig does not pick up a shared system
    // libcrypto. OpenSSL digests (and their asm) come from this static build.
    const lib_name = if (target.result.os.tag == .windows) "libcrypto.lib" else "libcrypto.a";
    mod.addObjectFile(b.path(opensslPath(b, target, b.fmt("lib/{s}", .{lib_name}))));
    switch (target.result.os.tag) {
        // libcrypto.a needs these on ELF (cpuid / threads / dlopen providers).
        // On Darwin they live in libSystem; a separate -ldl is not available.
        // FreeBSD: pthread only — dl* is in libc.
        .linux => {
            mod.linkSystemLibrary("pthread", .{});
            mod.linkSystemLibrary("dl", .{});
        },
        .freebsd => {
            mod.linkSystemLibrary("pthread", .{});
        },
        else => {},
    }
}

// Pin glibc low so release binaries run on common LTS distros (Ubuntu 18.04+
// has 2.27, Debian 10+ has 2.28, RHEL 8+ has 2.17). 2.17 was verified to build
// and run with only GLIBC_2.17 symbols; the earlier 2.38 pin (added to dodge a
// gcc16 SFrame relocation issue that no longer reproduces) needlessly raised
// the runtime floor and broke Ubuntu 22.04 / Debian 12 / RHEL 9.
const pinned_glibc: std.Target.Query.SemanticVersion = .{
    .major = 2,
    .minor = 17,
    .patch = 0,
};

fn materializeHostTriple(query: *std.Target.Query) void {
    if (query.cpu_arch == null) query.cpu_arch = builtin.cpu.arch;
    if (query.os_tag == null) query.os_tag = builtin.target.os.tag;
    if (query.abi == null) query.abi = builtin.target.abi;
}

fn needsHostTripleMaterialization(query: std.Target.Query) bool {
    if (query.cpu_arch != null or query.os_tag != null) return false;
    return switch (query.cpu_model) {
        .native, .explicit => true,
        .baseline, .determined_by_arch_os => false,
    };
}

fn resolveTarget(b: *std.Build) std.Build.ResolvedTarget {
    // Native default: MSVC ABI on Windows (prebuilt COFF .lib artifacts under
    // external_lib/), GNU ABI + pinned glibc 2.17 elsewhere. An explicit
    // -Dtarget=… passed by linux_build.sh / windows_build.ps1 overrides both.
    const default_abi: std.Target.Abi = if (builtin.os.tag == .windows) .msvc else .gnu;
    const default_target: std.Target.Query = .{
        .abi = default_abi,
        // glibc_version is intentionally left null here: pinning it would
        // serialize as `.2.17` into the resolved triple, which is an invalid
        // ABI-version suffix for the MSVC target. The linux+gnu pin is applied
        // conditionally further below.
    };

    var query = b.standardTargetOptionsQueryOnly(.{
        .default_target = default_target,
    });

    if (needsHostTripleMaterialization(query)) {
        materializeHostTriple(&query);
    }

    // `-march=haswell` for x86_64: enables SSE4.2/crc32 used by crc32.c's HW
    // CRC32C path. Only replace the portable baseline default — honor
    // `-Dcpu=…` (e.g. Windows core2 portable builds).
    // aarch64-macos defaults to apple_m1 (Apple Silicon baseline for M1+).
    // aarch64-linux baseline adds +crc (near-universal ARMv8 CRC for crc32c).
    const arch = query.cpu_arch orelse builtin.cpu.arch;
    const os = query.os_tag orelse builtin.target.os.tag;
    if (arch == .x86_64) {
        switch (query.cpu_model) {
            .baseline, .determined_by_arch_os => {
                query.cpu_model = .{ .explicit = &std.Target.x86.cpu.haswell };
            },
            .native, .explicit => {},
        }
    } else if (arch == .aarch64 and os == .macos) {
        switch (query.cpu_model) {
            .baseline, .determined_by_arch_os => {
                query.cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m1 };
            },
            .native, .explicit => {},
        }
    } else if (arch == .aarch64 and os == .linux) {
        switch (query.cpu_model) {
            .baseline, .determined_by_arch_os => {
                query.cpu_features_add = std.Target.aarch64.featureSet(&.{.crc});
            },
            .native, .explicit => {},
        }
    }

    if (query.glibc_version == null and os == .linux) {
        const abi = query.abi orelse builtin.target.abi;
        if (abi.isGnu()) {
            query.glibc_version = pinned_glibc;
        }
    }

    return b.resolveTargetQuery(query);
}

fn addCryptoLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const srclib = "src/srclib";
    const tomcrypt = "src/libtomcrypt";

    const lib = b.addLibrary(.{
        .name = "hc-crypto",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            // Zig defaults can leave -fsanitize-c on C objects even in
            // ReleaseFast; that tanks hash throughput (~3–10×).
            .sanitize_c = .off,
        }),
    });

    const mod = lib.root_module;
    mod.addIncludePath(b.path(srclib));
    mod.addIncludePath(b.path(tomcrypt ++ "/src/headers"));
    mod.addIncludePath(b.path(opensslPath(b, target, "include")));
    mod.addCMacro("BLAKE3_NO_AVX512", "1");
    // Portable Blake3 on non-x86 (no SSE/AVX asm; no NEON kernels linked).
    if (target.result.cpu.arch != .x86_64) {
        mod.addCMacro("BLAKE3_USE_NEON", "0");
    }
    // Allow OpenSSL 3+ deprecated low-level digests (MD5/SHA*/RIPEMD160/WHIRLPOOL).
    mod.addCMacro("OPENSSL_API_COMPAT", "0x10100000L");

    const sph_sources = [_][]const u8{
        "byte_order.c",
        "crc32.c",
        "edonr.c",
        "gost.c",
        "haval.c",
        "md2.c",
        "md4.c",
        "ripemd.c",
        "snefru.c",
        "tiger.c",
        "tiger_sbox.c",
        "rhash_tiger.c",
        "tth.c",
        "blake3.c",
        "blake3_dispatch.c",
        "blake3_portable.c",
    };
    const tomcrypt_sources = [_][]const u8{
        "hashes/rmd128.c",
        "hashes/rmd160.c",
        "hashes/rmd256.c",
        "hashes/rmd320.c",
        "misc/crypt/crypt_argchk.c",
        "misc/zeromem.c",
    };

    const is_x86_64 = target.result.cpu.arch == .x86_64;
    const is_windows = target.result.os.tag == .windows;

    // Build flat source list. b.fmt is runtime (arena-dup'd), so no fixed buffer.
    const n = sph_sources.len + tomcrypt_sources.len;
    const c_sources = b.allocator.alloc([]const u8, n) catch @panic("OOM");
    var ci: usize = 0;
    for (sph_sources) |s| {
        c_sources[ci] = b.fmt("{s}/{s}", .{ srclib, s });
        ci += 1;
    }
    for (tomcrypt_sources) |s| {
        c_sources[ci] = b.fmt("{s}/src/{s}", .{ tomcrypt, s });
        ci += 1;
    }

    // -O3 is not always implied for C objs in every Zig version.
    const flags: []const []const u8 = if (is_windows)
        &.{ "-Wall", "-O3", "-fno-sanitize=undefined", "-DLTC_NO_ROLC" }
    else
        &.{ "-Wall", "-O3", "-fno-sanitize=undefined", "-pthread", "-DLTC_NO_ROLC" };

    mod.addCSourceFiles(.{
        .files = c_sources,
        .flags = flags,
    });

    // Hand-written SIMD kernels (unix gas).
    if (is_x86_64 and !is_windows) {
        const asm_sources = [_][]const u8{
            "blake3_avx2_x86-64_unix.S",
            "blake3_avx512_x86-64_unix.S",
            "blake3_sse2_x86-64_unix.S",
            "blake3_sse41_x86-64_unix.S",
        };
        for (asm_sources) |s| {
            mod.addAssemblyFile(b.path(b.fmt("{s}/{s}", .{ srclib, s })));
        }
    } else if (is_x86_64 and is_windows) {
        // MSVC/COFF target: the unix .S kernels don't assemble, so compile the
        // intrinsic C kernels instead. AVX512 stays off via BLAKE3_NO_AVX512.
        //
        // Only emit kernels whose ISA is in the *module* CPU features: under
        // `-Dcpu=core2`, per-file `-mavx2`/`-msse4.1` do not override `-mcpu
        // core2` on windows-msvc (always_inline / builtin target-feature
        // errors). Haswell keeps avx2+sse41+sse2; core2 keeps sse2 only.
        const feats = target.result.cpu.features;
        const has_avx2 = std.Target.x86.featureSetHas(feats, .avx2);
        const has_sse41 = std.Target.x86.featureSetHas(feats, .sse4_1);
        const has_sse2 = std.Target.x86.featureSetHas(feats, .sse2);
        if (!has_avx2) mod.addCMacro("BLAKE3_NO_AVX2", "1");
        if (!has_sse41) mod.addCMacro("BLAKE3_NO_SSE41", "1");
        if (!has_sse2) mod.addCMacro("BLAKE3_NO_SSE2", "1");

        const simd = [_]struct {
            file: []const u8,
            flags: []const []const u8,
            enabled: bool,
        }{
            .{ .file = "blake3_avx2.c", .flags = &.{ "-O3", "-fno-sanitize=undefined", "-mavx2" }, .enabled = has_avx2 },
            .{ .file = "blake3_sse41.c", .flags = &.{ "-O3", "-fno-sanitize=undefined", "-msse4.1" }, .enabled = has_sse41 },
            .{ .file = "blake3_sse2.c", .flags = &.{ "-O3", "-fno-sanitize=undefined", "-msse2" }, .enabled = has_sse2 },
        };
        for (simd) |e| {
            if (!e.enabled) continue;
            mod.addCSourceFile(.{
                .file = b.path(b.fmt("{s}/{s}", .{ srclib, e.file })),
                .flags = e.flags,
            });
        }
    }

    return lib;
}

/// Pool-free brute-force core (`bf_core.c`) plus Zig-side digest callbacks
/// (`bf_shim.c`). Kept out of `hc-crypto` so targets like `l2h` that already
/// ship a tiny C surface don't collide.
fn addBfLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "hc-bf",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .sanitize_c = .off,
        }),
    });

    const mod = lib.root_module;
    mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    mod.addIncludePath(b.path("src/abi"));
    mod.addIncludePath(b.path("src/hc"));
    mod.addCMacro("LTC_NO_ROLC", "1");

    const sources = [_][]const u8{
        "src/hc/bf_core.c",
        "src/hc/bf_shim.c",
    };

    const is_windows = target.result.os.tag == .windows;
    const flags: []const []const u8 = if (is_windows)
        &.{ "-Wall", "-O3", "-fno-sanitize=undefined" }
    else
        &.{ "-Wall", "-O3", "-fno-sanitize=undefined", "-pthread" };

    mod.addCSourceFiles(.{
        .files = &sources,
        .flags = flags,
    });

    return lib;
}

/// Extra directories (beyond PATH) where `nvcc` may live.
/// Prefers `CUDA_PATH` / `CUDA_HOME`, then NVIDIA versioned vars
/// (`CUDA_PATH_V13_2`, … — often set when `CUDA_PATH` itself is not), then
/// common install locations. PATH is already searched by findProgram before
/// these extras.
fn cudaBinSearchPaths(b: *std.Build) []const []const u8 {
    var buf: [24][]const u8 = undefined;
    var n: usize = 0;

    if (b.graph.environ_map.get("CUDA_PATH")) |cuda_path| {
        buf[n] = b.pathJoin(&.{ cuda_path, "bin" });
        n += 1;
    }
    if (b.graph.environ_map.get("CUDA_HOME")) |cuda_home| {
        buf[n] = b.pathJoin(&.{ cuda_home, "bin" });
        n += 1;
    }

    switch (builtin.os.tag) {
        .linux => {
            if (n < buf.len) {
                buf[n] = "/opt/cuda/bin";
                n += 1;
            }
            if (n < buf.len) {
                buf[n] = "/usr/local/cuda/bin";
                n += 1;
            }
        },
        .windows => {
            // Versioned installer vars (CUDA_PATH_V13_2 etc.). Lower priority than the
            // unversioned CUDA_PATH / CUDA_HOME so an explicit current-toolkit pin wins.
            for (b.graph.environ_map.keys(), b.graph.environ_map.values()) |key, value| {
                if (n >= buf.len) break;
                if (key.len <= "CUDA_PATH_V".len) continue;
                if (!std.mem.startsWith(u8, key, "CUDA_PATH_V")) continue;
                buf[n] = b.pathJoin(&.{ value, "bin" });
                n += 1;
            }
        },
        // macOS (and others): only CUDA_PATH / CUDA_HOME above — target GPU
        // backends are Windows / Linux gnu only (`targetSupportsGpuBackend`).
        else => {},
    }

    return b.allocator.dupe([]const u8, buf[0..n]) catch @panic("OOM");
}

fn cudaLibDirForRoot(b: *std.Build, root: []const u8) []const u8 {
    return switch (builtin.os.tag) {
        .windows => b.pathJoin(&.{ root, "lib", "x64" }),
        else => b.pathJoin(&.{ root, "lib64" }),
    };
}

fn cudaLibSearchPath(b: *std.Build) ?[]const u8 {
    if (b.graph.environ_map.get("CUDA_PATH")) |root| {
        return cudaLibDirForRoot(b, root);
    }
    if (b.graph.environ_map.get("CUDA_HOME")) |root| {
        return cudaLibDirForRoot(b, root);
    }
    // Derive toolkit root from the nvcc we would actually invoke.
    if (b.findProgram(&.{"nvcc"}, cudaBinSearchPaths(b))) |nvcc| {
        const bin_dir = std.fs.path.dirname(nvcc) orelse return null;
        const root = std.fs.path.dirname(bin_dir) orelse return null;
        return cudaLibDirForRoot(b, root);
    } else |_| {}
    return null;
}

fn nvccAvailable(b: *std.Build) bool {
    _ = b.findProgram(&.{"nvcc"}, cudaBinSearchPaths(b)) catch return false;
    return true;
}

/// `true` when the user wants the backend: default on if eligible, `-D*=false` opts out.
fn gpuBackendWanted(opt: ?bool, eligible: bool) bool {
    return (opt orelse true) and eligible;
}

fn warnGpuFlagIfIgnored(
    target: std.Build.ResolvedTarget,
    opt: ?bool,
    eligible: bool,
    comptime flag: []const u8,
) void {
    if (opt != true or eligible) return;
    std.debug.print(
        "\nWARNING: -D{s}=true ignored for {s}-{s}-{s}; {s} is only linked for native Windows and Linux gnu.\n" ++
            "Building without that GPU backend.\n\n",
        .{
            flag,
            @tagName(target.result.cpu.arch),
            @tagName(target.result.os.tag),
            @tagName(target.result.abi),
            flag,
        },
    );
}

fn reportMissingNvcc(target: std.Build.ResolvedTarget, enable_opencl: bool) void {
    if (target.result.os.tag == .windows and !enable_opencl) {
        @panic(
            \\CUDA requested (-Dcuda=true / default) but `nvcc` was not found.
            \\Windows builds require the CUDA toolkit (or OpenCL) for GPU parity with Linux.
            \\Install the toolkit, set CUDA_PATH (or CUDA_PATH_V*), ensure nvcc is
            \\on PATH, or pass -Dcuda=false to opt into the CPU-only stub.
        );
    }
    if (enable_opencl) {
        std.debug.print(
            "\nWARNING: CUDA requested but `nvcc` was not found.\n" ++
                "Building with OpenCL only (install the CUDA toolkit for a CUDA+OpenCL dual binary).\n" ++
                "Install the CUDA toolkit / set CUDA_PATH, or pass -Dcuda=false to silence this.\n\n",
            .{},
        );
    } else {
        std.debug.print(
            "\nWARNING: CUDA requested (-Dcuda=true / default) but `nvcc` was not found.\n" ++
                "Building with the CPU-only GPU stub — GPU-accelerated hashes will be disabled.\n" ++
                "Install the CUDA toolkit / set CUDA_PATH, or pass -Dcuda=false to silence this.\n\n",
            .{},
        );
    }
}

/// CUDA/OpenCL artefacts match the host toolkit. Only native Windows and
/// Linux gnu may link them; musl, macOS, and cross-arch use the CPU stub.
fn targetSupportsGpuBackend(target: std.Build.ResolvedTarget) bool {
    const t = target.result;
    if (t.cpu.arch != builtin.cpu.arch) return false;
    return switch (t.os.tag) {
        .windows => true,
        .linux => t.abi.isGnu(),
        else => false,
    };
}

fn addGpuLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    enable_cuda: bool,
    enable_opencl: bool,
) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "hc-gpu",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    lib.root_module.addIncludePath(b.path("src/abi"));

    if (enable_cuda) {
        // nvcc is guaranteed present (guarded by nvccAvailable at the call site).
        const nvcc = b.findProgram(&.{"nvcc"}, cudaBinSearchPaths(b)) catch
            @panic("nvcc not found despite enable_cuda");
        // abi/: canonical gpu_abi.h. cuda_include/: per-algorithm host decls.
        const inc_abi = b.pathFromRoot("src/abi");
        const inc_cu = b.pathFromRoot("src/cuda_include");
        // Dual build: rename CUDA ABI to cuda_* so OpenCL can keep ocl_* and
        // gpu_dispatch.c owns the public unprefixed symbols.
        const dual = enable_opencl;
        const cuda_prefix = b.pathFromRoot("src/cuda/cuda_prefix.h");

        lib.root_module.addCSourceFile(.{
            .file = b.path("src/hc/gpu_cuda_marker.c"),
            .flags = &.{},
        });

        // Per-file nvcc compilation → host+device objects (cached individually).
        const is_windows = target.result.os.tag == .windows;
        const obj_ext = if (is_windows) "obj" else "o";
        const cu_bases = [_][]const u8{
            "blake2b", "blake2s", "crc32",  "gpu",    "md2",  "md4",    "md5",    "rmd128", "rmd160",    "rmd256",
            "rmd320",  "sha1",    "sha224", "sha256", "sha3", "sha384", "sha512", "tiger",  "whirlpool",
        };
        for (cu_bases) |base| {
            const step = b.addSystemCommand(&.{nvcc});
            step.addArgs(&.{ "-c", "-arch=sm_75", "-std=c++17", "-O2" });
            if (!is_windows) step.addArgs(&.{ "--compiler-options", "-fPIC" });
            if (dual) step.addArgs(&.{ "-include", cuda_prefix });
            step.addArgs(&.{ "-I", inc_abi, "-I", inc_cu, "-o" });
            step.setCwd(b.path("."));
            const obj = step.addOutputFileArg(b.fmt("{s}.{s}", .{ base, obj_ext }));
            step.addFileArg(b.path(b.fmt("src/cuda/{s}.cu", .{base})));
            lib.root_module.addObjectFile(obj);
        }
    }

    if (enable_opencl) {
        // Always compile OpenCL under ocl_* names; public ABI comes from either
        // ocl_shim.c (OpenCL-only) or gpu_dispatch.c (CUDA+OpenCL).
        // Absolute -include path: relative -include triggers Zig CacheCheckFailed.
        const ocl_prefix = b.pathFromRoot("src/opencl/ocl_prefix.h");
        // c23: #embed of kernels/*.cl in ocl_algos.c.
        const ocl_flags = [_][]const u8{ "-std=c23", "-include", ocl_prefix };
        lib.root_module.addIncludePath(b.path("src/cuda_include"));
        lib.root_module.addIncludePath(b.path("src/opencl"));
        if (target.result.os.tag != .windows) {
            lib.root_module.linkSystemLibrary("dl", .{});
        }
        lib.root_module.addCSourceFiles(.{
            .files = &.{
                "src/opencl/ocl_dyn.c",
                "src/opencl/ocl_gpu.c",
                "src/opencl/ocl_common.c",
                "src/opencl/ocl_algos.c",
            },
            .flags = &ocl_flags,
        });
        if (enable_cuda) {
            lib.root_module.addCSourceFile(.{
                .file = b.path("src/hc/gpu_dispatch.c"),
                .flags = &.{"-std=c11"},
            });
        } else {
            lib.root_module.addCSourceFile(.{
                .file = b.path("src/opencl/ocl_shim.c"),
                .flags = &.{"-std=c11"},
            });
        }
    }

    if (!enable_cuda and !enable_opencl) {
        lib.root_module.addCSourceFile(.{
            .file = b.path("src/hc/gpu_stub.c"),
            .flags = &.{},
        });
    }
    return lib;
}

fn attachCudaArchive(b: *std.Build, mod: *std.Build.Module) void {
    // The kernel archive is linked into gpu_lib by addGpuLib; here we only pull
    // in the CUDA runtime + the host-code support libs nvcc objects reference.
    if (cudaLibSearchPath(b)) |lib_dir| {
        mod.addLibraryPath(.{ .cwd_relative = lib_dir });
    }
    // Static CUDA runtime: libcudart_static.a (driver is dlopen'd at runtime,
    // so no libcuda link needed).
    mod.linkSystemLibrary("cudart_static", .{ .preferred_link_mode = .static });

    // Linux nvcc host objects pull in these; Windows uses the MSVC/MinGW runtime.
    if (builtin.os.tag == .linux) {
        mod.linkSystemLibrary("dl", .{});
        mod.linkSystemLibrary("pthread", .{});
        mod.linkSystemLibrary("rt", .{});
    }
    // nvcc host code references the C++ ABI (__cxa_guard_*, _Unwind_Resume, …).
    // Skip on MSVC Windows where the C++ runtime is linked differently.
    const abi = if (mod.resolved_target) |t| t.result.abi else builtin.target.abi;
    if (abi != .msvc) {
        mod.linkSystemLibrary("stdc++", .{});
    }
}

fn attachOpenclRuntime(mod: *std.Build.Module) void {
    // ICD is loaded at runtime (dlopen / LoadLibrary). libdl only on ELF.
    const os = if (mod.resolved_target) |t| t.result.os.tag else builtin.os.tag;
    if (os != .windows) {
        mod.linkSystemLibrary("dl", .{});
    }
}

/// Builds the `hc` executable plus its run/test steps using the shared module
/// graph assembled earlier in `build()`.
fn buildHc(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: bool,
    crypto_lib: *std.Build.Step.Compile,
    bf_lib: *std.Build.Step.Compile,
    gpu_lib: *std.Build.Step.Compile,
    lib_mod: *std.Build.Module,
    hashes_mod: *std.Build.Module,
    modes_mod: *std.Build.Module,
    bf_mod: *std.Build.Module,
    gpu_mod: *std.Build.Module,
    yazap: *std.Build.Dependency,
    build_options_mod: *std.Build.Module,
    test_step: *std.Build.Step,
    enable_cuda: bool,
    enable_opencl: bool,
) void {
    // hc executable: Zig CLI entry point.
    const hc_mod = b.createModule(.{
        .root_source_file = b.path("src/hc/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    hc_mod.linkLibrary(crypto_lib);
    hc_mod.linkLibrary(bf_lib);
    hc_mod.linkLibrary(gpu_lib);
    hc_mod.addImport("lib", lib_mod);
    hc_mod.addImport("hashes", hashes_mod);
    hc_mod.addImport("modes", modes_mod);
    hc_mod.addImport("bf", bf_mod);
    hc_mod.addImport("gpu", gpu_mod);
    hc_mod.addImport("yazap", yazap.module("yazap"));
    hc_mod.addImport("build_options", build_options_mod);
    linkUnixLibs(hc_mod, target);
    if (enable_cuda) attachCudaArchive(b, hc_mod);
    if (enable_opencl) attachOpenclRuntime(hc_mod);

    const hc = b.addExecutable(.{
        .name = "hc",
        .root_module = hc_mod,
    });
    b.installArtifact(hc);

    const run_hc = b.addRunArtifact(hc);
    run_hc.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_hc.addArgs(args);
    const run_hc_step = b.step("run-hc", "Run the hc CLI");
    run_hc_step.dependOn(&run_hc.step);

    const hc_tests = b.addTest(.{ .root_module = hc_mod });
    const run_hc_tests = b.addRunArtifact(hc_tests);
    const hc_test_step = b.step("test-hc", "Run hc unit tests");
    hc_test_step.dependOn(&run_hc_tests.step);
    test_step.dependOn(&run_hc_tests.step);
}

/// Wires the l2h (linq2hash) query frontend: runs flex/bison to generate the
/// parser, compiles the generated C into a static lib, exposes the token table
/// and types to Zig through translate-c, and builds the `l2h` executable.
/// Mirrors the grok build pattern (b.addSystemCommand + addLibrary + addTranslateC).
fn buildL2h(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    lib_mod: *std.Build.Module,
    hashes_mod: *std.Build.Module,
    modes_mod: *std.Build.Module,
    yazap: *std.Build.Dependency,
    build_options_mod: *std.Build.Module,
    test_step: *std.Build.Step,
    enable_cuda: bool,
) void {
    const c_code_path = "src/l2h/grammar";
    // b.fmt is the idiomatic 0.16 helper for build-time strings: it returns an
    // arena-duplicated slice and cannot fail, so flex/bison argv are never the
    // empty string that the previous `allocPrint(...) catch ""` produced on OOM.
    const generated_path = b.fmt("{s}/generated", .{c_code_path});
    // Zig 0.16: createDirPath replaces the old fs.makePath (creates parents).
    std.Io.Dir.cwd().createDirPath(b.graph.io, b.pathFromRoot(generated_path)) catch {};

    const flex_input = b.fmt("{s}/l2h.lex", .{c_code_path});
    const flex_src = b.fmt("{s}/l2h.flex.c", .{generated_path});
    const flex_hdr = b.fmt("{s}/l2h.flex.h", .{generated_path});
    const flex_opt = b.fmt("--outfile={s}", .{flex_src});
    const flex_hdr_opt = b.fmt("--header-file={s}", .{flex_hdr});

    const bison_input = b.fmt("{s}/l2h.y", .{c_code_path});
    const bison_src = b.fmt("{s}/l2h.tab.c", .{generated_path});
    const bison_opt = b.fmt("--output={s}", .{bison_src});

    const c_sources = [_][]const u8{
        flex_src,
        bison_src,
    };

    var flex_args: []const []const u8 = undefined;
    var bison_args: []const []const u8 = undefined;

    switch (builtin.os.tag) {
        .linux => {
            flex_args = &[_][]const u8{ "flex", "--fast", flex_opt, flex_hdr_opt, flex_input };
            bison_args = &[_][]const u8{ "bison", bison_opt, "-dy", "-Wno-yacc", "-Wno-other", bison_input };
        },
        .windows => {
            flex_args = &[_][]const u8{ "flex", "--fast", "--wincompat", flex_opt, flex_hdr_opt, flex_input };
            bison_args = &[_][]const u8{ "bison", bison_opt, "-dy", "-Wno-yacc", "-Wno-other", bison_input };
        },
        .macos => {
            flex_args = &[_][]const u8{ "/usr/local/opt/flex/bin/flex", "--fast", flex_opt, flex_hdr_opt, flex_input };
            bison_args = &[_][]const u8{ "/usr/local/opt/bison/bin/bison", bison_opt, "-dy", "-Wno-yacc", "-Wno-other", bison_input };
        },
        else => @compileError("Unsupported OS for l2h: " ++ @tagName(builtin.os.tag)),
    }

    const flex = b.addSystemCommand(flex_args);
    const bison = b.addSystemCommand(bison_args);
    bison.step.dependOn(&flex.step);

    const l2h_c_lib = b.addLibrary(.{
        .name = "l2h-c",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    l2h_c_lib.root_module.addIncludePath(b.path(c_code_path));
    l2h_c_lib.root_module.addIncludePath(b.path(generated_path));
    l2h_c_lib.root_module.addIncludePath(b.path("src/srclib"));
    // clang under the MSVC target is stricter than gcc on the generated
    // bison/flex C: it errors on bison's const-discard (l2h.tab.c) and warns on
    // flex's POSIX `read()` name (l2h.flex.c, generated even with --wincompat).
    // Suppress both on windows; the unix path keeps the original empty flag set.
    const l2h_c_flags: []const []const u8 = if (target.result.os.tag == .windows)
        &.{ "-Wno-incompatible-pointer-types-discards-qualifiers", "-Wno-deprecated-declarations" }
    else
        &.{};
    l2h_c_lib.root_module.addCSourceFiles(.{ .files = &c_sources, .flags = l2h_c_flags });
    l2h_c_lib.step.dependOn(&bison.step);

    // Surface tokens/YYSTYPE/callback externs to Zig.
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/l2h/c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_c.addIncludePath(b.path(c_code_path));
    translate_c.addIncludePath(b.path(generated_path));
    translate_c.addIncludePath(b.path("src/srclib"));
    translate_c.step.dependOn(&bison.step);

    // PCRE2: MATCH ("~") / NOT MATCH ("!~"). Override gnu's default shared
    // linkage so both gnu and musl get a static archive.
    const pcre2_dep = b.dependency("pcre2", .{
        .target = target,
        .optimize = optimize,
        .linkage = .static,
    });
    const translate_pcre = b.addTranslateC(.{
        .root_source_file = pcre2_dep.namedLazyPath("pcre2.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_pcre.defineCMacro("PCRE2_CODE_UNIT_WIDTH", "8");
    translate_pcre.defineCMacro("PCRE2_STATIC", "");
    translate_pcre.step.dependOn(&pcre2_dep.artifact("pcre2-8").step);

    const fehler_dep = b.dependency("fehler", .{});

    const strip = optimize != .Debug;
    // l2h executable: parser driver (main.zig) + frontend/backend/processor.
    const l2h_mod = b.createModule(.{
        .root_source_file = b.path("src/l2h/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = strip,
    });
    l2h_mod.linkLibrary(l2h_c_lib);
    l2h_mod.addImport("c", translate_c.createModule());
    l2h_mod.addImport("re", translate_pcre.createModule());
    l2h_mod.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    // Shared computation backends (lib / hashes / modes).
    l2h_mod.addImport("lib", lib_mod);
    l2h_mod.addImport("hashes", hashes_mod);
    l2h_mod.addImport("modes", modes_mod);
    l2h_mod.addImport("build_options", build_options_mod);
    l2h_mod.addImport("fehler", fehler_dep.module("fehler"));
    l2h_mod.addImport("yazap", yazap.module("yazap"));
    if (enable_cuda) attachCudaArchive(b, l2h_mod);

    const l2h = b.addExecutable(.{
        .name = "l2h",
        .root_module = l2h_mod,
    });
    b.installArtifact(l2h);

    const run_l2h = b.addRunArtifact(l2h);
    run_l2h.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_l2h.addArgs(args);
    const run_l2h_step = b.step("run-l2h", "Run the l2h query frontend");
    run_l2h_step.dependOn(&run_l2h.step);

    // Unit tests for the Zig-side frontend/backend/processor semantics.
    const l2h_tests = b.addTest(.{ .root_module = l2h_mod });
    const run_l2h_tests = b.addRunArtifact(l2h_tests);
    const l2h_test_step = b.step("test-l2h", "Run l2h unit tests");
    l2h_test_step.dependOn(&run_l2h_tests.step);
    test_step.dependOn(&run_l2h_tests.step);
}
