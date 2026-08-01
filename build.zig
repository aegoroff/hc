const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = resolveTarget(b);
    const optimize = b.standardOptimizeOption(.{});
    const strip = optimize != .Debug;

    const arch_name = archName(target.result.cpu.arch);
    const crypto_lib = addCryptoLib(b, target, optimize, arch_name);
    const bf_lib = addBfLib(b, target, optimize, arch_name);

    const yazap = b.dependency("yazap", .{});

    const version_opt = b.option([]const u8, "version", "Application version") orelse "6.0.0";
    // CUDA only for native Windows and Linux gnu (host arch). musl / macOS /
    // cross-arch targets always use the CPU stub — nvcc objects and
    // libcudart_static match the host toolkit, not the cross triple.
    const cuda_opt = b.option(bool, "cuda", "Link CUDA when nvcc is available (native Windows / Linux gnu only)");
    const cuda_eligible = targetSupportsCuda(target);
    if (cuda_opt == true and !cuda_eligible) {
        std.debug.print(
            "\nWARNING: -Dcuda=true ignored for {s}-{s}-{s}; CUDA is only linked for native Windows and Linux gnu.\n" ++
                "Building with the CPU-only GPU stub.\n\n",
            .{
                @tagName(target.result.cpu.arch),
                @tagName(target.result.os.tag),
                @tagName(target.result.abi),
            },
        );
    }
    const want_cuda = (cuda_opt orelse true) and cuda_eligible;
    const enable_cuda = want_cuda and nvccAvailable(b);
    // Missing toolkit: Windows hard-fails (release binaries must ship GPU kernels;
    // silent stub would break parity with Linux gnu). Elsewhere warn and fall back
    // to the CPU stub. -Dcuda=false (musl/tooling/cross) opts out of both paths.
    if (want_cuda and !enable_cuda) {
        if (target.result.os.tag == .windows) {
            @panic(
                \\CUDA requested (-Dcuda=true / default) but `nvcc` was not found.
                \\Windows builds require the CUDA toolkit for GPU parity with Linux.
                \\Install the toolkit, set CUDA_PATH (or CUDA_PATH_V*), ensure nvcc is
                \\on PATH, or pass -Dcuda=false to opt into the CPU-only stub.
            );
        }
        std.debug.print(
            "\nWARNING: CUDA requested (-Dcuda=true / default) but `nvcc` was not found.\n" ++
                "Building with the CPU-only GPU stub — GPU-accelerated hashes will be disabled.\n" ++
                "Install the CUDA toolkit / set CUDA_PATH, or pass -Dcuda=false to silence this.\n\n",
            .{},
        );
    }

    const options = b.addOptions();
    options.addOption([]const u8, "version", version_opt);
    options.addOption(bool, "enable_cuda", enable_cuda);
    const build_options_mod = options.createModule();

    const gpu_lib = addGpuLib(b, target, optimize, enable_cuda);

    // C headers → Zig modules via addTranslateC (replaces deprecated @cImport).
    // Pattern mirrors grok / l2h: umbrella .h + include paths + defineCMacro.
    const translate_hashes = b.addTranslateC(.{
        .root_source_file = b.path("src/hc/hashes_c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_hashes.addIncludePath(b.path("src/srclib"));
    translate_hashes.addIncludePath(b.path(opensslIncludeRel(b, target)));
    translate_hashes.defineCMacro("USE_KECCAK", "1");
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
    translate_bf.addIncludePath(b.path("src/srclib"));
    translate_bf.addIncludePath(b.path("src/hc"));
    translate_bf.addIncludePath(b.path("src/abi"));
    translate_bf.defineCMacro("ARCH", arch_name);
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
    if (builtin.os.tag != .windows) {
        bf_mod.linkSystemLibrary("pthread", .{});
        bf_mod.linkSystemLibrary("dl", .{});
        bf_mod.linkSystemLibrary("m", .{});
    }

    const bf_tests = b.addTest(.{ .name = "bf_tests", .root_module = bf_mod });
    const run_bf_tests = b.addRunArtifact(bf_tests);

    const modes_mod = b.createModule(.{
        .root_source_file = b.path("src/hc/modes.zig"),
        .target = target,
        .optimize = optimize,
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
    );
    buildL2h(b, target, optimize, lib_mod, hashes_mod, modes_mod, build_options_mod, test_step, enable_cuda);

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

    // GoogleTest BruteForceTest parity (src/tests/brute_force_test.zig).
    // Mirrors the bf module wiring: links bf_core + lib helpers and imports
    // the reusable bf module so its lib/hashes/gpu deps resolve.
    const bf_gtest_mod = b.createModule(.{
        .root_source_file = b.path("src/tests/brute_force_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    bf_gtest_mod.linkLibrary(crypto_lib);
    bf_gtest_mod.linkLibrary(bf_lib);
    bf_gtest_mod.linkLibrary(gpu_lib);
    bf_gtest_mod.addImport("lib", lib_mod);
    bf_gtest_mod.addImport("hashes", hashes_mod);
    bf_gtest_mod.addImport("gpu", gpu_mod);
    bf_gtest_mod.addImport("bf", bf_mod);
    if (builtin.os.tag != .windows) {
        bf_gtest_mod.linkSystemLibrary("pthread", .{});
        bf_gtest_mod.linkSystemLibrary("dl", .{});
        bf_gtest_mod.linkSystemLibrary("m", .{});
    }
    const bf_gtest = b.addTest(.{ .name = "bf_gtest", .root_module = bf_gtest_mod });
    const run_bf_gtest = b.addRunArtifact(bf_gtest);
    test_step.dependOn(&run_bf_gtest.step);
}

fn archName(arch: std.Target.Cpu.Arch) []const u8 {
    return switch (arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        .x86 => "i386",
        else => "unknown",
    };
}

// External C dependency layouts differ by target:
//   Unix: scripts/build_external_libs.sh installs per triple under
//         external_lib/lib/openssl-${arch}-${os}-${abi}/ (lib/libcrypto.a)
//   Windows: scripts/build_external_libs.ps1 -> external_lib/openssl/...

fn opensslOsName(os_tag: std.Target.Os.Tag) []const u8 {
    return switch (os_tag) {
        .linux => "linux",
        .macos => "macos",
        else => @tagName(os_tag),
    };
}

/// Per-triple OpenSSL prefix for non-Windows targets (matches build_external_libs.sh).
fn opensslUnixPrefix(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    const t = target.result;
    return b.fmt("external_lib/lib/openssl-{s}-{s}-{s}", .{
        @tagName(t.cpu.arch),
        opensslOsName(t.os.tag),
        @tagName(t.abi),
    });
}

/// OpenSSL public headers (MD5/SHA*/RIPEMD160/WHIRLPOOL low-level APIs).
fn opensslIncludeRel(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    if (target.result.os.tag == .windows) return "external_lib/openssl/include";
    return b.pathJoin(&.{ opensslUnixPrefix(b, target), "include" });
}

/// Directory containing libcrypto.a / libcrypto.lib after `make install_sw`.
fn opensslLibDirRel(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    if (target.result.os.tag == .windows) return "external_lib/openssl/lib";
    return b.pathJoin(&.{ opensslUnixPrefix(b, target), "lib" });
}

/// Static libcrypto archive path for addObjectFile.
fn opensslCryptoArchiveRel(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    if (target.result.os.tag == .windows) return "external_lib/openssl/lib/libcrypto.lib";
    return b.pathJoin(&.{ opensslUnixPrefix(b, target), "lib", "libcrypto.a" });
}

fn linkOpenSslCrypto(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    mod.addLibraryPath(b.path(opensslLibDirRel(b, target)));
    // Prefer the explicit archive so Zig does not pick up a shared system
    // libcrypto. OpenSSL digests (and their asm) come from this static build.
    mod.addObjectFile(b.path(opensslCryptoArchiveRel(b, target)));
    if (target.result.os.tag == .linux) {
        // libcrypto.a needs these on ELF (cpuid / threads / dlopen providers).
        // On Darwin they live in libSystem; a separate -ldl is not available.
        mod.linkSystemLibrary("pthread", .{});
        mod.linkSystemLibrary("dl", .{});
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
    // Native default mirrors CMake's platform convention: MSVC ABI on Windows
    // (the repo ships prebuilt COFF .lib artifacts under external_lib/), GNU ABI
    // + pinned glibc 2.17 elsewhere. An explicit -Dtarget=… passed by
    // linux_build.sh / windows_build.ps1 overrides both.
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

    // Match CMake `-march=haswell` for x86_64: enables SSE4.2/crc32 used by
    // crc32.c. Only replace the portable baseline default — honor `-Dcpu=…`.
    // aarch64-macos defaults to apple_m1 (Apple Silicon baseline for M1+).
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
    arch_name: []const u8,
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
            // ReleaseFast; that tanks hash throughput (~3–10×). Match CMake.
            .sanitize_c = .off,
        }),
    });

    const mod = lib.root_module;
    mod.addIncludePath(b.path(srclib));
    mod.addIncludePath(b.path(tomcrypt ++ "/src/headers"));
    mod.addIncludePath(b.path(opensslIncludeRel(b, target)));
    mod.addCMacro("USE_KECCAK", "1");
    mod.addCMacro("BLAKE3_NO_AVX512", "1");
    // Portable Blake3 on non-x86 (no SSE/AVX asm; no NEON kernels linked).
    if (target.result.cpu.arch != .x86_64) {
        mod.addCMacro("BLAKE3_USE_NEON", "0");
    }
    // Allow OpenSSL 3+ deprecated low-level digests (MD5/SHA*/RIPEMD160/WHIRLPOOL).
    mod.addCMacro("OPENSSL_API_COMPAT", "0x10100000L");
    mod.addCMacro("ARCH", arch_name);

    var c_sources: [40][]const u8 = undefined;
    var n: usize = 0;
    const sph_sources = [_][]const u8{
        "byte_order.c",
        "crc32.c",
        "edonr.c",
        "gost.c",
        "haval.c",
        "md2.c",
        "md4.c",
        "ripemd.c",
        "sha3.c",
        "snefru.c",
        "tiger.c",
        "tiger_sbox.c",
        "rhash_tiger.c",
        "tth.c",
        "blake3.c",
        "blake3_dispatch.c",
        "blake3_portable.c",
    };
    for (sph_sources) |s| {
        c_sources[n] = b.fmt("{s}/{s}", .{ srclib, s });
        n += 1;
    }

    const tomcrypt_sources = [_][]const u8{
        "hashes/rmd128.c",
        "hashes/rmd160.c",
        "hashes/rmd256.c",
        "hashes/rmd320.c",
        "misc/crypt/crypt_argchk.c",
        "misc/zeromem.c",
    };
    for (tomcrypt_sources) |s| {
        c_sources[n] = b.fmt("{s}/src/{s}", .{ tomcrypt, s });
        n += 1;
    }

    const is_x86_64 = target.result.cpu.arch == .x86_64;
    const is_windows = target.result.os.tag == .windows;

    var flags: [8][]const u8 = undefined;
    var nf: usize = 0;
    flags[nf] = "-Wall";
    nf += 1;
    // Match CMake Release: -O3 is not always implied for C objs in every Zig version.
    flags[nf] = "-O3";
    nf += 1;
    flags[nf] = "-fno-sanitize=undefined";
    nf += 1;
    if (!is_windows) {
        flags[nf] = "-pthread";
        nf += 1;
    }
    flags[nf] = "-DLTC_NO_ROLC";
    nf += 1;

    mod.addCSourceFiles(.{
        .files = c_sources[0..n],
        .flags = flags[0..nf],
    });

    // Match CMake `project(... ASM)`: hand-written SIMD kernels (unix gas).
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
        // intrinsic C kernels (same SIMD degree as the asm path) instead. AVX512
        // stays off via BLAKE3_NO_AVX512 (matches the unix build's active paths).
        // Each file needs its own -m flag so the compiler only emits that ISA.
        const simd = [_]struct { file: []const u8, flags: []const []const u8 }{
            .{ .file = "blake3_avx2.c", .flags = &.{ "-O3", "-fno-sanitize=undefined", "-mavx2" } },
            .{ .file = "blake3_sse41.c", .flags = &.{ "-O3", "-fno-sanitize=undefined", "-msse4.1" } },
            .{ .file = "blake3_sse2.c", .flags = &.{ "-O3", "-fno-sanitize=undefined", "-msse2" } },
        };
        for (simd) |e| {
            mod.addCSourceFile(.{
                .file = b.path(b.fmt("{s}/{s}", .{ srclib, e.file })),
                .flags = e.flags,
            });
        }
    }

    return lib;
}

/// Pool-free brute-force core (`bf_core.c`) plus `lib.c` helpers and Zig-side
/// digest callbacks (`bf_shim.c`). Kept out of `hc-crypto` so targets like
/// `l2h` that already ship a tiny `lib_*` shim don't collide.
fn addBfLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    arch_name: []const u8,
) *std.Build.Step.Compile {
    const srclib = "src/srclib";

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
    mod.addIncludePath(b.path(srclib));
    mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    mod.addIncludePath(b.path("src/abi"));
    mod.addIncludePath(b.path("src/hc"));
    mod.addCMacro("ARCH", arch_name);
    mod.addCMacro("LTC_NO_ROLC", "1");

    const sources = [_][]const u8{
        b.fmt("{s}/lib.c", .{srclib}),
        "src/hc/bf_core.c",
        "src/hc/bf_shim.c",
    };

    const is_windows = target.result.os.tag == .windows;
    var flags: [6][]const u8 = undefined;
    var nf: usize = 0;
    flags[nf] = "-Wall";
    nf += 1;
    flags[nf] = "-O3";
    nf += 1;
    flags[nf] = "-fno-sanitize=undefined";
    nf += 1;
    if (!is_windows) {
        flags[nf] = "-pthread";
        nf += 1;
    }

    mod.addCSourceFiles(.{
        .files = &sources,
        .flags = flags[0..nf],
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
    // Versioned installer vars (CUDA_PATH_V13_2 etc.). Lower priority than the
    // unversioned CUDA_PATH / CUDA_HOME so an explicit current-toolkit pin wins.
    for (b.graph.environ_map.keys(), b.graph.environ_map.values()) |key, value| {
        if (n >= buf.len) break;
        if (key.len <= "CUDA_PATH_V".len) continue;
        if (!std.mem.startsWith(u8, key, "CUDA_PATH_V")) continue;
        buf[n] = b.pathJoin(&.{ value, "bin" });
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
        .macos => {
            if (n < buf.len) {
                buf[n] = "/usr/local/cuda/bin";
                n += 1;
            }
            if (n < buf.len) {
                buf[n] = "/opt/cuda/bin";
                n += 1;
            }
        },
        // Windows: CUDA_PATH / CUDA_PATH_V* (scanned above) + PATH. No stock
        // Program Files walk — Zig 0.16 moved absolute dir APIs to std.Io, and
        // windows_build.ps1 already normalizes CUDA_PATH from CUDA_PATH_V*.
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

/// CUDA runtime/objects come from the host toolkit. Only native Windows and
/// Linux gnu builds may link them; musl, macOS, and cross-arch use the stub.
fn targetSupportsCuda(target: std.Build.ResolvedTarget) bool {
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
) *std.Build.Step.Compile {
    if (enable_cuda) {
        // nvcc is guaranteed present (guarded by nvccAvailable at the call site).
        const nvcc = b.findProgram(&.{"nvcc"}, cudaBinSearchPaths(b)) catch
            @panic("nvcc not found despite enable_cuda");
        // abi/: canonical gpu_abi.h (structs + CUDA macros). cuda_include/:
        // per-algorithm host declarations (md5.h, crc32cu.h, ...) pulled in by
        // the .cu sources. Both are needed by nvcc.
        const inc_abi = b.pathFromRoot("src/abi");
        const inc_cu = b.pathFromRoot("src/cuda_include");

        const lib = b.addLibrary(.{
            .name = "hc-gpu",
            .linkage = .static,
            .root_module = b.createModule(.{
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        lib.root_module.addCSourceFile(.{
            .file = b.path("src/hc/gpu_cuda_marker.c"),
            .flags = &.{},
        });

        // Per-file nvcc compilation → host+device objects (cached individually).
        // Packed into libhc-gpu via addObjectFile (archive-within-archive via ar
        // would yield "not an ELF/COFF file"). Linux keeps -fPIC for the gcc/clang
        // host path; MSVC rejects -fPIC, so Windows omits --compiler-options.
        const is_windows = target.result.os.tag == .windows;
        const obj_ext = if (is_windows) "obj" else "o";
        const cu_bases = [_][]const u8{
            "crc32", "gpu",    "md2",    "md4",    "md5",    "rmd128", "rmd160", "rmd256", "rmd320",
            "sha1",  "sha224", "sha256", "sha384", "sha512", "tiger",  "whirlpool",
        };
        for (cu_bases) |base| {
            const step = b.addSystemCommand(&.{nvcc});
            step.addArgs(&.{ "-c", "-arch=sm_75", "-std=c++17", "-O2" });
            if (!is_windows) step.addArgs(&.{ "--compiler-options", "-fPIC" });
            step.addArgs(&.{ "-I", inc_abi, "-I", inc_cu, "-o" });
            step.setCwd(b.path("."));
            const obj = step.addOutputFileArg(b.fmt("{s}.{s}", .{ base, obj_ext }));
            step.addFileArg(b.path(b.fmt("src/cuda/{s}.cu", .{base})));
            lib.root_module.addObjectFile(obj);
        }

        return lib;
    }

    const stub = b.addLibrary(.{
        .name = "hc-gpu",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    stub.root_module.addIncludePath(b.path("src/abi"));
    stub.root_module.addCSourceFile(.{
        .file = b.path("src/hc/gpu_stub.c"),
        .flags = &.{},
    });
    return stub;
}

fn attachCudaArchive(b: *std.Build, mod: *std.Build.Module) void {
    // The kernel archive is linked into gpu_lib by addGpuLib; here we only pull
    // in the CUDA runtime + the host-code support libs nvcc objects reference.
    if (cudaLibSearchPath(b)) |lib_dir| {
        mod.addLibraryPath(.{ .cwd_relative = lib_dir });
    }
    // Static CUDA runtime: libcudart_static.a (driver is dlopen'd at runtime,
    // so no libcuda link needed). Mirrors the previous release's static linking.
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

fn linkCudaRuntime(b: *std.Build, mod: *std.Build.Module) void {
    attachCudaArchive(b, mod);
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
) void {
    // hc executable: Zig CLI entry point (replaces src/hc/hc.c).
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
    if (builtin.os.tag != .windows) {
        hc_mod.linkSystemLibrary("pthread", .{});
        hc_mod.linkSystemLibrary("dl", .{});
        hc_mod.linkSystemLibrary("m", .{});
    }

    if (enable_cuda) {
        linkCudaRuntime(b, hc_mod);
    }

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
    build_options_mod: *std.Build.Module,
    test_step: *std.Build.Step,
    enable_cuda: bool,
) void {
    const c_code_path = "src/l2h/grammar";
    // b.fmt is the idiomatic 0.16 helper for build-time strings: it returns an
    // arena-duplicated slice and cannot fail, so flex/bison argv are never the
    // empty string that the previous `allocPrint(...) catch ""` produced on OOM.
    const generated_path = b.fmt("{s}/generated", .{c_code_path});

    ensureDirExists(b, generated_path);

    const flex_input = b.fmt("{s}/l2h.lex", .{c_code_path});
    const flex_src = b.fmt("{s}/l2h.flex.c", .{generated_path});
    const flex_hdr = b.fmt("{s}/l2h.flex.h", .{generated_path});
    const flex_opt = b.fmt("--outfile={s}", .{flex_src});
    const flex_hdr_opt = b.fmt("--header-file={s}", .{flex_hdr});

    const bison_input = b.fmt("{s}/l2h.y", .{c_code_path});
    const bison_src = b.fmt("{s}/l2h.tab.c", .{generated_path});
    const bison_opt = b.fmt("--output={s}", .{bison_src});

    // Variadic lib_fprintf/lib_printf for yyerror come from srclib/lib.c via
    // modes -> bf -> hc-bf (Zig cannot export C varargs).
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

    // Static C lib from the generated parser sources. Variadic lib_* printers
    // come from hc-bf (via modes -> bf) rather than a local shim.
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

    // PCRE2: query language MATCH ("~") / NOT MATCH ("!~") operators (processor.c).
    const pcre2_dep = b.dependency("pcre2", .{ .target = target, .optimize = optimize });
    const translate_pcre = b.addTranslateC(.{
        .root_source_file = pcre2_dep.namedLazyPath("pcre2.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_pcre.defineCMacro("PCRE2_CODE_UNIT_WIDTH", "8");
    translate_pcre.step.dependOn(&pcre2_dep.artifact("pcre2-8").step);

    const fehler_dep = b.dependency("fehler", .{});
    const yazap_dep = b.dependency("yazap", .{});

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
    // Computation backends reused from the Zig port.
    l2h_mod.addImport("lib", lib_mod);
    l2h_mod.addImport("hashes", hashes_mod);
    l2h_mod.addImport("modes", modes_mod);
    l2h_mod.addImport("build_options", build_options_mod);
    l2h_mod.addImport("fehler", fehler_dep.module("fehler"));
    l2h_mod.addImport("yazap", yazap_dep.module("yazap"));
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

fn ensureDirExists(b: *std.Build, dir_path: []const u8) void {
    const full_path = b.pathFromRoot(dir_path);
    var dir = std.Io.Dir.cwd().openDir(b.graph.io, full_path, .{}) catch {
        std.Io.Dir.cwd().createDir(b.graph.io, full_path, .default_dir) catch |err| {
            std.debug.print("Failed to create directory '{s}': {s}\n", .{ full_path, @errorName(err) });
        };
        return;
    };
    dir.close(b.graph.io);
}
