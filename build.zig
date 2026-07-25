const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = resolveTarget(b);
    const optimize = b.standardOptimizeOption(.{});
    const strip = optimize != .Debug;

    const arch_name = archName(target.result.cpu.arch);
    const crypto_lib = addCryptoLib(b, target, optimize, arch_name);
    const bf_lib = addBfLib(b, target, optimize, arch_name);
    const whirlpool_lib = addWhirlpoolLib(b, target, optimize);

    const yazap = b.dependency("yazap", .{});

    const version_opt = b.option([]const u8, "version", "Application version") orelse "5.5.0";
    // One binary: link CUDA kernels whenever nvcc is present. Hashes without a
    // GPU implementation (or with no usable device at runtime) stay on CPU.
    // Pass -Dcuda=false only to force stubs (e.g. tooling without a toolkit).
    const want_cuda = b.option(bool, "cuda", "Link CUDA when nvcc is available") orelse true;
    const enable_cuda = want_cuda and nvccAvailable(b);

    const options = b.addOptions();
    options.addOption([]const u8, "version", version_opt);
    options.addOption(bool, "enable_cuda", enable_cuda);
    const build_options_mod = options.createModule();

    const gpu_lib = addGpuLib(b, target, optimize, enable_cuda);

    const lib_mod = b.addModule("lib", .{
        .root_source_file = b.path("src/zig/lib.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
    });

    const gpu_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/gpu.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    gpu_mod.linkLibrary(gpu_lib);
    gpu_mod.addImport("build_options", build_options_mod);
    if (enable_cuda) attachCudaArchive(b, gpu_mod);

    const hashes_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/hashes.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    hashes_mod.linkLibrary(crypto_lib);
    hashes_mod.linkLibrary(gpu_lib);
    hashes_mod.addIncludePath(b.path("src/srclib"));
    hashes_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    hashes_mod.addIncludePath(b.path("external_lib/lib/openssl/include"));
    hashes_mod.addCMacro("USE_KECCAK", "1");
    hashes_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    hashes_mod.addCMacro("OPENSSL_API_COMPAT", "0x10100000L");
    hashes_mod.addCMacro("ARCH", arch_name);
    // Whirlpool: only wp_*.o (+ cleanse stub), not full libcrypto.a — see addWhirlpoolLib.
    hashes_mod.linkLibrary(whirlpool_lib);
    hashes_mod.addImport("lib", lib_mod);
    hashes_mod.addImport("gpu", gpu_mod);

    const hashes_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/hashes.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    hashes_test_mod.linkLibrary(crypto_lib);
    hashes_test_mod.linkLibrary(gpu_lib);
    hashes_test_mod.linkLibrary(whirlpool_lib);
    hashes_test_mod.addIncludePath(b.path("src/srclib"));
    hashes_test_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    hashes_test_mod.addIncludePath(b.path("external_lib/lib/openssl/include"));
    hashes_test_mod.addCMacro("USE_KECCAK", "1");
    hashes_test_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    hashes_test_mod.addCMacro("OPENSSL_API_COMPAT", "0x10100000L");
    hashes_test_mod.addCMacro("ARCH", arch_name);
    hashes_test_mod.addImport("lib", lib_mod);
    hashes_test_mod.addImport("gpu", gpu_mod);

    const hashes_tests = b.addTest(.{ .root_module = hashes_test_mod });
    const run_hashes_tests = b.addRunArtifact(hashes_tests);

    const probe_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/crypto_probe.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    probe_mod.linkLibrary(crypto_lib);
    probe_mod.addIncludePath(b.path("src/srclib"));
    probe_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    probe_mod.addCMacro("USE_KECCAK", "1");
    probe_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    probe_mod.addCMacro("ARCH", arch_name);

    const probe = b.addExecutable(.{
        .name = "crypto_probe",
        .root_module = probe_mod,
    });
    b.installArtifact(probe);

    const run_cmd = b.addRunArtifact(probe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the crypto probe");
    run_step.dependOn(&run_cmd.step);

    const probe_tests = b.addTest(.{
        .root_module = probe_mod,
    });
    const run_tests = b.addRunArtifact(probe_tests);

    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/zig/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const bf_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/bf.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    bf_test_mod.linkLibrary(crypto_lib);
    bf_test_mod.linkLibrary(bf_lib);
    bf_test_mod.addIncludePath(b.path("src/srclib"));
    bf_test_mod.addIncludePath(b.path("src/zig"));
    bf_test_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    bf_test_mod.addIncludePath(b.path("external_lib/lib/apr/include/apr-1"));
    bf_test_mod.addIncludePath(b.path("src/zig/cuda_include"));
    bf_test_mod.addCMacro("USE_KECCAK", "1");
    bf_test_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    bf_test_mod.addCMacro("ARCH", arch_name);
    bf_test_mod.addImport("lib", lib_mod);
    bf_test_mod.addImport("hashes", hashes_mod);
    bf_test_mod.addImport("gpu", gpu_mod);
    bf_test_mod.linkLibrary(gpu_lib);
    bf_test_mod.addObjectFile(b.path("external_lib/lib/apr/lib/libapr-1.a"));
    if (builtin.os.tag != .windows) {
        bf_test_mod.linkSystemLibrary("pthread", .{});
        bf_test_mod.linkSystemLibrary("dl", .{});
    }
    bf_test_mod.linkSystemLibrary("m", .{});

    const bf_tests = b.addTest(.{ .root_module = bf_test_mod });
    const run_bf_tests = b.addRunArtifact(bf_tests);

    const modes_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/modes.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    modes_mod.linkLibrary(crypto_lib);
    modes_mod.linkLibrary(gpu_lib);
    modes_mod.addIncludePath(b.path("src/srclib"));
    modes_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    modes_mod.addCMacro("USE_KECCAK", "1");
    modes_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    modes_mod.addCMacro("ARCH", arch_name);
    modes_mod.addImport("lib", lib_mod);
    modes_mod.addImport("hashes", hashes_mod);

    const modes_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/modes.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    modes_test_mod.linkLibrary(crypto_lib);
    modes_test_mod.linkLibrary(gpu_lib);
    modes_test_mod.addIncludePath(b.path("src/srclib"));
    modes_test_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    modes_test_mod.addCMacro("USE_KECCAK", "1");
    modes_test_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    modes_test_mod.addCMacro("ARCH", arch_name);
    modes_test_mod.addImport("lib", lib_mod);
    modes_test_mod.addImport("hashes", hashes_mod);

    const modes_tests = b.addTest(.{ .root_module = modes_test_mod });
    const run_modes_tests = b.addRunArtifact(modes_tests);

    // Reusable bf module (mirrors hashes_mod setup) so the hc executable and
    // future targets can @import("bf") without re-deriving the crypto wiring.
    const bf_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/bf.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    bf_mod.linkLibrary(crypto_lib);
    bf_mod.linkLibrary(bf_lib);
    bf_mod.addIncludePath(b.path("src/srclib"));
    bf_mod.addIncludePath(b.path("src/zig"));
    bf_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    bf_mod.addIncludePath(b.path("external_lib/lib/apr/include/apr-1"));
    bf_mod.addIncludePath(b.path("src/zig/cuda_include"));
    bf_mod.addCMacro("USE_KECCAK", "1");
    bf_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    bf_mod.addCMacro("ARCH", arch_name);
    bf_mod.addImport("lib", lib_mod);
    bf_mod.addImport("hashes", hashes_mod);
    bf_mod.addImport("gpu", gpu_mod);
    bf_mod.linkLibrary(gpu_lib);
    bf_mod.addObjectFile(b.path("external_lib/lib/apr/lib/libapr-1.a"));
    if (builtin.os.tag != .windows) {
        bf_mod.linkSystemLibrary("pthread", .{});
        bf_mod.linkSystemLibrary("dl", .{});
    }
    bf_mod.linkSystemLibrary("m", .{});

    // modes need bf for hash-restore
    modes_mod.addImport("bf", bf_mod);
    modes_test_mod.addImport("bf", bf_mod);
    modes_test_mod.addImport("gpu", gpu_mod);

    // hc executable: Zig CLI entry point (replaces src/hc/hc.c).
    const hc_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    hc_mod.linkLibrary(crypto_lib);
    hc_mod.linkLibrary(bf_lib);
    hc_mod.linkLibrary(gpu_lib);
    hc_mod.addIncludePath(b.path("src/srclib"));
    hc_mod.addIncludePath(b.path("src/zig"));
    hc_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    hc_mod.addIncludePath(b.path("external_lib/lib/apr/include/apr-1"));
    hc_mod.addIncludePath(b.path("src/zig/cuda_include"));
    hc_mod.addCMacro("USE_KECCAK", "1");
    hc_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    hc_mod.addCMacro("ARCH", arch_name);
    hc_mod.addImport("lib", lib_mod);
    hc_mod.addImport("hashes", hashes_mod);
    hc_mod.addImport("modes", modes_mod);
    hc_mod.addImport("bf", bf_mod);
    hc_mod.addImport("gpu", gpu_mod);
    hc_mod.addImport("yazap", yazap.module("yazap"));
    hc_mod.addImport("build_options", build_options_mod);
    hc_mod.addObjectFile(b.path("external_lib/lib/apr/lib/libapr-1.a"));
    if (builtin.os.tag != .windows) {
        hc_mod.linkSystemLibrary("pthread", .{});
        hc_mod.linkSystemLibrary("dl", .{});
    }
    hc_mod.linkSystemLibrary("m", .{});

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

    const test_step = b.step("test", "Run unit tests");

    addL2h(b, target, optimize, lib_mod, hashes_mod, modes_mod, test_step, enable_cuda);

    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_hashes_tests.step);
    test_step.dependOn(&run_bf_tests.step);
    test_step.dependOn(&run_modes_tests.step);
    test_step.dependOn(&run_hc_tests.step);

    const gpu_tests = b.addTest(.{ .root_module = gpu_mod });
    const run_gpu_tests = b.addRunArtifact(gpu_tests);
    test_step.dependOn(&run_gpu_tests.step);

    const encoding_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/zig/encoding.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_encoding_tests = b.addRunArtifact(encoding_tests);
    test_step.dependOn(&run_encoding_tests.step);

    const hash_gtest_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/tests/hash_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    hash_gtest_mod.linkLibrary(crypto_lib);
    hash_gtest_mod.linkLibrary(gpu_lib);
    hash_gtest_mod.addIncludePath(b.path("src/srclib"));
    hash_gtest_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    hash_gtest_mod.addCMacro("USE_KECCAK", "1");
    hash_gtest_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    hash_gtest_mod.addCMacro("ARCH", arch_name);
    hash_gtest_mod.addImport("lib", lib_mod);
    hash_gtest_mod.addImport("hashes", hashes_mod);
    hash_gtest_mod.addImport("gpu", gpu_mod);
    const hash_gtest = b.addTest(.{ .root_module = hash_gtest_mod });
    const run_hash_gtest = b.addRunArtifact(hash_gtest);
    test_step.dependOn(&run_hash_gtest.step);

    // GoogleTest BruteForceTest parity (src/zig/tests/brute_force_test.zig).
    // Mirrors the bf module wiring: links the C brute-force path + APR helpers
    // and imports the reusable bf module so its lib/hashes/gpu deps resolve.
    const bf_gtest_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/tests/brute_force_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    bf_gtest_mod.linkLibrary(crypto_lib);
    bf_gtest_mod.linkLibrary(bf_lib);
    bf_gtest_mod.linkLibrary(gpu_lib);
    bf_gtest_mod.addIncludePath(b.path("src/srclib"));
    bf_gtest_mod.addIncludePath(b.path("src/zig"));
    bf_gtest_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    bf_gtest_mod.addIncludePath(b.path("external_lib/lib/apr/include/apr-1"));
    bf_gtest_mod.addIncludePath(b.path("src/zig/cuda_include"));
    bf_gtest_mod.addCMacro("USE_KECCAK", "1");
    bf_gtest_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    bf_gtest_mod.addCMacro("ARCH", arch_name);
    bf_gtest_mod.addImport("lib", lib_mod);
    bf_gtest_mod.addImport("hashes", hashes_mod);
    bf_gtest_mod.addImport("gpu", gpu_mod);
    bf_gtest_mod.addImport("bf", bf_mod);
    bf_gtest_mod.addObjectFile(b.path("external_lib/lib/apr/lib/libapr-1.a"));
    if (builtin.os.tag != .windows) {
        bf_gtest_mod.linkSystemLibrary("pthread", .{});
        bf_gtest_mod.linkSystemLibrary("dl", .{});
    }
    bf_gtest_mod.linkSystemLibrary("m", .{});
    const bf_gtest = b.addTest(.{ .root_module = bf_gtest_mod });
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

const pinned_glibc: std.Target.Query.SemanticVersion = .{
    .major = 2,
    .minor = 38,
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
    const default_target: std.Target.Query = .{
        .abi = .gnu,
        .glibc_version = pinned_glibc,
    };

    var query = b.standardTargetOptionsQueryOnly(.{
        .default_target = default_target,
    });

    if (needsHostTripleMaterialization(query)) {
        materializeHostTriple(&query);
    }

    // Match CMake `-march=haswell` for x86_64: enables SSE4.2/crc32 used by
    // crc32.c. Only replace the portable baseline default — honor `-Dcpu=…`.
    const arch = query.cpu_arch orelse builtin.cpu.arch;
    if (arch == .x86_64) {
        switch (query.cpu_model) {
            .baseline, .determined_by_arch_os => {
                query.cpu_model = .{ .explicit = &std.Target.x86.cpu.haswell };
            },
            .native, .explicit => {},
        }
    }

    if (query.glibc_version == null) {
        const os = query.os_tag orelse builtin.target.os.tag;
        if (os == .linux) {
            const abi = query.abi orelse builtin.target.abi;
            if (abi.isGnu()) {
                query.glibc_version = pinned_glibc;
            }
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
    mod.addIncludePath(b.path("external_lib/lib/openssl/include"));
    mod.addCMacro("USE_KECCAK", "1");
    mod.addCMacro("BLAKE3_NO_AVX512", "1");
    // Allow OpenSSL 3 deprecated WHIRLPOOL_* (same as the CMake/OpenSSL build).
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
        "hashes/blake2b.c",
        "hashes/blake2s.c",
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
    }

    return lib;
}

/// C brute-force path (`bf.c`) plus APR helpers (`lib.c`, output, encoding, b64)
/// and Zig-side digest callbacks (`bf_shim.c`). Kept out of `hc-crypto` so
/// targets like `l2h` that already ship a tiny `lib_*` shim don't collide —
/// those targets either omit this lib or drop their shim and use `lib.c`.
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
    mod.addIncludePath(b.path("external_lib/lib/apr/include/apr-1"));
    // bf.h pulls gpu types used by the C brute-force path.
    mod.addIncludePath(b.path("src/zig/cuda_include"));
    mod.addIncludePath(b.path("src/zig")); // bf_shim.h
    mod.addCMacro("ARCH", arch_name);
    mod.addCMacro("LTC_NO_ROLC", "1");

    const sources = [_][]const u8{
        b.fmt("{s}/bf.c", .{srclib}),
        b.fmt("{s}/lib.c", .{srclib}),
        b.fmt("{s}/output.c", .{srclib}),
        b.fmt("{s}/encoding.c", .{srclib}),
        b.fmt("{s}/b64.c", .{srclib}),
        "src/zig/bf_shim.c",
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

/// WHIRLPOOL compiled directly from vendored openssl-4.0.0 sources instead of
/// extracting .o objects from the prebuilt libcrypto.a. On x86_64 it uses the
/// asm whirlpool_block (wp-x86_64.S generated from openssl's perlasm, with the
/// CET note section rewritten to clang-accepted syntax); other architectures
/// fall back to the portable C whirlpool_block in wp_block.c. Either way a
/// small cryptlib.h/cleanse stub is all that's needed — no OPENSSL_ia32cap_P
/// dependency on x86_64 (GO_FOR_MMX is i386-only) and no DT_INIT SEGV from
/// x86_64cpuid.o.
fn addWhirlpoolLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "hc-whirlpool",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .sanitize_c = .off,
        }),
    });
    lib.root_module.addIncludePath(b.path("external_lib/lib/openssl/include"));
    lib.root_module.addIncludePath(b.path("src/zig/openssl_src"));
    lib.root_module.addIncludePath(b.path("src/zig/openssl_src/whrlpool"));

    if (target.result.cpu.arch == .x86_64) {
        // asm-optimized: wp_dgst.c delegates whirlpool_block to wp-x86_64.S.
        lib.root_module.addCSourceFiles(.{
            .files = &.{
                "src/zig/openssl_src/whrlpool/wp_dgst.c",
                "src/zig/openssl_cleanse_stub.c",
            },
            .flags = &.{ "-fno-sanitize=undefined", "-DWHIRLPOOL_ASM" },
        });
        lib.root_module.addCSourceFile(.{
            .file = b.path("src/zig/openssl_src/whrlpool/wp-x86_64.S"),
            .flags = &.{"-fno-sanitize=undefined"},
        });
    } else {
        // portable C fallback: wp_block.c supplies whirlpool_block.
        lib.root_module.addCSourceFiles(.{
            .files = &.{
                "src/zig/openssl_src/whrlpool/wp_dgst.c",
                "src/zig/openssl_src/whrlpool/wp_block.c",
                "src/zig/openssl_cleanse_stub.c",
            },
            .flags = &.{"-fno-sanitize=undefined"},
        });
    }
    return lib;
}

/// Extra directories (beyond PATH) where `nvcc` may live.
/// Prefers `CUDA_PATH` / `CUDA_HOME` (set by the NVIDIA installer on Windows and
/// often on Linux), then falls back to common distro install locations.
fn cudaBinSearchPaths(b: *std.Build) []const []const u8 {
    var buf: [6][]const u8 = undefined;
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
            buf[n] = "/opt/cuda/bin";
            n += 1;
            buf[n] = "/usr/local/cuda/bin";
            n += 1;
        },
        .macos => {
            buf[n] = "/usr/local/cuda/bin";
            n += 1;
            buf[n] = "/opt/cuda/bin";
            n += 1;
        },
        // Windows: CUDA_PATH is the supported discovery mechanism; PATH is
        // already searched by findProgram before these extras.
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
        const inc = b.pathFromRoot("src/zig/cuda_include");

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
            .file = b.path("src/zig/gpu_cuda_marker.c"),
            .flags = &.{},
        });

        // Per-file nvcc compilation → .o objects (cached individually). Each .o
        // is packed straight into libhc-gpu.a via addObjectFile (packing an
        // archive-within-an-archive via ar would yield "not an ELF file").
        const cu_bases = [_][]const u8{
            "crc32", "gpu",    "md2",    "md4",    "md5",    "rmd160",
            "sha1",  "sha224", "sha256", "sha384", "sha512", "whirlpool",
        };
        for (cu_bases) |base| {
            const step = b.addSystemCommand(&.{
                nvcc,          "-c",
                "-arch=sm_75", "-std=c++17",
                "-O2",         "--compiler-options",
                "-fPIC",       "-I",
                inc,           "-o",
            });
            step.setCwd(b.path("."));
            const obj = step.addOutputFileArg(b.fmt("{s}.o", .{base}));
            step.addFileArg(b.path(b.fmt("src/hc/{s}.cu", .{base})));
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
    stub.root_module.addIncludePath(b.path("src/zig/abi"));
    stub.root_module.addCSourceFile(.{
        .file = b.path("src/zig/gpu_stub.c"),
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

/// Wires the l2h (linq2hash) query frontend: runs flex/bison to generate the
/// parser, compiles the generated C into a static lib, exposes the token table
/// and types to Zig through translate-c, and builds the `l2h` executable.
/// Mirrors the grok build pattern (b.addSystemCommand + addLibrary + addTranslateC).
fn addL2h(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    lib_mod: *std.Build.Module,
    hashes_mod: *std.Build.Module,
    modes_mod: *std.Build.Module,
    test_step: *std.Build.Step,
    enable_cuda: bool,
) void {
    const c_code_path = "src/l2h";
    const generated_path = std.fmt.allocPrint(b.allocator, "{s}/generated", .{c_code_path}) catch "";

    ensureDirExists(b, generated_path);

    const flex_input = std.fmt.allocPrint(b.allocator, "{s}/l2h.lex", .{c_code_path}) catch "";
    const flex_src = std.fmt.allocPrint(b.allocator, "{s}/l2h.flex.c", .{generated_path}) catch "";
    const flex_hdr = std.fmt.allocPrint(b.allocator, "{s}/l2h.flex.h", .{generated_path}) catch "";
    const flex_opt = std.fmt.allocPrint(b.allocator, "--outfile={s}", .{flex_src}) catch "";
    const flex_hdr_opt = std.fmt.allocPrint(b.allocator, "--header-file={s}", .{flex_hdr}) catch "";

    const bison_input = std.fmt.allocPrint(b.allocator, "{s}/l2h.y", .{c_code_path}) catch "";
    const bison_src = std.fmt.allocPrint(b.allocator, "{s}/l2h.tab.c", .{generated_path}) catch "";
    const bison_opt = std.fmt.allocPrint(b.allocator, "--output={s}", .{bison_src}) catch "";

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
            flex_args = &[_][]const u8{ "win_flex.exe", "--fast", "--wincompat", flex_opt, flex_hdr_opt, flex_input };
            bison_args = &[_][]const u8{ "win_bison.exe", bison_opt, "-dy", "-Wno-yacc", "-Wno-other", bison_input };
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
    l2h_c_lib.root_module.addIncludePath(b.path("src/zig/l2h/include"));
    l2h_c_lib.root_module.addCSourceFiles(.{ .files = &c_sources, .flags = &[_][]const u8{} });
    l2h_c_lib.step.dependOn(&bison.step);

    // Surface tokens/YYSTYPE/callback externs to Zig.
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/zig/l2h/c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_c.addIncludePath(b.path(c_code_path));
    translate_c.addIncludePath(b.path(generated_path));
    translate_c.addIncludePath(b.path("src/srclib"));
    translate_c.addIncludePath(b.path("src/zig/l2h/include"));
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

    const glob_dep = b.dependency("glob", .{ .target = target, .optimize = optimize });
    const fehler_dep = b.dependency("fehler", .{});
    const yazap_dep = b.dependency("yazap", .{});

    // l2h executable: parser driver (main.zig) + frontend/backend/processor.
    const l2h_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/l2h/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    l2h_mod.linkLibrary(l2h_c_lib);
    l2h_mod.addImport("c", translate_c.createModule());
    l2h_mod.addImport("re", translate_pcre.createModule());
    l2h_mod.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    // Computation backends reused from the Zig port.
    l2h_mod.addImport("lib", lib_mod);
    l2h_mod.addImport("hashes", hashes_mod);
    l2h_mod.addImport("modes", modes_mod);
    // Optional deps surfaced for parity with the grok toolchain.
    l2h_mod.addImport("glob", glob_dep.module("glob"));
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
