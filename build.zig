const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = resolveTarget(b);
    const optimize = b.standardOptimizeOption(.{});

    const arch_name = archName(target.result.cpu.arch);
    const crypto_lib = addCryptoLib(b, target, optimize, arch_name);

    const lib_mod = b.addModule("lib", .{
        .root_source_file = b.path("src/zig/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const hashes_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/hashes.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    hashes_mod.linkLibrary(crypto_lib);
    hashes_mod.addIncludePath(b.path("src/srclib"));
    hashes_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    hashes_mod.addCMacro("USE_KECCAK", "1");
    hashes_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    hashes_mod.addCMacro("ARCH", arch_name);
    hashes_mod.addImport("lib", lib_mod);

    const hashes_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/hashes.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    hashes_test_mod.linkLibrary(crypto_lib);
    hashes_test_mod.addIncludePath(b.path("src/srclib"));
    hashes_test_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    hashes_test_mod.addCMacro("USE_KECCAK", "1");
    hashes_test_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    hashes_test_mod.addCMacro("ARCH", arch_name);
    hashes_test_mod.addImport("lib", lib_mod);

    const hashes_tests = b.addTest(.{ .root_module = hashes_test_mod });
    const run_hashes_tests = b.addRunArtifact(hashes_tests);

    const probe_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/crypto_probe.zig"),
        .target = target,
        .optimize = optimize,
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

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_hashes_tests.step);
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
        }),
    });

    const mod = lib.root_module;
    mod.addIncludePath(b.path(srclib));
    mod.addIncludePath(b.path(tomcrypt ++ "/src/headers"));
    mod.addCMacro("USE_KECCAK", "1");
    mod.addCMacro("BLAKE3_NO_AVX512", "1");
    mod.addCMacro("ARCH", arch_name);

    var c_sources: [32][]const u8 = undefined;
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

    if (is_x86_64 and !is_windows) {
        const asm_sources = [_][]const u8{
            "blake3_avx2_x86-64_unix.S",
            "blake3_avx512_x86-64_unix.S",
            "blake3_sse2_x86-64_unix.S",
            "blake3_sse41_x86-64_unix.S",
        };
        for (asm_sources) |s| {
            c_sources[n] = b.fmt("{s}/{s}", .{ srclib, s });
            n += 1;
        }
    }

    var flags: [8][]const u8 = undefined;
    var nf: usize = 0;
    flags[nf] = "-Wall";
    nf += 1;
    flags[nf] = "-pthread";
    nf += 1;
    flags[nf] = "-DLTC_NO_ROLC";
    nf += 1;

    mod.addCSourceFiles(.{
        .files = c_sources[0..n],
        .flags = flags[0..nf],
    });

    return lib;
}
