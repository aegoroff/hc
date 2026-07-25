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

    const bf_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/bf.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    bf_test_mod.linkLibrary(crypto_lib);
    bf_test_mod.addIncludePath(b.path("src/srclib"));
    bf_test_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    bf_test_mod.addCMacro("USE_KECCAK", "1");
    bf_test_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    bf_test_mod.addCMacro("ARCH", arch_name);
    bf_test_mod.addImport("lib", lib_mod);
    bf_test_mod.addImport("hashes", hashes_mod);

    const bf_tests = b.addTest(.{ .root_module = bf_test_mod });
    const run_bf_tests = b.addRunArtifact(bf_tests);

    const modes_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/modes.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    modes_mod.linkLibrary(crypto_lib);
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
    modes_test_mod.addIncludePath(b.path("src/srclib"));
    modes_test_mod.addIncludePath(b.path("src/libtomcrypt/src/headers"));
    modes_test_mod.addCMacro("USE_KECCAK", "1");
    modes_test_mod.addCMacro("BLAKE3_NO_AVX512", "1");
    modes_test_mod.addCMacro("ARCH", arch_name);
    modes_test_mod.addImport("lib", lib_mod);
    modes_test_mod.addImport("hashes", hashes_mod);

    const modes_tests = b.addTest(.{ .root_module = modes_test_mod });
    const run_modes_tests = b.addRunArtifact(modes_tests);

    addL2h(b, target, optimize);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_hashes_tests.step);
    test_step.dependOn(&run_bf_tests.step);
    test_step.dependOn(&run_modes_tests.step);
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

/// Wires the l2h (linq2hash) query frontend: runs flex/bison to generate the
/// parser, compiles the generated C into a static lib, exposes the token table
/// and types to Zig through translate-c, and builds the `l2h` executable.
/// Mirrors the grok build pattern (b.addSystemCommand + addLibrary + addTranslateC).
fn addL2h(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
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

    // Static C lib from the generated parser sources.
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

    // Minimal l2h executable.
    const l2h_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/l2h/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    l2h_mod.linkLibrary(l2h_c_lib);
    l2h_mod.addImport("c", translate_c.createModule());

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
