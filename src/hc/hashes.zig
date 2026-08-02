const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
// libtomcrypt hashes (ripemd256/320) share the hash_state union.
const ltc = @import("ltc");

/// CRC32C is offered on x86/x86_64 (SSE4.2 HW, or software on older CPUs
/// such as core2). Not offered on aarch64/etc.
pub const have_crc32c = switch (builtin.cpu.arch) {
    .x86_64, .x86 => true,
    else => false,
};

/// Run OpenSSL's CPUID / `OPENSSL_ia32cap_P` setup once.
///
/// Statically linking `libcrypto.a` into a Zig executable often drops the
/// ELF `.init` constructor that would call `OPENSSL_cpuid_setup`, so SHA-NI
/// and related ASM paths never activate (`OPENSSL_ia32cap` env is also inert
/// because that env is read inside cpuid_setup).
///
/// Call this from process startup (`main`) only. Invoking it from every
/// OpenSSL digest path pulls cpuid into unit-test binaries in a way that
/// SEGVs under ReleaseFast/Safe with Zig 0.16 + musl static libcrypto; the
/// software SHA path remains correct for tests.
var openssl_cpuid_done: std.atomic.Value(bool) = .init(false);

/// Ensure OpenSSL ASM dispatch is initialized. Safe to call repeatedly.
pub fn ensureOpenSslReady() void {
    if (openssl_cpuid_done.load(.acquire)) return;
    c.OPENSSL_cpuid_setup();
    openssl_cpuid_done.store(true, .release);
}

pub const InitFn = *const fn (context: *anyopaque) callconv(.c) void;
pub const UpdateFn = *const fn (context: *anyopaque, input: [*]const u8, len: usize) callconv(.c) void;
pub const FinalFn = *const fn (context: *anyopaque, digest: [*]u8) callconv(.c) void;
pub const DigestFn = *const fn (digest: [*]u8, input: [*]const u8, len: usize) callconv(.c) void;

pub const HashDefinition = struct {
    name: []const u8,
    hash_length: usize,
    weight: i32 = 0,
    use_wide_string: bool = false,
    context_size: usize,
    init: InitFn,
    update: UpdateFn,
    final: FinalFn,
    digest: DigestFn,
};

fn streamingDigest(
    comptime Ctx: type,
    comptime initFn: anytype,
    comptime updateFn: anytype,
    comptime closeFn: anytype,
) DigestFn {
    return struct {
        fn call(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
            // Match C DIGEST_BODY: stack ctx + init(), no full memset (blake3
            // hasher is ~2KiB; zeroing it every attempt dominated the crack loop).
            var ctx: Ctx = undefined;
            initFn(&ctx);
            if (input_len != 0) updateFn(&ctx, input, input_len);
            closeFn(&ctx, digest);
        }
    }.call;
}

fn blake3Digest(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
    var hasher: c.blake3_hasher = undefined;
    c.blake3_hasher_init(&hasher);
    if (input_len != 0) c.blake3_hasher_update(&hasher, input, input_len);
    c.blake3_hasher_finalize(&hasher, digest, 32);
}

fn blake3Final(context: *anyopaque, digest: [*]u8) callconv(.c) void {
    c.blake3_hasher_finalize(@ptrCast(@alignCast(context)), digest, 32);
}

// OpenSSL low-level digests (MD5/SHA*/RIPEMD160/WHIRLPOOL): Final(md, ctx)
// is the reverse of our FinalFn order — same as CMake hashes.c.
fn opensslInit(comptime initFn: anytype) InitFn {
    return struct {
        fn call(context: *anyopaque) callconv(.c) void {
            _ = initFn(@ptrCast(@alignCast(context)));
        }
    }.call;
}

fn opensslUpdate(comptime updateFn: anytype) UpdateFn {
    return struct {
        fn call(context: *anyopaque, input: [*]const u8, len: usize) callconv(.c) void {
            _ = updateFn(@ptrCast(@alignCast(context)), input, len);
        }
    }.call;
}

fn opensslFinal(comptime finalFn: anytype) FinalFn {
    return struct {
        fn call(context: *anyopaque, digest: [*]u8) callconv(.c) void {
            _ = finalFn(digest, @ptrCast(@alignCast(context)));
        }
    }.call;
}

fn opensslDigest(
    comptime Ctx: type,
    comptime initFn: anytype,
    comptime updateFn: anytype,
    comptime finalFn: anytype,
) DigestFn {
    return struct {
        fn call(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
            var ctx: Ctx = undefined;
            _ = initFn(&ctx);
            if (input_len != 0) _ = updateFn(&ctx, input, input_len);
            _ = finalFn(digest, &ctx);
        }
    }.call;
}

// HAVAL family: sph_haval_* use untyped (void*) parameters, so the context
// pointer must be cast explicitly (unlike the typed sph/rhash APIs above).
fn havalDigest(
    comptime initFn: anytype,
    comptime updateFn: anytype,
    comptime closeFn: anytype,
) DigestFn {
    return struct {
        fn call(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
            var ctx: c.sph_haval_context = undefined;
            initFn(@ptrCast(&ctx));
            if (input_len != 0) updateFn(@ptrCast(&ctx), @ptrCast(input), input_len);
            closeFn(@ptrCast(&ctx), @ptrCast(digest));
        }
    }.call;
}

// libtomcrypt hashes: init/process/done return int and operate on hash_state.
fn ltcInit(comptime initFn: anytype) InitFn {
    return struct {
        fn call(context: *anyopaque) callconv(.c) void {
            _ = initFn(@ptrCast(@alignCast(context)));
        }
    }.call;
}

fn ltcUpdate(comptime processFn: anytype) UpdateFn {
    return struct {
        fn call(context: *anyopaque, input: [*]const u8, len: usize) callconv(.c) void {
            _ = processFn(@ptrCast(@alignCast(context)), input, @intCast(len));
        }
    }.call;
}

fn ltcFinal(comptime doneFn: anytype) FinalFn {
    return struct {
        fn call(context: *anyopaque, digest: [*]u8) callconv(.c) void {
            _ = doneFn(@ptrCast(@alignCast(context)), digest);
        }
    }.call;
}

fn ltcDigest(comptime initFn: anytype, comptime processFn: anytype, comptime doneFn: anytype) DigestFn {
    return struct {
        fn call(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
            var ctx: ltc.hash_state = undefined;
            _ = initFn(&ctx);
            if (input_len != 0) _ = processFn(&ctx, input, @intCast(input_len));
            _ = doneFn(&ctx, digest);
        }
    }.call;
}

/// Adapters for Zig `std.crypto.hash.*` types (`init`/`update`/`final`).
fn zigHashInit(comptime Hash: type) InitFn {
    return struct {
        fn call(context: *anyopaque) callconv(.c) void {
            const hasher: *Hash = @ptrCast(@alignCast(context));
            hasher.* = Hash.init(.{});
        }
    }.call;
}

fn zigHashUpdate(comptime Hash: type) UpdateFn {
    return struct {
        fn call(context: *anyopaque, input: [*]const u8, len: usize) callconv(.c) void {
            const hasher: *Hash = @ptrCast(@alignCast(context));
            if (len != 0) hasher.update(input[0..len]);
        }
    }.call;
}

fn zigHashFinal(comptime Hash: type) FinalFn {
    return struct {
        fn call(context: *anyopaque, digest: [*]u8) callconv(.c) void {
            const hasher: *Hash = @ptrCast(@alignCast(context));
            hasher.final(digest[0..Hash.digest_length]);
        }
    }.call;
}

fn zigHashDigest(comptime Hash: type) DigestFn {
    return struct {
        fn call(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
            var hasher = Hash.init(.{});
            if (input_len != 0) hasher.update(input[0..input_len]);
            hasher.final(digest[0..Hash.digest_length]);
        }
    }.call;
}

const Blake2b512 = std.crypto.hash.blake2.Blake2b512;
const Blake2s256 = std.crypto.hash.blake2.Blake2s256;

const crc32c_hashes = if (have_crc32c) [_]HashDefinition{
    .{
        .name = "crc32c",
        .hash_length = c.CRC32_HASH_SIZE,
        .weight = 2,
        .context_size = @sizeOf(c.crc32_context_t),
        .init = @ptrCast(&c.crc32c_init),
        .update = @ptrCast(&c.crc32c_update),
        .final = @ptrCast(&c.crc32c_final),
        .digest = streamingDigest(c.crc32_context_t, c.crc32c_init, c.crc32c_update, c.crc32c_final),
    },
} else [_]HashDefinition{};

pub const hashes = [_]HashDefinition{
    .{
        .name = "tiger",
        .hash_length = 24,
        .context_size = @sizeOf(c.sph_tiger_context),
        .init = @ptrCast(&c.sph_tiger_init),
        .update = @ptrCast(&c.sph_tiger),
        .final = @ptrCast(&c.sph_tiger_close),
        .digest = streamingDigest(c.sph_tiger_context, c.sph_tiger_init, c.sph_tiger, c.sph_tiger_close),
    },
    .{
        .name = "tiger2",
        .hash_length = 24,
        .context_size = @sizeOf(c.sph_tiger_context),
        .init = @ptrCast(&c.sph_tiger2_init),
        .update = @ptrCast(&c.sph_tiger2),
        .final = @ptrCast(&c.sph_tiger2_close),
        .digest = streamingDigest(c.sph_tiger_context, c.sph_tiger2_init, c.sph_tiger2, c.sph_tiger2_close),
    },
    .{
        .name = "md2",
        .hash_length = 16,
        .context_size = @sizeOf(c.sph_md2_context),
        .init = @ptrCast(&c.sph_md2_init),
        .update = @ptrCast(&c.sph_md2),
        .final = @ptrCast(&c.sph_md2_close),
        .digest = streamingDigest(c.sph_md2_context, c.sph_md2_init, c.sph_md2, c.sph_md2_close),
    },
    .{
        .name = "md4",
        .hash_length = 16,
        .context_size = @sizeOf(c.sph_md4_context),
        .init = @ptrCast(&c.sph_md4_init),
        .update = @ptrCast(&c.sph_md4),
        .final = @ptrCast(&c.sph_md4_close),
        .digest = streamingDigest(c.sph_md4_context, c.sph_md4_init, c.sph_md4, c.sph_md4_close),
    },
    .{
        // NTLM is MD4 over UTF-16LE (wide) passwords.
        .name = "ntlm",
        .hash_length = 16,
        .use_wide_string = true,
        .context_size = @sizeOf(c.sph_md4_context),
        .init = @ptrCast(&c.sph_md4_init),
        .update = @ptrCast(&c.sph_md4),
        .final = @ptrCast(&c.sph_md4_close),
        .digest = streamingDigest(c.sph_md4_context, c.sph_md4_init, c.sph_md4, c.sph_md4_close),
    },
    .{
        .name = "ripemd160",
        .hash_length = c.RIPEMD160_DIGEST_LENGTH,
        .context_size = @sizeOf(c.RIPEMD160_CTX),
        .init = opensslInit(c.RIPEMD160_Init),
        .update = opensslUpdate(c.RIPEMD160_Update),
        .final = opensslFinal(c.RIPEMD160_Final),
        .digest = opensslDigest(c.RIPEMD160_CTX, c.RIPEMD160_Init, c.RIPEMD160_Update, c.RIPEMD160_Final),
    },
    .{
        .name = "ripemd128",
        .hash_length = 16,
        .context_size = @sizeOf(c.sph_ripemd128_context),
        .init = @ptrCast(&c.sph_ripemd128_init),
        .update = @ptrCast(&c.sph_ripemd128),
        .final = @ptrCast(&c.sph_ripemd128_close),
        .digest = streamingDigest(c.sph_ripemd128_context, c.sph_ripemd128_init, c.sph_ripemd128, c.sph_ripemd128_close),
    },
    .{
        .name = "blake3",
        .hash_length = 32,
        .context_size = @sizeOf(c.blake3_hasher),
        .init = @ptrCast(&c.blake3_hasher_init),
        .update = @ptrCast(&c.blake3_hasher_update),
        .final = &blake3Final,
        .digest = &blake3Digest,
    },
    .{
        .name = "whirlpool",
        .hash_length = c.WHIRLPOOL_DIGEST_LENGTH,
        .context_size = @sizeOf(c.WHIRLPOOL_CTX),
        .init = opensslInit(c.WHIRLPOOL_Init),
        .update = opensslUpdate(c.WHIRLPOOL_Update),
        .final = opensslFinal(c.WHIRLPOOL_Final),
        .digest = opensslDigest(c.WHIRLPOOL_CTX, c.WHIRLPOOL_Init, c.WHIRLPOOL_Update, c.WHIRLPOOL_Final),
    },

    // ---- GOST (CryptoPro S-box, matches the app's "gost" algorithm) ----
    // NOTE: rhash_gost_init_table() is intentionally NOT called. The build does
    // not define GENERATE_GOST_LOOKUP_TABLE, so the S-box lookup tables are
    // statically pre-initialized in gost.c and ready to use.
    .{
        .name = "gost",
        .hash_length = 32,
        .context_size = @sizeOf(c.gost_ctx),
        .init = @ptrCast(&c.rhash_gost_cryptopro_init),
        .update = @ptrCast(&c.rhash_gost_update),
        .final = @ptrCast(&c.rhash_gost_final),
        .digest = streamingDigest(c.gost_ctx, c.rhash_gost_cryptopro_init, c.rhash_gost_update, c.rhash_gost_final),
    },
    .{
        .name = "tth",
        .hash_length = 24,
        .context_size = @sizeOf(c.tth_ctx),
        .init = @ptrCast(&c.rhash_tth_init),
        .update = @ptrCast(&c.rhash_tth_update),
        .final = @ptrCast(&c.rhash_tth_final),
        .digest = streamingDigest(c.tth_ctx, c.rhash_tth_init, c.rhash_tth_update, c.rhash_tth_final),
    },
    .{
        .name = "snefru128",
        .hash_length = 16,
        .context_size = @sizeOf(c.snefru_ctx),
        .init = @ptrCast(&c.rhash_snefru128_init),
        .update = @ptrCast(&c.rhash_snefru_update),
        .final = @ptrCast(&c.rhash_snefru_final),
        .digest = streamingDigest(c.snefru_ctx, c.rhash_snefru128_init, c.rhash_snefru_update, c.rhash_snefru_final),
    },
    .{
        .name = "snefru256",
        .hash_length = 32,
        .context_size = @sizeOf(c.snefru_ctx),
        .init = @ptrCast(&c.rhash_snefru256_init),
        .update = @ptrCast(&c.rhash_snefru_update),
        .final = @ptrCast(&c.rhash_snefru_final),
        .digest = streamingDigest(c.snefru_ctx, c.rhash_snefru256_init, c.rhash_snefru_update, c.rhash_snefru_final),
    },
    .{
        .name = "edonr256",
        .hash_length = 32,
        .context_size = @sizeOf(c.edonr_ctx),
        .init = @ptrCast(&c.rhash_edonr256_init),
        .update = @ptrCast(&c.rhash_edonr256_update),
        .final = @ptrCast(&c.rhash_edonr256_final),
        .digest = streamingDigest(c.edonr_ctx, c.rhash_edonr256_init, c.rhash_edonr256_update, c.rhash_edonr256_final),
    },
    .{
        .name = "edonr512",
        .hash_length = 64,
        .context_size = @sizeOf(c.edonr_ctx),
        .init = @ptrCast(&c.rhash_edonr512_init),
        .update = @ptrCast(&c.rhash_edonr512_update),
        .final = @ptrCast(&c.rhash_edonr512_final),
        .digest = streamingDigest(c.edonr_ctx, c.rhash_edonr512_init, c.rhash_edonr512_update, c.rhash_edonr512_final),
    },

    // ---- HAVAL family (15 variants; shared sph_haval_context) ----
    .{
        .name = "haval-128-3",
        .hash_length = 16,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval128_3_init),
        .update = @ptrCast(&c.sph_haval128_3),
        .final = @ptrCast(&c.sph_haval128_3_close),
        .digest = havalDigest(c.sph_haval128_3_init, c.sph_haval128_3, c.sph_haval128_3_close),
    },
    .{
        .name = "haval-128-4",
        .hash_length = 16,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval128_4_init),
        .update = @ptrCast(&c.sph_haval128_4),
        .final = @ptrCast(&c.sph_haval128_4_close),
        .digest = havalDigest(c.sph_haval128_4_init, c.sph_haval128_4, c.sph_haval128_4_close),
    },
    .{
        .name = "haval-128-5",
        .hash_length = 16,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval128_5_init),
        .update = @ptrCast(&c.sph_haval128_5),
        .final = @ptrCast(&c.sph_haval128_5_close),
        .digest = havalDigest(c.sph_haval128_5_init, c.sph_haval128_5, c.sph_haval128_5_close),
    },
    .{
        .name = "haval-160-3",
        .hash_length = 20,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval160_3_init),
        .update = @ptrCast(&c.sph_haval160_3),
        .final = @ptrCast(&c.sph_haval160_3_close),
        .digest = havalDigest(c.sph_haval160_3_init, c.sph_haval160_3, c.sph_haval160_3_close),
    },
    .{
        .name = "haval-160-4",
        .hash_length = 20,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval160_4_init),
        .update = @ptrCast(&c.sph_haval160_4),
        .final = @ptrCast(&c.sph_haval160_4_close),
        .digest = havalDigest(c.sph_haval160_4_init, c.sph_haval160_4, c.sph_haval160_4_close),
    },
    .{
        .name = "haval-160-5",
        .hash_length = 20,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval160_5_init),
        .update = @ptrCast(&c.sph_haval160_5),
        .final = @ptrCast(&c.sph_haval160_5_close),
        .digest = havalDigest(c.sph_haval160_5_init, c.sph_haval160_5, c.sph_haval160_5_close),
    },
    .{
        .name = "haval-192-3",
        .hash_length = 24,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval192_3_init),
        .update = @ptrCast(&c.sph_haval192_3),
        .final = @ptrCast(&c.sph_haval192_3_close),
        .digest = havalDigest(c.sph_haval192_3_init, c.sph_haval192_3, c.sph_haval192_3_close),
    },
    .{
        .name = "haval-192-4",
        .hash_length = 24,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval192_4_init),
        .update = @ptrCast(&c.sph_haval192_4),
        .final = @ptrCast(&c.sph_haval192_4_close),
        .digest = havalDigest(c.sph_haval192_4_init, c.sph_haval192_4, c.sph_haval192_4_close),
    },
    .{
        .name = "haval-192-5",
        .hash_length = 24,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval192_5_init),
        .update = @ptrCast(&c.sph_haval192_5),
        .final = @ptrCast(&c.sph_haval192_5_close),
        .digest = havalDigest(c.sph_haval192_5_init, c.sph_haval192_5, c.sph_haval192_5_close),
    },
    .{
        .name = "haval-224-3",
        .hash_length = 28,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval224_3_init),
        .update = @ptrCast(&c.sph_haval224_3),
        .final = @ptrCast(&c.sph_haval224_3_close),
        .digest = havalDigest(c.sph_haval224_3_init, c.sph_haval224_3, c.sph_haval224_3_close),
    },
    .{
        .name = "haval-224-4",
        .hash_length = 28,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval224_4_init),
        .update = @ptrCast(&c.sph_haval224_4),
        .final = @ptrCast(&c.sph_haval224_4_close),
        .digest = havalDigest(c.sph_haval224_4_init, c.sph_haval224_4, c.sph_haval224_4_close),
    },
    .{
        .name = "haval-224-5",
        .hash_length = 28,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval224_5_init),
        .update = @ptrCast(&c.sph_haval224_5),
        .final = @ptrCast(&c.sph_haval224_5_close),
        .digest = havalDigest(c.sph_haval224_5_init, c.sph_haval224_5, c.sph_haval224_5_close),
    },
    .{
        .name = "haval-256-3",
        .hash_length = 32,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval256_3_init),
        .update = @ptrCast(&c.sph_haval256_3),
        .final = @ptrCast(&c.sph_haval256_3_close),
        .digest = havalDigest(c.sph_haval256_3_init, c.sph_haval256_3, c.sph_haval256_3_close),
    },
    .{
        .name = "haval-256-4",
        .hash_length = 32,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval256_4_init),
        .update = @ptrCast(&c.sph_haval256_4),
        .final = @ptrCast(&c.sph_haval256_4_close),
        .digest = havalDigest(c.sph_haval256_4_init, c.sph_haval256_4, c.sph_haval256_4_close),
    },
    .{
        .name = "haval-256-5",
        .hash_length = 32,
        .context_size = @sizeOf(c.sph_haval_context),
        .init = @ptrCast(&c.sph_haval256_5_init),
        .update = @ptrCast(&c.sph_haval256_5),
        .final = @ptrCast(&c.sph_haval256_5_close),
        .digest = havalDigest(c.sph_haval256_5_init, c.sph_haval256_5, c.sph_haval256_5_close),
    },

    // ---- SHA-3 / Keccak (rhash; keccak_final differs from sha3_final) ----
    .{
        .name = "sha-3-224",
        .hash_length = 28,
        .context_size = @sizeOf(c.sha3_ctx),
        .init = @ptrCast(&c.rhash_sha3_224_init),
        .update = @ptrCast(&c.rhash_sha3_update),
        .final = @ptrCast(&c.rhash_sha3_final),
        .digest = streamingDigest(c.sha3_ctx, c.rhash_sha3_224_init, c.rhash_sha3_update, c.rhash_sha3_final),
    },
    .{
        .name = "sha-3-256",
        .hash_length = 32,
        .context_size = @sizeOf(c.sha3_ctx),
        .init = @ptrCast(&c.rhash_sha3_256_init),
        .update = @ptrCast(&c.rhash_sha3_update),
        .final = @ptrCast(&c.rhash_sha3_final),
        .digest = streamingDigest(c.sha3_ctx, c.rhash_sha3_256_init, c.rhash_sha3_update, c.rhash_sha3_final),
    },
    .{
        .name = "sha-3-384",
        .hash_length = 48,
        .context_size = @sizeOf(c.sha3_ctx),
        .init = @ptrCast(&c.rhash_sha3_384_init),
        .update = @ptrCast(&c.rhash_sha3_update),
        .final = @ptrCast(&c.rhash_sha3_final),
        .digest = streamingDigest(c.sha3_ctx, c.rhash_sha3_384_init, c.rhash_sha3_update, c.rhash_sha3_final),
    },
    .{
        .name = "sha-3-512",
        .hash_length = 64,
        .context_size = @sizeOf(c.sha3_ctx),
        .init = @ptrCast(&c.rhash_sha3_512_init),
        .update = @ptrCast(&c.rhash_sha3_update),
        .final = @ptrCast(&c.rhash_sha3_final),
        .digest = streamingDigest(c.sha3_ctx, c.rhash_sha3_512_init, c.rhash_sha3_update, c.rhash_sha3_final),
    },
    .{
        .name = "sha-3k-224",
        .hash_length = 28,
        .context_size = @sizeOf(c.sha3_ctx),
        .init = @ptrCast(&c.rhash_keccak_224_init),
        .update = @ptrCast(&c.rhash_keccak_update),
        .final = @ptrCast(&c.rhash_keccak_final),
        .digest = streamingDigest(c.sha3_ctx, c.rhash_keccak_224_init, c.rhash_keccak_update, c.rhash_keccak_final),
    },
    .{
        .name = "sha-3k-256",
        .hash_length = 32,
        .context_size = @sizeOf(c.sha3_ctx),
        .init = @ptrCast(&c.rhash_keccak_256_init),
        .update = @ptrCast(&c.rhash_keccak_update),
        .final = @ptrCast(&c.rhash_keccak_final),
        .digest = streamingDigest(c.sha3_ctx, c.rhash_keccak_256_init, c.rhash_keccak_update, c.rhash_keccak_final),
    },
    .{
        .name = "sha-3k-384",
        .hash_length = 48,
        .context_size = @sizeOf(c.sha3_ctx),
        .init = @ptrCast(&c.rhash_keccak_384_init),
        .update = @ptrCast(&c.rhash_keccak_update),
        .final = @ptrCast(&c.rhash_keccak_final),
        .digest = streamingDigest(c.sha3_ctx, c.rhash_keccak_384_init, c.rhash_keccak_update, c.rhash_keccak_final),
    },
    .{
        .name = "sha-3k-512",
        .hash_length = 64,
        .context_size = @sizeOf(c.sha3_ctx),
        .init = @ptrCast(&c.rhash_keccak_512_init),
        .update = @ptrCast(&c.rhash_keccak_update),
        .final = @ptrCast(&c.rhash_keccak_final),
        .digest = streamingDigest(c.sha3_ctx, c.rhash_keccak_512_init, c.rhash_keccak_update, c.rhash_keccak_final),
    },

    // ---- libtomcrypt (ripemd256/320) + std blake2 ----
    .{
        .name = "ripemd256",
        .hash_length = 32,
        .context_size = @sizeOf(ltc.hash_state),
        .init = ltcInit(ltc.rmd256_init),
        .update = ltcUpdate(ltc.rmd256_process),
        .final = ltcFinal(ltc.rmd256_done),
        .digest = ltcDigest(ltc.rmd256_init, ltc.rmd256_process, ltc.rmd256_done),
    },
    .{
        .name = "ripemd320",
        .hash_length = 40,
        .context_size = @sizeOf(ltc.hash_state),
        .init = ltcInit(ltc.rmd320_init),
        .update = ltcUpdate(ltc.rmd320_process),
        .final = ltcFinal(ltc.rmd320_done),
        .digest = ltcDigest(ltc.rmd320_init, ltc.rmd320_process, ltc.rmd320_done),
    },
    .{
        .name = "blake2b",
        .hash_length = Blake2b512.digest_length,
        .context_size = @sizeOf(Blake2b512),
        .init = zigHashInit(Blake2b512),
        .update = zigHashUpdate(Blake2b512),
        .final = zigHashFinal(Blake2b512),
        .digest = zigHashDigest(Blake2b512),
    },
    .{
        .name = "blake2s",
        .hash_length = Blake2s256.digest_length,
        .context_size = @sizeOf(Blake2s256),
        .init = zigHashInit(Blake2s256),
        .update = zigHashUpdate(Blake2s256),
        .final = zigHashFinal(Blake2s256),
        .digest = zigHashDigest(Blake2s256),
    },

    // ---- CRC32 / CRC32C (srclib; CRC32C is HW on SSE4.2, soft on core2) ----
    .{
        .name = "crc32",
        .hash_length = c.CRC32_HASH_SIZE,
        .weight = 2,
        .context_size = @sizeOf(c.crc32_context_t),
        .init = @ptrCast(&c.crc32_init),
        .update = @ptrCast(&c.crc32_update),
        .final = @ptrCast(&c.crc32_final),
        .digest = streamingDigest(c.crc32_context_t, c.crc32_init, c.crc32_update, c.crc32_final),
    },
} ++ crc32c_hashes ++ [_]HashDefinition{
    .{
        .name = "md5",
        .hash_length = c.MD5_DIGEST_LENGTH,
        .context_size = @sizeOf(c.MD5_CTX),
        .init = opensslInit(c.MD5_Init),
        .update = opensslUpdate(c.MD5_Update),
        .final = opensslFinal(c.MD5_Final),
        .digest = opensslDigest(c.MD5_CTX, c.MD5_Init, c.MD5_Update, c.MD5_Final),
    },
    .{
        .name = "sha1",
        .hash_length = c.SHA_DIGEST_LENGTH,
        .context_size = @sizeOf(c.SHA_CTX),
        .init = opensslInit(c.SHA1_Init),
        .update = opensslUpdate(c.SHA1_Update),
        .final = opensslFinal(c.SHA1_Final),
        .digest = opensslDigest(c.SHA_CTX, c.SHA1_Init, c.SHA1_Update, c.SHA1_Final),
    },
    .{
        .name = "sha224",
        .hash_length = c.SHA224_DIGEST_LENGTH,
        .context_size = @sizeOf(c.SHA256_CTX),
        .init = opensslInit(c.SHA224_Init),
        .update = opensslUpdate(c.SHA224_Update),
        .final = opensslFinal(c.SHA224_Final),
        .digest = opensslDigest(c.SHA256_CTX, c.SHA224_Init, c.SHA224_Update, c.SHA224_Final),
    },
    .{
        .name = "sha256",
        .hash_length = c.SHA256_DIGEST_LENGTH,
        .context_size = @sizeOf(c.SHA256_CTX),
        .init = opensslInit(c.SHA256_Init),
        .update = opensslUpdate(c.SHA256_Update),
        .final = opensslFinal(c.SHA256_Final),
        .digest = opensslDigest(c.SHA256_CTX, c.SHA256_Init, c.SHA256_Update, c.SHA256_Final),
    },
    .{
        .name = "sha384",
        .hash_length = c.SHA384_DIGEST_LENGTH,
        .context_size = @sizeOf(c.SHA512_CTX),
        .init = opensslInit(c.SHA384_Init),
        .update = opensslUpdate(c.SHA384_Update),
        .final = opensslFinal(c.SHA384_Final),
        .digest = opensslDigest(c.SHA512_CTX, c.SHA384_Init, c.SHA384_Update, c.SHA384_Final),
    },
    .{
        .name = "sha512",
        .hash_length = c.SHA512_DIGEST_LENGTH,
        .context_size = @sizeOf(c.SHA512_CTX),
        .init = opensslInit(c.SHA512_Init),
        .update = opensslUpdate(c.SHA512_Update),
        .final = opensslFinal(c.SHA512_Final),
        .digest = opensslDigest(c.SHA512_CTX, c.SHA512_Init, c.SHA512_Update, c.SHA512_Final),
    },
};

pub fn getHash(name: []const u8) ?*const HashDefinition {
    for (&hashes) |*h| {
        if (std.ascii.eqlIgnoreCase(h.name, name)) return h;
    }
    return null;
}

pub fn compute(h: *const HashDefinition, input: []const u8, out: []u8) void {
    h.digest(out.ptr, input.ptr, input.len);
}

pub fn count() usize {
    return hashes.len;
}

fn expectHash(h: *const HashDefinition, input: []const u8, expected_hex: []const u8) !void {
    var digest: [64]u8 align(8) = std.mem.zeroes([64]u8);
    compute(h, input, &digest);
    var expected: [64]u8 = std.mem.zeroes([64]u8);
    _ = try std.fmt.hexToBytes(expected[0..h.hash_length], expected_hex);
    try std.testing.expectEqualSlices(u8, expected[0..h.hash_length], digest[0..h.hash_length]);
}

test "tiger empty via dispatch table" {
    try expectHash(getHash("tiger").?, "", "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3");
}

test "blake3 empty via dispatch table" {
    try expectHash(getHash("blake3").?, "", "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");
}

test "md2 empty via dispatch table" {
    try expectHash(getHash("md2").?, "", "8350e5a3e24c153df2275c9f80692773");
}

test "ripemd160 empty via dispatch table" {
    try expectHash(getHash("ripemd160").?, "", "9c1185a5c5e9fc54612808977ee8f548b2258d31");
}

test "ripemd128 empty via dispatch table" {
    try expectHash(getHash("ripemd128").?, "", "cdf26213a150dc3ecb610f18f6b38b46");
}

test "getHash case-insensitive" {
    try std.testing.expect(getHash("TIGER") != null);
    try std.testing.expect(getHash("Blake3") != null);
    try std.testing.expect(getHash("nope") == null);
}

test "hash count" {
    const expected: usize = if (have_crc32c) 50 else 49;
    try std.testing.expectEqual(expected, count());
}

test "gost empty via dispatch table" {
    try expectHash(getHash("gost").?, "", "981e5f3ca30c841487830f84fb433e13ac1101569b9c13584ac483234cd656c0");
}

test "tth empty via dispatch table" {
    try expectHash(getHash("tth").?, "", "5d9ed00a030e638bdb753a6a24fb900e5a63b8e73e6c25b6");
}

test "snefru128 empty via dispatch table" {
    try expectHash(getHash("snefru128").?, "", "8617f366566a011837f4fb4ba5bedea2");
}

test "snefru256 empty via dispatch table" {
    try expectHash(getHash("snefru256").?, "", "8617f366566a011837f4fb4ba5bedea2b892f3ed8b894023d16ae344b2be5881");
}

test "edonr256 empty via dispatch table" {
    try expectHash(getHash("edonr256").?, "", "86e7c84024c55dbdc9339b395c95e88db8f781719851ad1d237c6e6a8e370b80");
}

test "edonr512 empty via dispatch table" {
    try expectHash(getHash("edonr512").?, "", "c7afbdf3e5b4590eb0b25000bf83fb16d4f9b722ee7f9a2dc2bd382035e8ee38d6f6f15c7b8eec85355ac59af989799950c64557eab0e687d0fcbdba90ae9704");
}

test "haval family empty via dispatch table" {
    try expectHash(getHash("haval-128-3").?, "", "c68f39913f901f3ddf44c707357a7d70");
    try expectHash(getHash("haval-128-4").?, "", "ee6bbf4d6a46a679b3a856c88538bb98");
    try expectHash(getHash("haval-128-5").?, "", "184b8482a0c050dca54b59c7f05bf5dd");
    try expectHash(getHash("haval-160-3").?, "", "d353c3ae22a25401d257643836d7231a9a95f953");
    try expectHash(getHash("haval-160-4").?, "", "1d33aae1be4146dbaaca0b6e70d7a11f10801525");
    try expectHash(getHash("haval-160-5").?, "", "255158cfc1eed1a7be7c55ddd64d9790415b933b");
    try expectHash(getHash("haval-192-3").?, "", "e9c48d7903eaf2a91c5b350151efcb175c0fc82de2289a4e");
    try expectHash(getHash("haval-192-4").?, "", "4a8372945afa55c7dead800311272523ca19d42ea47b72da");
    try expectHash(getHash("haval-192-5").?, "", "4839d0626f95935e17ee2fc4509387bbe2cc46cb382ffe85");
    try expectHash(getHash("haval-224-3").?, "", "c5aae9d47bffcaaf84a8c6e7ccacd60a0dd1932be7b1a192b9214b6d");
    try expectHash(getHash("haval-224-4").?, "", "3e56243275b3b81561750550e36fcd676ad2f5dd9e15f2e89e6ed78e");
    try expectHash(getHash("haval-224-5").?, "", "4a0513c032754f5582a758d35917ac9adf3854219b39e3ac77d1837e");
    try expectHash(getHash("haval-256-3").?, "", "4f6938531f0bc8991f62da7bbd6f7de3fad44562b8c6f4ebf146d5b4e46f7c17");
    try expectHash(getHash("haval-256-4").?, "", "c92b2e23091e80e375dadce26982482d197b1a2521be82da819f8ca2c579b99b");
    try expectHash(getHash("haval-256-5").?, "", "be417bb4dd5cfb76c7126f4f8eeb1553a449039307b1a3cd451dbfdc0fbbe330");
}

test "sha-3 family empty via dispatch table" {
    try expectHash(getHash("sha-3-224").?, "", "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7");
    try expectHash(getHash("sha-3-256").?, "", "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
    try expectHash(getHash("sha-3-384").?, "", "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2ac3713831264adb47fb6bd1e058d5f004");
    try expectHash(getHash("sha-3-512").?, "", "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26");
}

test "sha-3k (keccak) family empty via dispatch table" {
    try expectHash(getHash("sha-3k-224").?, "", "f71837502ba8e10837bdd8d365adb85591895602fc552b48b7390abd");
    try expectHash(getHash("sha-3k-256").?, "", "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
    try expectHash(getHash("sha-3k-384").?, "", "2c23146a63a29acf99e73b88f8c24eaa7dc60aa771780ccc006afbfa8fe2479b2dd2b21362337441ac12b515911957ff");
    try expectHash(getHash("sha-3k-512").?, "", "0eab42de4c3ceb9235fc91acffe746b29c29a8c366b7c60e4e67c466f36a4304c00fa9caf9d87976ba469bcbe06713b435f091ef2769fb160cdab33d3670680e");
}

test "ripemd256 empty via dispatch table" {
    try expectHash(getHash("ripemd256").?, "", "02ba4c4e5f8ecd1877fc52d64d30e37a2d9774fb1e5d026380ae0168e3c5522d");
}

test "ripemd320 empty via dispatch table" {
    try expectHash(getHash("ripemd320").?, "", "22d65d5661536cdc75c1fdf5c6de7b41b9f27325ebc61e8557177d705a0ec880151c3a32a00899b8");
}

test "blake2b empty via dispatch table" {
    try expectHash(getHash("blake2b").?, "", "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce");
}

test "blake2s empty via dispatch table" {
    try expectHash(getHash("blake2s").?, "", "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9");
}

test "md5 empty via dispatch table" {
    try expectHash(getHash("md5").?, "", "d41d8cd98f00b204e9800998ecf8427e");
}

test "crc32 empty and abc via dispatch table" {
    try expectHash(getHash("crc32").?, "", "00000000");
    // Matches _tst/HashTest.h vector for the shared sample input path ("123").
    try expectHash(getHash("crc32").?, "123", "884863d2");
}

test "crc32c of 123 via dispatch table" {
    if (!have_crc32c) return error.SkipZigTest;
    try expectHash(getHash("crc32c").?, "123", "107b2fb2");
}

test "sha1 empty via dispatch table" {
    try expectHash(getHash("sha1").?, "", "da39a3ee5e6b4b0d3255bfef95601890afd80709");
}

test "sha224 empty via dispatch table" {
    try expectHash(getHash("sha224").?, "", "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f");
}

test "sha256 empty via dispatch table" {
    try expectHash(getHash("sha256").?, "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

test "sha384 empty via dispatch table" {
    try expectHash(getHash("sha384").?, "", "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b");
}

test "sha512 empty via dispatch table" {
    try expectHash(getHash("sha512").?, "", "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}

test "tiger2 empty via dispatch table" {
    try expectHash(getHash("tiger2").?, "", "4441be75f6018773c206c22745374b924aa8313fef919f41");
}

// Non-empty inputs exercise the update() path (the empty-string tests above
// skip it). One representative per wrapper family: streamingDigest (gost),
// havalDigest, ltcDigest (ripemd256), zigHashDigest (blake2b).
test "update path: gost of abc" {
    try expectHash(getHash("gost").?, "abc", "b285056dbf18d7392d7677369524dd14747459ed8143997e163b2986f92fd42c");
}

test "update path: haval-256-3 of abc" {
    try expectHash(getHash("haval-256-3").?, "abc", "8699f1e3384d05b2a84b032693e2b6f46df85a13a50d93808d6874bb8fb9e86c");
}

test "update path: blake2b of abc" {
    try expectHash(getHash("blake2b").?, "abc", "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923");
}

test "update path: ripemd256 of abc" {
    try expectHash(getHash("ripemd256").?, "abc", "afbd6e228b9d8cbbcef5ca2d03e6dba10ac0bc7dcbe4680e1e42d2e975459b65");
}

test "update path: sha256 of abc" {
    try expectHash(getHash("sha256").?, "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}
