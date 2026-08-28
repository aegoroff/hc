const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const ltc = @import("ltc"); // libtomcrypt hashes (ripemd256/320) share the hash_state union.

/// CRC32C on x86/x86_64 (SSE4.2 HW or software) and aarch64 (CRC32 HW or soft).
pub const have_crc32c = switch (builtin.cpu.arch) {
    .x86_64, .x86, .aarch64 => true,
    else => false,
};

var openssl_cpuid_done: std.atomic.Value(bool) = .init(false);

/// Ensure OpenSSL ASM dispatch is initialized. Safe to call repeatedly.
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
    /// One-line CLI help text (`hc -h`, `hc <algo> -h`).
    description: []const u8,
    hash_length: usize,
    use_wide_string: bool = false,
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
// is the reverse of our FinalFn order.
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

fn murmur3_32Digest(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
    const data: []const u8 = if (input_len == 0) &.{} else input[0..input_len];
    const h = std.hash.Murmur3_32.hashWithSeed(data, 0);
    std.mem.writeInt(u32, digest[0..4], h, .big);
}

/// Generic typed-ctx C digests with void-returning init/update/close
/// (sph, rhash, haval, crc32c — everything sharing this ABI shape).
fn streamingEntry(
    comptime name: []const u8,
    comptime description: []const u8,
    comptime hash_length: usize,
    comptime Ctx: type,
    comptime initFn: anytype,
    comptime updateFn: anytype,
    comptime closeFn: anytype,
) HashDefinition {
    return .{
        .name = name,
        .description = description,
        .hash_length = hash_length,
        .init = @ptrCast(&initFn),
        .update = @ptrCast(&updateFn),
        .final = @ptrCast(&closeFn),
        .digest = streamingDigest(Ctx, initFn, updateFn, closeFn),
    };
}

fn opensslEntry(
    comptime name: []const u8,
    comptime description: []const u8,
    comptime hash_length: usize,
    comptime Ctx: type,
    comptime initFn: anytype,
    comptime updateFn: anytype,
    comptime finalFn: anytype,
) HashDefinition {
    return .{
        .name = name,
        .description = description,
        .hash_length = hash_length,
        .init = opensslInit(initFn),
        .update = opensslUpdate(updateFn),
        .final = opensslFinal(finalFn),
        .digest = opensslDigest(Ctx, initFn, updateFn, finalFn),
    };
}

fn zigHashEntry(comptime name: []const u8, comptime description: []const u8, comptime Hash: type) HashDefinition {
    return .{
        .name = name,
        .description = description,
        .hash_length = Hash.digest_length,
        .init = zigHashInit(Hash),
        .update = zigHashUpdate(Hash),
        .final = zigHashFinal(Hash),
        .digest = zigHashDigest(Hash),
    };
}

fn ltcEntry(
    comptime name: []const u8,
    comptime description: []const u8,
    comptime hash_length: usize,
    comptime initFn: anytype,
    comptime processFn: anytype,
    comptime doneFn: anytype,
) HashDefinition {
    return .{
        .name = name,
        .description = description,
        .hash_length = hash_length,
        .init = ltcInit(initFn),
        .update = ltcUpdate(processFn),
        .final = ltcFinal(doneFn),
        .digest = ltcDigest(initFn, processFn, doneFn),
    };
}

const Blake2b512 = std.crypto.hash.blake2.Blake2b512;
const Blake2s256 = std.crypto.hash.blake2.Blake2s256;
// Length-suffixed Blake2 variants (blake2b/blake2s keep default 512/256).
const Blake2b128 = std.crypto.hash.blake2.Blake2b128;
const Blake2b160 = std.crypto.hash.blake2.Blake2b160;
const Blake2b224 = std.crypto.hash.blake2.Blake2b(224);
const Blake2b256 = std.crypto.hash.blake2.Blake2b256;
const Blake2b384 = std.crypto.hash.blake2.Blake2b384;
const Blake2s128 = std.crypto.hash.blake2.Blake2s128;
const Blake2s160 = std.crypto.hash.blake2.Blake2s160;
const Blake2s224 = std.crypto.hash.blake2.Blake2s224;

// FIPS SHA-512/224 and SHA-512/256 (different IVs, not truncated SHA-512).
const Sha512_224 = std.crypto.hash.sha2.Sha512_224;
const Sha512_256 = std.crypto.hash.sha2.Sha512_256;

const sha3 = std.crypto.hash.sha3;
const Sha3_224 = sha3.Sha3_224;
const Sha3_256 = sha3.Sha3_256;
const Sha3_384 = sha3.Sha3_384;
const Sha3_512 = sha3.Sha3_512;
const Keccak256 = sha3.Keccak256;
const Keccak512 = sha3.Keccak512;
// std only names Keccak256/512; rhash also offered 224/384 (delim 0x01).
const Keccak224 = sha3.Keccak(1600, 224, 0x01, 24);
const Keccak384 = sha3.Keccak(1600, 384, 0x01, 24);
// SHAKE XOF: use std recommended lengths (32 / 64 bytes).
const Shake128 = sha3.Shake128;
const Shake256 = sha3.Shake256;

/// RFC 1950 Adler-32 via `std.hash.Adler32` (big-endian digest, like crc32).
const Adler32Digest = struct {
    state: std.hash.Adler32 = .{},
    pub const digest_length = 4;

    pub fn init(_: @TypeOf(.{})) Adler32Digest {
        return .{};
    }

    pub fn update(self: *Adler32Digest, data: []const u8) void {
        std.hash.Adler32.update(&self.state, data);
    }

    pub fn final(self: *Adler32Digest, out: []u8) void {
        std.mem.writeInt(u32, out[0..4], self.state.adler, .big);
    }
};

/// CRC-64 via `std.hash.crc` (big-endian digest, like crc32 / adler32).
fn Crc64Digest(comptime Crc: type) type {
    return struct {
        const Self = @This();
        state: Crc,
        pub const digest_length = 8;

        pub fn init(_: @TypeOf(.{})) Self {
            return .{ .state = Crc.init() };
        }

        pub fn update(self: *Self, data: []const u8) void {
            self.state.update(data);
        }

        pub fn final(self: *Self, out: []u8) void {
            std.mem.writeInt(u64, out[0..8], self.state.final(), .big);
        }
    };
}

const Crc64XzDigest = Crc64Digest(std.hash.crc.Crc64Xz);
const Crc64EcmaDigest = Crc64Digest(std.hash.crc.Crc64Ecma182);
const Crc64IsoDigest = Crc64Digest(std.hash.crc.Crc64GoIso);
const Crc64MsDigest = Crc64Digest(std.hash.crc.Crc64Ms);

/// xxHash via `std.hash.XxHash*` (seed 0, big-endian digest like crc / adler / xxhsum).
fn XxHashDigest(comptime H: type) type {
    return struct {
        const Self = @This();
        const Int = @TypeOf(H.hash(0, ""));
        state: H,
        pub const digest_length = @sizeOf(Int);

        pub fn init(_: @TypeOf(.{})) Self {
            return .{ .state = H.init(0) };
        }

        pub fn update(self: *Self, data: []const u8) void {
            self.state.update(data);
        }

        pub fn final(self: *Self, out: []u8) void {
            std.mem.writeInt(Int, out[0..digest_length], self.state.final(), .big);
        }
    };
}

const XxHash32Digest = XxHashDigest(std.hash.XxHash32);
const XxHash64Digest = XxHashDigest(std.hash.XxHash64);
const XxHash3Digest = XxHashDigest(std.hash.XxHash3);

/// Streaming MurmurHash3_x86_32, seed 0. std.hash.Murmur3_32 is one-shot only
/// (`hash` also uses a Murmur2 leftover seed); one-shot `digest` calls
/// `hashWithSeed` instead of this hasher.
const Murmur3_32Digest = struct {
    const Self = @This();
    const block_size = 4;
    const c1: u32 = 0xcc9e2d51;
    const c2: u32 = 0x1b873593;

    h1: u32 = 0,
    buf: [block_size]u8 = undefined,
    buf_len: u8 = 0,
    total_len: usize = 0,
    pub const digest_length = 4;

    pub fn init(_: @TypeOf(.{})) Self {
        return .{};
    }

    fn mixBlock(h1: u32, k: u32) u32 {
        var k1 = k;
        k1 *%= c1;
        k1 = std.math.rotl(u32, k1, 15);
        k1 *%= c2;
        var h = h1;
        h ^= k1;
        h = std.math.rotl(u32, h, 13);
        h *%= 5;
        h +%= 0xe6546b64;
        return h;
    }

    pub fn update(self: *Self, data: []const u8) void {
        self.total_len += data.len;
        var input = data;
        if (self.buf_len != 0) {
            const needed = block_size - self.buf_len;
            if (input.len < needed) {
                @memcpy(self.buf[self.buf_len..][0..input.len], input);
                self.buf_len += @intCast(input.len);
                return;
            }
            @memcpy(self.buf[self.buf_len..][0..needed], input[0..needed]);
            self.h1 = mixBlock(self.h1, std.mem.readInt(u32, self.buf[0..4], .little));
            self.buf_len = 0;
            input = input[needed..];
        }
        var i: usize = 0;
        while (i + block_size <= input.len) : (i += block_size) {
            self.h1 = mixBlock(self.h1, std.mem.readInt(u32, input[i..][0..4], .little));
        }
        const rest = input.len - i;
        if (rest != 0) {
            @memcpy(self.buf[0..rest], input[i..]);
            self.buf_len = @intCast(rest);
        }
    }

    pub fn final(self: *Self, out: []u8) void {
        var h1 = self.h1;
        if (self.buf_len != 0) {
            var k1: u32 = 0;
            if (self.buf_len == 3) k1 ^= @as(u32, self.buf[2]) << 16;
            if (self.buf_len >= 2) k1 ^= @as(u32, self.buf[1]) << 8;
            k1 ^= @as(u32, self.buf[0]);
            k1 *%= c1;
            k1 = std.math.rotl(u32, k1, 15);
            k1 *%= c2;
            h1 ^= k1;
        }
        h1 ^= @as(u32, @truncate(self.total_len));
        h1 ^= h1 >> 16;
        h1 *%= 0x85ebca6b;
        h1 ^= h1 >> 13;
        h1 *%= 0xc2b2ae35;
        h1 ^= h1 >> 16;
        std.mem.writeInt(u32, out[0..4], h1, .big);
    }
};

/// MurmurHash3_x64_128, seed 0 (Appleby / mmh3). Not in Zig std.
const Murmur3_128Digest = struct {
    const Self = @This();
    const block_size = 16;
    const c1: u64 = 0x87c37b91114253d5;
    const c2: u64 = 0x4cf5ad432745937f;

    h1: u64 = 0,
    h2: u64 = 0,
    buf: [block_size]u8 = undefined,
    buf_len: u8 = 0,
    total_len: usize = 0,
    pub const digest_length = 16;

    pub fn init(_: @TypeOf(.{})) Self {
        return .{};
    }

    fn mixBlock(h1: *u64, h2: *u64, block: *const [16]u8) void {
        var k1 = std.mem.readInt(u64, block[0..8], .little);
        var k2 = std.mem.readInt(u64, block[8..16], .little);
        k1 *%= c1;
        k1 = std.math.rotl(u64, k1, 31);
        k1 *%= c2;
        h1.* ^= k1;
        h1.* = std.math.rotl(u64, h1.*, 27);
        h1.* +%= h2.*;
        h1.* = h1.* *% 5 +% 0x52dce729;

        k2 *%= c2;
        k2 = std.math.rotl(u64, k2, 33);
        k2 *%= c1;
        h2.* ^= k2;
        h2.* = std.math.rotl(u64, h2.*, 31);
        h2.* +%= h1.*;
        h2.* = h2.* *% 5 +% 0x38495ab5;
    }

    fn fmix64(k0: u64) u64 {
        var k = k0;
        k ^= k >> 33;
        k *%= 0xff51afd7ed558ccd;
        k ^= k >> 33;
        k *%= 0xc4ceb9fe1a85ec53;
        k ^= k >> 33;
        return k;
    }

    pub fn update(self: *Self, data: []const u8) void {
        self.total_len += data.len;
        var input = data;
        if (self.buf_len != 0) {
            const needed = block_size - self.buf_len;
            if (input.len < needed) {
                @memcpy(self.buf[self.buf_len..][0..input.len], input);
                self.buf_len += @intCast(input.len);
                return;
            }
            @memcpy(self.buf[self.buf_len..][0..needed], input[0..needed]);
            mixBlock(&self.h1, &self.h2, self.buf[0..block_size]);
            self.buf_len = 0;
            input = input[needed..];
        }
        var i: usize = 0;
        while (i + block_size <= input.len) : (i += block_size) {
            mixBlock(&self.h1, &self.h2, input[i..][0..block_size]);
        }
        const rest = input.len - i;
        if (rest != 0) {
            @memcpy(self.buf[0..rest], input[i..]);
            self.buf_len = @intCast(rest);
        }
    }

    pub fn final(self: *Self, out: []u8) void {
        var h1 = self.h1;
        var h2 = self.h2;
        if (self.buf_len > 8) {
            var k2: u64 = 0;
            var i: usize = 8;
            while (i < self.buf_len) : (i += 1) {
                k2 ^= @as(u64, self.buf[i]) << @as(u6, @intCast(8 * (i - 8)));
            }
            k2 *%= c2;
            k2 = std.math.rotl(u64, k2, 33);
            k2 *%= c1;
            h2 ^= k2;
        }
        if (self.buf_len != 0) {
            var k1: u64 = 0;
            const n = @min(self.buf_len, @as(u8, 8));
            var i: usize = 0;
            while (i < n) : (i += 1) {
                k1 ^= @as(u64, self.buf[i]) << @as(u6, @intCast(8 * i));
            }
            k1 *%= c1;
            k1 = std.math.rotl(u64, k1, 31);
            k1 *%= c2;
            h1 ^= k1;
        }
        const len: u64 = @as(u32, @truncate(self.total_len));
        h1 ^= len;
        h2 ^= len;
        h1 +%= h2;
        h2 +%= h1;
        h1 = fmix64(h1);
        h2 = fmix64(h2);
        h1 +%= h2;
        h2 +%= h1;
        std.mem.writeInt(u128, out[0..16], @as(u128, h1) | (@as(u128, h2) << 64), .big);
    }
};

const crc32c_hashes = if (have_crc32c) [_]HashDefinition{
    streamingEntry("crc32c", "CRC-32C Castagnoli, 32-bit", c.CRC32_HASH_SIZE, c.crc32_context_t, c.crc32c_init, c.crc32c_update, c.crc32c_final),
} else [_]HashDefinition{};

pub const hashes = [_]HashDefinition{
    streamingEntry("tiger", "Tiger, 192-bit", 24, c.sph_tiger_context, c.sph_tiger_init, c.sph_tiger, c.sph_tiger_close),
    streamingEntry("tiger2", "Tiger2, 192-bit (different padding)", 24, c.sph_tiger_context, c.sph_tiger2_init, c.sph_tiger2, c.sph_tiger2_close),
    streamingEntry("md2", "MD2, 128-bit (RFC 1319)", 16, c.sph_md2_context, c.sph_md2_init, c.sph_md2, c.sph_md2_close),
    streamingEntry("md4", "MD4, 128-bit (RFC 1320)", 16, c.sph_md4_context, c.sph_md4_init, c.sph_md4, c.sph_md4_close),
    // NTLM is MD4 over UTF-16LE (wide) passwords.
    blk: {
        var e = streamingEntry("ntlm", "NTLM (MD4 of UTF-16LE password)", 16, c.sph_md4_context, c.sph_md4_init, c.sph_md4, c.sph_md4_close);
        e.use_wide_string = true;
        break :blk e;
    },
    opensslEntry("ripemd160", "RIPEMD-160, 160-bit", c.RIPEMD160_DIGEST_LENGTH, c.RIPEMD160_CTX, c.RIPEMD160_Init, c.RIPEMD160_Update, c.RIPEMD160_Final),
    streamingEntry("ripemd128", "RIPEMD-128, 128-bit", 16, c.sph_ripemd128_context, c.sph_ripemd128_init, c.sph_ripemd128, c.sph_ripemd128_close),
    .{
        .name = "blake3",
        .description = "BLAKE3, 256-bit",
        .hash_length = 32,
        .init = @ptrCast(&c.blake3_hasher_init),
        .update = @ptrCast(&c.blake3_hasher_update),
        .final = &blake3Final,
        .digest = &blake3Digest,
    },
    opensslEntry("whirlpool", "Whirlpool, 512-bit", c.WHIRLPOOL_DIGEST_LENGTH, c.WHIRLPOOL_CTX, c.WHIRLPOOL_Init, c.WHIRLPOOL_Update, c.WHIRLPOOL_Final),

    // GOST CryptoPro S-box (GENERATE_GOST_LOOKUP_TABLE not set; tables are static in gost.c).
    streamingEntry("gost", "GOST R 34.11-94 CryptoPro, 256-bit", 32, c.gost_ctx, c.rhash_gost_cryptopro_init, c.rhash_gost_update, c.rhash_gost_final),
    // GOST R 34.11-2012 (Streebog); same rhash streaming ABI as gost/edonr.
    streamingEntry("streebog256", "Streebog-256 (GOST R 34.11-2012)", 32, c.gost12_ctx, c.rhash_gost12_256_init, c.rhash_gost12_update, c.rhash_gost12_final),
    streamingEntry("streebog512", "Streebog-512 (GOST R 34.11-2012)", 64, c.gost12_ctx, c.rhash_gost12_512_init, c.rhash_gost12_update, c.rhash_gost12_final),
    streamingEntry("tth", "Tiger Tree Hash (TTH), 192-bit", 24, c.tth_ctx, c.rhash_tth_init, c.rhash_tth_update, c.rhash_tth_final),
    streamingEntry("snefru128", "Snefru-128, 8 passes", 16, c.snefru_ctx, c.rhash_snefru128_init, c.rhash_snefru_update, c.rhash_snefru_final),
    streamingEntry("snefru256", "Snefru-256, 8 passes", 32, c.snefru_ctx, c.rhash_snefru256_init, c.rhash_snefru_update, c.rhash_snefru_final),
    streamingEntry("edonr256", "EDON-R, 256-bit", 32, c.edonr_ctx, c.rhash_edonr256_init, c.rhash_edonr256_update, c.rhash_edonr256_final),
    streamingEntry("edonr512", "EDON-R, 512-bit", 64, c.edonr_ctx, c.rhash_edonr512_init, c.rhash_edonr512_update, c.rhash_edonr512_final),

    // HAVAL family (15 variants; shared sph_haval_context).
    streamingEntry("haval-128-3", "HAVAL-128, 3 passes", 16, c.sph_haval_context, c.sph_haval128_3_init, c.sph_haval128_3, c.sph_haval128_3_close),
    streamingEntry("haval-128-4", "HAVAL-128, 4 passes", 16, c.sph_haval_context, c.sph_haval128_4_init, c.sph_haval128_4, c.sph_haval128_4_close),
    streamingEntry("haval-128-5", "HAVAL-128, 5 passes", 16, c.sph_haval_context, c.sph_haval128_5_init, c.sph_haval128_5, c.sph_haval128_5_close),
    streamingEntry("haval-160-3", "HAVAL-160, 3 passes", 20, c.sph_haval_context, c.sph_haval160_3_init, c.sph_haval160_3, c.sph_haval160_3_close),
    streamingEntry("haval-160-4", "HAVAL-160, 4 passes", 20, c.sph_haval_context, c.sph_haval160_4_init, c.sph_haval160_4, c.sph_haval160_4_close),
    streamingEntry("haval-160-5", "HAVAL-160, 5 passes", 20, c.sph_haval_context, c.sph_haval160_5_init, c.sph_haval160_5, c.sph_haval160_5_close),
    streamingEntry("haval-192-3", "HAVAL-192, 3 passes", 24, c.sph_haval_context, c.sph_haval192_3_init, c.sph_haval192_3, c.sph_haval192_3_close),
    streamingEntry("haval-192-4", "HAVAL-192, 4 passes", 24, c.sph_haval_context, c.sph_haval192_4_init, c.sph_haval192_4, c.sph_haval192_4_close),
    streamingEntry("haval-192-5", "HAVAL-192, 5 passes", 24, c.sph_haval_context, c.sph_haval192_5_init, c.sph_haval192_5, c.sph_haval192_5_close),
    streamingEntry("haval-224-3", "HAVAL-224, 3 passes", 28, c.sph_haval_context, c.sph_haval224_3_init, c.sph_haval224_3, c.sph_haval224_3_close),
    streamingEntry("haval-224-4", "HAVAL-224, 4 passes", 28, c.sph_haval_context, c.sph_haval224_4_init, c.sph_haval224_4, c.sph_haval224_4_close),
    streamingEntry("haval-224-5", "HAVAL-224, 5 passes", 28, c.sph_haval_context, c.sph_haval224_5_init, c.sph_haval224_5, c.sph_haval224_5_close),
    streamingEntry("haval-256-3", "HAVAL-256, 3 passes", 32, c.sph_haval_context, c.sph_haval256_3_init, c.sph_haval256_3, c.sph_haval256_3_close),
    streamingEntry("haval-256-4", "HAVAL-256, 4 passes", 32, c.sph_haval_context, c.sph_haval256_4_init, c.sph_haval256_4, c.sph_haval256_4_close),
    streamingEntry("haval-256-5", "HAVAL-256, 5 passes", 32, c.sph_haval_context, c.sph_haval256_5_init, c.sph_haval256_5, c.sph_haval256_5_close),

    // SHA-3 / Keccak (std.crypto.hash.sha3; keccak delim 0x01).
    zigHashEntry("sha-3-224", "SHA-3-224 (FIPS 202)", Sha3_224),
    zigHashEntry("sha-3-256", "SHA-3-256 (FIPS 202)", Sha3_256),
    zigHashEntry("sha-3-384", "SHA-3-384 (FIPS 202)", Sha3_384),
    zigHashEntry("sha-3-512", "SHA-3-512 (FIPS 202)", Sha3_512),
    zigHashEntry("sha-3k-224", "Keccak-224 (non-FIPS)", Keccak224),
    zigHashEntry("sha-3k-256", "Keccak-256 (non-FIPS / Ethereum)", Keccak256),
    zigHashEntry("sha-3k-384", "Keccak-384 (non-FIPS)", Keccak384),
    zigHashEntry("sha-3k-512", "Keccak-512 (non-FIPS)", Keccak512),
    zigHashEntry("shake128", "SHAKE128 XOF, 256-bit output", Shake128),
    zigHashEntry("shake256", "SHAKE256 XOF, 512-bit output", Shake256),

    // libtomcrypt (ripemd256/320) + std blake2.
    ltcEntry("ripemd256", "RIPEMD-256, 256-bit", 32, ltc.rmd256_init, ltc.rmd256_process, ltc.rmd256_done),
    ltcEntry("ripemd320", "RIPEMD-320, 320-bit", 40, ltc.rmd320_init, ltc.rmd320_process, ltc.rmd320_done),
    zigHashEntry("blake2b", "BLAKE2b, 512-bit", Blake2b512),
    zigHashEntry("blake2b-128", "BLAKE2b, 128-bit", Blake2b128),
    zigHashEntry("blake2b-160", "BLAKE2b, 160-bit", Blake2b160),
    zigHashEntry("blake2b-224", "BLAKE2b, 224-bit", Blake2b224),
    zigHashEntry("blake2b-256", "BLAKE2b, 256-bit", Blake2b256),
    zigHashEntry("blake2b-384", "BLAKE2b, 384-bit", Blake2b384),
    zigHashEntry("blake2s", "BLAKE2s, 256-bit", Blake2s256),
    zigHashEntry("blake2s-128", "BLAKE2s, 128-bit", Blake2s128),
    zigHashEntry("blake2s-160", "BLAKE2s, 160-bit", Blake2s160),
    zigHashEntry("blake2s-224", "BLAKE2s, 224-bit", Blake2s224),

    // Adler-32 (Zig std) + CRC32 / CRC32C (srclib; CRC32C is HW on SSE4.2, soft on core2).
    // CRC-64 variants (Zig std catalog; explicit names, no bare `crc64`).
    // xxHash (Zig std; seed 0; explicit names, no bare `xxhash`).
    // MurmurHash3 (seed 0; x86-32 / x64-128; explicit names, no bare `murmur3`).
    zigHashEntry("adler32", "Adler-32 checksum (RFC 1950)", Adler32Digest),
    streamingEntry("crc32", "CRC-32 (ISO 3309 / ITU-T)", c.CRC32_HASH_SIZE, c.crc32_context_t, c.crc32_init, c.crc32_update, c.crc32_final),
    zigHashEntry("crc64-xz", "CRC-64-XZ (reflected ECMA-182)", Crc64XzDigest),
    zigHashEntry("crc64-ecma", "CRC-64-ECMA-182", Crc64EcmaDigest),
    zigHashEntry("crc64-iso", "CRC-64-ISO", Crc64IsoDigest),
    zigHashEntry("crc64-ms", "CRC-64-MS (Microsoft)", Crc64MsDigest),
    zigHashEntry("xxhash32", "xxHash32, 32-bit, seed 0 (non-cryptographic)", XxHash32Digest),
    zigHashEntry("xxhash64", "xxHash64, 64-bit, seed 0 (non-cryptographic)", XxHash64Digest),
    zigHashEntry("xxhash3", "xxHash3, 64-bit, seed 0 (non-cryptographic)", XxHash3Digest),
    blk: {
        var e = zigHashEntry("murmur3-32", "MurmurHash3 x86-32, seed 0 (non-cryptographic)", Murmur3_32Digest);
        e.digest = &murmur3_32Digest;
        break :blk e;
    },
    zigHashEntry("murmur3-128", "MurmurHash3 x64-128, seed 0 (non-cryptographic)", Murmur3_128Digest),
} ++ crc32c_hashes ++ [_]HashDefinition{
    opensslEntry("md5", "MD5, 128-bit (RFC 1321)", c.MD5_DIGEST_LENGTH, c.MD5_CTX, c.MD5_Init, c.MD5_Update, c.MD5_Final),
    opensslEntry("sha1", "SHA-1, 160-bit (FIPS 180-4)", c.SHA_DIGEST_LENGTH, c.SHA_CTX, c.SHA1_Init, c.SHA1_Update, c.SHA1_Final),
    opensslEntry("sha224", "SHA-224, 224-bit (FIPS 180-4)", c.SHA224_DIGEST_LENGTH, c.SHA256_CTX, c.SHA224_Init, c.SHA224_Update, c.SHA224_Final),
    opensslEntry("sha256", "SHA-256, 256-bit (FIPS 180-4)", c.SHA256_DIGEST_LENGTH, c.SHA256_CTX, c.SHA256_Init, c.SHA256_Update, c.SHA256_Final),
    opensslEntry("sha384", "SHA-384, 384-bit (FIPS 180-4)", c.SHA384_DIGEST_LENGTH, c.SHA512_CTX, c.SHA384_Init, c.SHA384_Update, c.SHA384_Final),
    opensslEntry("sha512", "SHA-512, 512-bit (FIPS 180-4)", c.SHA512_DIGEST_LENGTH, c.SHA512_CTX, c.SHA512_Init, c.SHA512_Update, c.SHA512_Final),
    zigHashEntry("sha512-224", "SHA-512/224 (FIPS 180-4)", Sha512_224),
    zigHashEntry("sha512-256", "SHA-512/256 (FIPS 180-4)", Sha512_256),
    // Low-level ossl_sm3_* (same ABI as MD5/SHA); avoid EVP_sm3 under BF threads.
    opensslEntry("sm3", "SM3, 256-bit (GM/T 0004)", c.SM3_DIGEST_LENGTH, c.SM3_CTX, c.ossl_sm3_init, c.ossl_sm3_update, c.ossl_sm3_final),
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

/// Digest of a string, widening to UTF-16LE first when the hash requires it
/// (`use_wide_string`).
pub fn createStringDigest(h: *const HashDefinition, input: []const u8, out: []u8, gpa: std.mem.Allocator) !void {
    if (!h.use_wide_string) return compute(h, input, out);
    const wide = try std.unicode.utf8ToUtf16LeAlloc(gpa, input);
    defer gpa.free(wide);
    compute(h, std.mem.sliceAsBytes(wide), out);
}

fn expectHash(h: *const HashDefinition, input: []const u8, expected_hex: []const u8) !void {
    var digest: [64]u8 align(8) = std.mem.zeroes([64]u8);
    compute(h, input, &digest);
    var expected: [64]u8 = std.mem.zeroes([64]u8);
    _ = try std.fmt.hexToBytes(expected[0..h.hash_length], expected_hex);
    try std.testing.expectEqualSlices(u8, expected[0..h.hash_length], digest[0..h.hash_length]);
}

test "empty via dispatch table" {
    const cases = [_]struct { name: []const u8, hex: []const u8 }{
        .{ .name = "blake2b", .hex = "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce" },
        .{ .name = "blake2b-128", .hex = "cae66941d9efbd404e4d88758ea67670" },
        .{ .name = "blake2b-160", .hex = "3345524abf6bbe1809449224b5972c41790b6cf2" },
        .{ .name = "blake2b-224", .hex = "836cc68931c2e4e3e838602eca1902591d216837bafddfe6f0c8cb07" },
        .{ .name = "blake2b-256", .hex = "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8" },
        .{ .name = "blake2b-384", .hex = "b32811423377f52d7862286ee1a72ee540524380fda1724a6f25d7978c6fd3244a6caf0498812673c5e05ef583825100" },
        .{ .name = "blake2s", .hex = "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9" },
        .{ .name = "blake2s-128", .hex = "64550d6ffe2c0a01a14aba1eade0200c" },
        .{ .name = "blake2s-160", .hex = "354c9c33f735962418bdacb9479873429c34916f" },
        .{ .name = "blake2s-224", .hex = "1fa1291e65248b37b3433475b2a0dd63d54a11ecc4e3e034e7bc1ef4" },
        .{ .name = "blake3", .hex = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262" },
        .{ .name = "adler32", .hex = "00000001" },
        .{ .name = "crc32", .hex = "00000000" },
        .{ .name = "crc64-xz", .hex = "0000000000000000" },
        .{ .name = "crc64-ecma", .hex = "0000000000000000" },
        .{ .name = "crc64-iso", .hex = "0000000000000000" },
        .{ .name = "crc64-ms", .hex = "ffffffffffffffff" },
        .{ .name = "edonr256", .hex = "86e7c84024c55dbdc9339b395c95e88db8f781719851ad1d237c6e6a8e370b80" },
        .{ .name = "edonr512", .hex = "c7afbdf3e5b4590eb0b25000bf83fb16d4f9b722ee7f9a2dc2bd382035e8ee38d6f6f15c7b8eec85355ac59af989799950c64557eab0e687d0fcbdba90ae9704" },
        .{ .name = "gost", .hex = "981e5f3ca30c841487830f84fb433e13ac1101569b9c13584ac483234cd656c0" },
        .{ .name = "streebog256", .hex = "3f539a213e97c802cc229d474c6aa32a825a360b2a933a949fd925208d9ce1bb" },
        .{ .name = "streebog512", .hex = "8e945da209aa869f0455928529bcae4679e9873ab707b55315f56ceb98bef0a7362f715528356ee83cda5f2aac4c6ad2ba3a715c1bcd81cb8e9f90bf4c1c1a8a" },
        .{ .name = "haval-128-3", .hex = "c68f39913f901f3ddf44c707357a7d70" },
        .{ .name = "haval-128-4", .hex = "ee6bbf4d6a46a679b3a856c88538bb98" },
        .{ .name = "haval-128-5", .hex = "184b8482a0c050dca54b59c7f05bf5dd" },
        .{ .name = "haval-160-3", .hex = "d353c3ae22a25401d257643836d7231a9a95f953" },
        .{ .name = "haval-160-4", .hex = "1d33aae1be4146dbaaca0b6e70d7a11f10801525" },
        .{ .name = "haval-160-5", .hex = "255158cfc1eed1a7be7c55ddd64d9790415b933b" },
        .{ .name = "haval-192-3", .hex = "e9c48d7903eaf2a91c5b350151efcb175c0fc82de2289a4e" },
        .{ .name = "haval-192-4", .hex = "4a8372945afa55c7dead800311272523ca19d42ea47b72da" },
        .{ .name = "haval-192-5", .hex = "4839d0626f95935e17ee2fc4509387bbe2cc46cb382ffe85" },
        .{ .name = "haval-224-3", .hex = "c5aae9d47bffcaaf84a8c6e7ccacd60a0dd1932be7b1a192b9214b6d" },
        .{ .name = "haval-224-4", .hex = "3e56243275b3b81561750550e36fcd676ad2f5dd9e15f2e89e6ed78e" },
        .{ .name = "haval-224-5", .hex = "4a0513c032754f5582a758d35917ac9adf3854219b39e3ac77d1837e" },
        .{ .name = "haval-256-3", .hex = "4f6938531f0bc8991f62da7bbd6f7de3fad44562b8c6f4ebf146d5b4e46f7c17" },
        .{ .name = "haval-256-4", .hex = "c92b2e23091e80e375dadce26982482d197b1a2521be82da819f8ca2c579b99b" },
        .{ .name = "haval-256-5", .hex = "be417bb4dd5cfb76c7126f4f8eeb1553a449039307b1a3cd451dbfdc0fbbe330" },
        .{ .name = "md2", .hex = "8350e5a3e24c153df2275c9f80692773" },
        .{ .name = "md4", .hex = "31d6cfe0d16ae931b73c59d7e0c089c0" },
        .{ .name = "md5", .hex = "d41d8cd98f00b204e9800998ecf8427e" },
        .{ .name = "murmur3-128", .hex = "00000000000000000000000000000000" },
        .{ .name = "murmur3-32", .hex = "00000000" },
        .{ .name = "ntlm", .hex = "31d6cfe0d16ae931b73c59d7e0c089c0" },
        .{ .name = "ripemd128", .hex = "cdf26213a150dc3ecb610f18f6b38b46" },
        .{ .name = "ripemd160", .hex = "9c1185a5c5e9fc54612808977ee8f548b2258d31" },
        .{ .name = "ripemd256", .hex = "02ba4c4e5f8ecd1877fc52d64d30e37a2d9774fb1e5d026380ae0168e3c5522d" },
        .{ .name = "ripemd320", .hex = "22d65d5661536cdc75c1fdf5c6de7b41b9f27325ebc61e8557177d705a0ec880151c3a32a00899b8" },
        .{ .name = "sha-3-224", .hex = "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7" },
        .{ .name = "sha-3-256", .hex = "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a" },
        .{ .name = "sha-3-384", .hex = "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2ac3713831264adb47fb6bd1e058d5f004" },
        .{ .name = "sha-3-512", .hex = "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26" },
        .{ .name = "sha-3k-224", .hex = "f71837502ba8e10837bdd8d365adb85591895602fc552b48b7390abd" },
        .{ .name = "sha-3k-256", .hex = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470" },
        .{ .name = "sha-3k-384", .hex = "2c23146a63a29acf99e73b88f8c24eaa7dc60aa771780ccc006afbfa8fe2479b2dd2b21362337441ac12b515911957ff" },
        .{ .name = "sha-3k-512", .hex = "0eab42de4c3ceb9235fc91acffe746b29c29a8c366b7c60e4e67c466f36a4304c00fa9caf9d87976ba469bcbe06713b435f091ef2769fb160cdab33d3670680e" },
        .{ .name = "sha1", .hex = "da39a3ee5e6b4b0d3255bfef95601890afd80709" },
        .{ .name = "sha224", .hex = "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f" },
        .{ .name = "sha256", .hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        .{ .name = "sha384", .hex = "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b" },
        .{ .name = "sha512", .hex = "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e" },
        .{ .name = "sha512-224", .hex = "6ed0dd02806fa89e25de060c19d3ac86cabb87d6a0ddd05c333b84f4" },
        .{ .name = "sha512-256", .hex = "c672b8d1ef56ed28ab87c3622c5114069bdd3ad7b8f9737498d0c01ecef0967a" },
        .{ .name = "shake128", .hex = "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26" },
        .{ .name = "shake256", .hex = "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762fd75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be" },
        .{ .name = "sm3", .hex = "1ab21d8355cfa17f8e61194831e81a8f22bec8c728fefb747ed035eb5082aa2b" },
        .{ .name = "snefru128", .hex = "8617f366566a011837f4fb4ba5bedea2" },
        .{ .name = "snefru256", .hex = "8617f366566a011837f4fb4ba5bedea2b892f3ed8b894023d16ae344b2be5881" },
        .{ .name = "tiger", .hex = "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3" },
        .{ .name = "tiger2", .hex = "4441be75f6018773c206c22745374b924aa8313fef919f41" },
        .{ .name = "tth", .hex = "5d9ed00a030e638bdb753a6a24fb900e5a63b8e73e6c25b6" },
        .{ .name = "whirlpool", .hex = "19fa61d75522a4669b44e39c1d2e1726c530232130d407f89afee0964997f7a73e83be698b288febcf88e3e03c4f0757ea8964e59b63d93708b138cc42a66eb3" },
        .{ .name = "xxhash3", .hex = "2d06800538d394c2" },
        .{ .name = "xxhash32", .hex = "02cc5d05" },
        .{ .name = "xxhash64", .hex = "ef46db3751d8e999" },
    };
    try std.testing.expectEqual(@as(usize, 74), cases.len);
    for (cases) |case| {
        errdefer std.debug.print("failed: {s}\n", .{case.name});
        try expectHash(getHash(case.name).?, "", case.hex);
    }
}

test "getHash case-insensitive" {
    try std.testing.expect(getHash("TIGER") != null);
    try std.testing.expect(getHash("Blake3") != null);
    try std.testing.expect(getHash("nope") == null);
}

test "hash count" {
    const expected: usize = if (have_crc32c) 75 else 74;
    try std.testing.expectEqual(expected, hashes.len);
}

test "every hash has a non-empty description" {
    for (hashes) |h| {
        errdefer std.debug.print("missing description: {s}\n", .{h.name});
        try std.testing.expect(h.description.len > 0);
    }
}

test "seeded hashes mention seed 0 in description" {
    const seeded = [_][]const u8{ "xxhash32", "xxhash64", "xxhash3", "murmur3-32", "murmur3-128" };
    for (seeded) |name| {
        const h = getHash(name).?;
        try std.testing.expect(std.mem.indexOf(u8, h.description, "seed 0") != null);
    }
}

test "xxhash3 fits file streaming context slot" {
    // Must stay within modes/types.zig MAX_CONTEXT_SIZE / MAX_CONTEXT_ALIGN
    // (align is CPU-dependent: 16 baseline, 32 AVX2, 64 AVX-512).
    try std.testing.expect(@sizeOf(XxHash3Digest) <= 4096);
    try std.testing.expect(@alignOf(XxHash3Digest) <= 64);
}

test "murmur3-32 matches std.hash.Murmur3_32 seed 0" {
    const samples = [_][]const u8{
        "",
        "a",
        "abc",
        "123",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog",
    };
    for (samples) |s| {
        const want = std.hash.Murmur3_32.hashWithSeed(s, 0);
        var digest: [4]u8 = undefined;
        compute(getHash("murmur3-32").?, s, &digest);
        try std.testing.expectEqual(want, std.mem.readInt(u32, &digest, .big));
    }
}

test "murmur3 streaming matches one-shot across splits" {
    const payload = "The quick brown fox jumps over the lazy dog";
    for ([_][]const u8{ "murmur3-32", "murmur3-128" }) |name| {
        const h = getHash(name).?;
        var whole: [16]u8 align(8) = std.mem.zeroes([16]u8);
        compute(h, payload, &whole);
        var split_at: usize = 0;
        while (split_at <= payload.len) : (split_at += 1) {
            var ctx: [64]u8 align(8) = undefined;
            var got: [16]u8 align(8) = std.mem.zeroes([16]u8);
            h.init(&ctx);
            if (split_at != 0) h.update(&ctx, payload.ptr, split_at);
            if (split_at != payload.len) h.update(&ctx, payload[split_at..].ptr, payload.len - split_at);
            h.final(&ctx, &got);
            try std.testing.expectEqualSlices(u8, whole[0..h.hash_length], got[0..h.hash_length]);
        }
    }
}

// Non-empty inputs exercise the update() path (the empty-string test above
// skips it). One representative per wrapper family.
test "update path: adler32 of abc" {
    try expectHash(getHash("adler32").?, "abc", "024d0127");
}

test "update path: crc64-xz of abc" {
    try expectHash(getHash("crc64-xz").?, "abc", "2cd8094a1a277627");
}

test "update path: xxhash32 of abc" {
    try expectHash(getHash("xxhash32").?, "abc", "32d153ff");
}

test "update path: murmur3-128 of abc" {
    try expectHash(getHash("murmur3-128").?, "abc", "3ba2744126ca2d52b4963f3f3fad7867");
}

test "update path: gost of abc" {
    try expectHash(getHash("gost").?, "abc", "b285056dbf18d7392d7677369524dd14747459ed8143997e163b2986f92fd42c");
}

test "update path: streebog256 of abc" {
    try expectHash(getHash("streebog256").?, "abc", "4e2919cf137ed41ec4fb6270c61826cc4fffb660341e0af3688cd0626d23b481");
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
