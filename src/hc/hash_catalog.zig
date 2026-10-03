//! Hash catalog without C: the name, description and digest length of every
//! hash hc knows, plus the implementations that need nothing but Zig std.
//!
//! `hashes.zig` binds the remaining (native) hashes to their C code with
//! `bind`. The l2h fuzz build cannot link that C code and uses
//! `bindStandIns` instead, so both builds share one catalog.

const std = @import("std");
const builtin = @import("builtin");

/// CRC32C on x86/x86_64 (SSE4.2 HW or software) and aarch64 (CRC32 HW or soft).
pub const HAVE_CRC32C = switch (builtin.cpu.arch) {
    .x86_64, .x86, .aarch64 => true,
    else => false,
};

pub const InitFn = *const fn (context: *anyopaque) callconv(.c) void;
pub const UpdateFn = *const fn (context: *anyopaque, input: [*]const u8, len: usize) callconv(.c) void;
pub const FinalFn = *const fn (context: *anyopaque, digest: [*]u8) callconv(.c) void;
pub const DigestFn = *const fn (digest: [*]u8, input: [*]const u8, len: usize) callconv(.c) void;

/// Size and alignment of the on-stack context slot the file/dir streaming path
/// (`modes/file.zig` `hashFileWindow`) reuses for every algorithm. The digest
/// vtable is type-erased (`*anyopaque`), so `init`/`update`/`final` write into
/// this slot without a compiler-visible type. Every implementation
/// constructor comptime-asserts its concrete context fits here (see
/// `assertCtxFits`) so a newly added hash with an oversized or over-aligned
/// context fails to compile instead of corrupting the stack at runtime.
pub const MAX_CONTEXT_SIZE: usize = 4096;
/// Stack hashing context alignment (XxHash3 SIMD: 32 on AVX2, 64 on AVX-512).
pub const MAX_CONTEXT_ALIGN: usize = 64;

/// Compile-time guard that `Ctx` fits the shared streaming context slot.
pub fn assertCtxFits(comptime Ctx: type) void {
    comptime {
        if (@sizeOf(Ctx) > MAX_CONTEXT_SIZE) @compileError(
            @typeName(Ctx) ++ " exceeds MAX_CONTEXT_SIZE",
        );
        if (@alignOf(Ctx) > MAX_CONTEXT_ALIGN) @compileError(
            @typeName(Ctx) ++ " exceeds MAX_CONTEXT_ALIGN",
        );
    }
}

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

/// Streaming and one-shot entry points of one hash.
pub const Impl = struct {
    init: InitFn,
    update: UpdateFn,
    final: FinalFn,
    digest: DigestFn,
};

/// One catalog row.
pub const Entry = struct {
    name: []const u8,
    description: []const u8,
    hash_length: usize,
    use_wide_string: bool = false,
    /// hc binds this hash to C code by name (see `bind`).
    native: bool = false,
    /// Pure-Zig implementation. Required unless `native`; for a native hash
    /// it is a stand-in that must produce the same digest as the C code.
    zig: ?Impl = null,
};

/// C implementation of a native catalog entry.
pub const Native = struct {
    name: []const u8,
    impl: Impl,
    /// Digest length the C code reports; checked against the catalog.
    hash_length: ?usize = null,
};

/// Adapters for Zig hash types with `init(.{})` / `update` / `final`.
pub fn zigImpl(comptime Hash: type) Impl {
    assertCtxFits(Hash);
    return .{
        .init = struct {
            fn call(context: *anyopaque) callconv(.c) void {
                const hasher: *Hash = @ptrCast(@alignCast(context));
                hasher.* = Hash.init(.{});
            }
        }.call,
        .update = struct {
            fn call(context: *anyopaque, input: [*]const u8, len: usize) callconv(.c) void {
                const hasher: *Hash = @ptrCast(@alignCast(context));
                if (len != 0) hasher.update(input[0..len]);
            }
        }.call,
        .final = struct {
            fn call(context: *anyopaque, digest: [*]u8) callconv(.c) void {
                const hasher: *Hash = @ptrCast(@alignCast(context));
                hasher.final(digest[0..Hash.digest_length]);
            }
        }.call,
        .digest = struct {
            fn call(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
                var hasher = Hash.init(.{});
                if (input_len != 0) hasher.update(input[0..input_len]);
                hasher.final(digest[0..Hash.digest_length]);
            }
        }.call,
    };
}

/// All-zero digest of `len` bytes, for native hashes without a stand-in.
fn zeroImpl(comptime len: usize) Impl {
    const Zero = struct {
        fn init(_: *anyopaque) callconv(.c) void {}
        fn update(_: *anyopaque, _: [*]const u8, _: usize) callconv(.c) void {}
        fn final(_: *anyopaque, out: [*]u8) callconv(.c) void {
            @memset(out[0..len], 0);
        }
        fn oneShot(out: [*]u8, _: [*]const u8, _: usize) callconv(.c) void {
            @memset(out[0..len], 0);
        }
    };
    return .{ .init = Zero.init, .update = Zero.update, .final = Zero.final, .digest = Zero.oneShot };
}

fn define(comptime e: Entry, comptime impl: Impl) HashDefinition {
    return .{
        .name = e.name,
        .description = e.description,
        .hash_length = e.hash_length,
        .use_wide_string = e.use_wide_string,
        .init = impl.init,
        .update = impl.update,
        .final = impl.final,
        .digest = impl.digest,
    };
}

fn entryNamed(comptime name: []const u8) ?Entry {
    for (entries) |e| {
        if (std.mem.eql(u8, e.name, name)) return e;
    }
    return null;
}

fn nativeNamed(comptime natives: []const Native, comptime name: []const u8) ?Native {
    for (natives) |n| {
        if (std.mem.eql(u8, n.name, name)) return n;
    }
    return null;
}

/// Hash table with every native entry bound to its C implementation.
/// A native entry without a binding, or a binding the catalog does not list
/// as native, is a compile error.
pub fn bind(comptime natives: []const Native) [entries.len]HashDefinition {
    comptime {
        @setEvalBranchQuota(100_000);
        for (natives) |n| {
            const e = entryNamed(n.name) orelse @compileError("native hash " ++ n.name ++ " is not in the catalog");
            if (!e.native) @compileError("hash " ++ n.name ++ " is not native in the catalog");
            if (n.hash_length) |len| {
                if (len != e.hash_length) @compileError("hash " ++ n.name ++ " digest length differs from the catalog");
            }
        }
        var defs: [entries.len]HashDefinition = undefined;
        for (entries, 0..) |e, i| {
            const impl = if (e.native)
                (nativeNamed(natives, e.name) orelse @compileError("native hash " ++ e.name ++ " has no binding")).impl
            else
                e.zig.?;
            defs[i] = define(e, impl);
        }
        return defs;
    }
}

/// Hash table without C: native entries use their Zig stand-in, or an
/// all-zero digest when there is none.
pub fn bindStandIns() [entries.len]HashDefinition {
    comptime {
        @setEvalBranchQuota(100_000);
        var defs: [entries.len]HashDefinition = undefined;
        for (entries, 0..) |e, i| {
            defs[i] = define(e, e.zig orelse zeroImpl(e.hash_length));
        }
        return defs;
    }
}

/// Case-insensitive lookup in a table built by `bind` or `bindStandIns`.
pub fn find(table: []const HashDefinition, name: []const u8) ?*const HashDefinition {
    for (table) |*h| {
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

const crypto = std.crypto.hash;
const blake2 = crypto.blake2;
const sha2 = crypto.sha2;
const sha3 = crypto.sha3;

const Keccak224 = sha3.Keccak(1600, 224, 0x01, 24);
const Keccak384 = sha3.Keccak(1600, 384, 0x01, 24);

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

/// CRC via `std.hash.crc` (big-endian digest, like adler32).
fn CrcDigest(comptime Crc: type) type {
    return struct {
        const Self = @This();
        const Int = @TypeOf(Crc.init().final());
        state: Crc,
        pub const digest_length = @sizeOf(Int);

        pub fn init(_: @TypeOf(.{})) Self {
            return .{ .state = Crc.init() };
        }

        pub fn update(self: *Self, data: []const u8) void {
            self.state.update(data);
        }

        pub fn final(self: *Self, out: []u8) void {
            std.mem.writeInt(Int, out[0..digest_length], self.state.final(), .big);
        }
    };
}

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

pub const XxHash3Digest = XxHashDigest(std.hash.XxHash3);

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

fn murmur3_32Digest(digest: [*]u8, input: [*]const u8, input_len: usize) callconv(.c) void {
    const data: []const u8 = if (input_len == 0) &.{} else input[0..input_len];
    const h = std.hash.Murmur3_32.hashWithSeed(data, 0);
    std.mem.writeInt(u32, digest[0..4], h, .big);
}

fn murmur3_32Impl() Impl {
    var impl = zigImpl(Murmur3_32Digest);
    impl.digest = &murmur3_32Digest;
    return impl;
}

fn native(name: []const u8, description: []const u8, hash_length: usize) Entry {
    return .{ .name = name, .description = description, .hash_length = hash_length, .native = true };
}

/// Native hash with a Zig std stand-in that yields the same digest.
fn nativeStd(name: []const u8, description: []const u8, comptime Hash: type) Entry {
    return .{
        .name = name,
        .description = description,
        .hash_length = Hash.digest_length,
        .native = true,
        .zig = zigImpl(Hash),
    };
}

fn zig(name: []const u8, description: []const u8, comptime Hash: type) Entry {
    return .{ .name = name, .description = description, .hash_length = Hash.digest_length, .zig = zigImpl(Hash) };
}

const CRC32C_ENTRIES = if (HAVE_CRC32C) [_]Entry{
    nativeStd("crc32c", "CRC-32C Castagnoli, 32-bit", CrcDigest(std.hash.crc.@"CRC-32/ISCSI")),
} else [_]Entry{};

/// Every hash, in `hc -h` order.
pub const entries = [_]Entry{
    native("tiger", "Tiger, 192-bit", 24),
    native("tiger2", "Tiger2, 192-bit (different padding)", 24),
    native("md2", "MD2, 128-bit (RFC 1319)", 16),
    native("md4", "MD4, 128-bit (RFC 1320)", 16),
    // NTLM is MD4 over UTF-16LE (wide) passwords.
    .{ .name = "ntlm", .description = "NTLM (MD4 of UTF-16LE password)", .hash_length = 16, .use_wide_string = true, .native = true },
    native("ripemd160", "RIPEMD-160, 160-bit", 20),
    native("ripemd128", "RIPEMD-128, 128-bit", 16),
    nativeStd("blake3", "BLAKE3, 256-bit", crypto.Blake3),
    native("whirlpool", "Whirlpool, 512-bit", 64),

    native("gost", "GOST R 34.11-94 CryptoPro, 256-bit", 32),
    native("streebog256", "Streebog-256 (GOST R 34.11-2012)", 32),
    native("streebog512", "Streebog-512 (GOST R 34.11-2012)", 64),
    native("tth", "Tiger Tree Hash (TTH), 192-bit", 24),
    native("snefru128", "Snefru-128, 8 passes", 16),
    native("snefru256", "Snefru-256, 8 passes", 32),
    native("edonr256", "EDON-R, 256-bit", 32),
    native("edonr512", "EDON-R, 512-bit", 64),

    native("haval-128-3", "HAVAL-128, 3 passes", 16),
    native("haval-128-4", "HAVAL-128, 4 passes", 16),
    native("haval-128-5", "HAVAL-128, 5 passes", 16),
    native("haval-160-3", "HAVAL-160, 3 passes", 20),
    native("haval-160-4", "HAVAL-160, 4 passes", 20),
    native("haval-160-5", "HAVAL-160, 5 passes", 20),
    native("haval-192-3", "HAVAL-192, 3 passes", 24),
    native("haval-192-4", "HAVAL-192, 4 passes", 24),
    native("haval-192-5", "HAVAL-192, 5 passes", 24),
    native("haval-224-3", "HAVAL-224, 3 passes", 28),
    native("haval-224-4", "HAVAL-224, 4 passes", 28),
    native("haval-224-5", "HAVAL-224, 5 passes", 28),
    native("haval-256-3", "HAVAL-256, 3 passes", 32),
    native("haval-256-4", "HAVAL-256, 4 passes", 32),
    native("haval-256-5", "HAVAL-256, 5 passes", 32),

    // SHA-3 / Keccak (keccak delim 0x01); SHAKE XOF at the std recommended
    // lengths (32 / 64 bytes).
    zig("sha-3-224", "SHA-3-224 (FIPS 202)", sha3.Sha3_224),
    zig("sha-3-256", "SHA-3-256 (FIPS 202)", sha3.Sha3_256),
    zig("sha-3-384", "SHA-3-384 (FIPS 202)", sha3.Sha3_384),
    zig("sha-3-512", "SHA-3-512 (FIPS 202)", sha3.Sha3_512),
    zig("sha-3k-224", "Keccak-224 (non-FIPS)", Keccak224),
    zig("sha-3k-256", "Keccak-256 (non-FIPS / Ethereum)", sha3.Keccak256),
    zig("sha-3k-384", "Keccak-384 (non-FIPS)", Keccak384),
    zig("sha-3k-512", "Keccak-512 (non-FIPS)", sha3.Keccak512),
    zig("shake128", "SHAKE128 XOF, 256-bit output", sha3.Shake128),
    zig("shake256", "SHAKE256 XOF, 512-bit output", sha3.Shake256),

    native("ripemd256", "RIPEMD-256, 256-bit", 32),
    native("ripemd320", "RIPEMD-320, 320-bit", 40),
    zig("blake2b", "BLAKE2b, 512-bit", blake2.Blake2b512),
    zig("blake2b-128", "BLAKE2b, 128-bit", blake2.Blake2b128),
    zig("blake2b-160", "BLAKE2b, 160-bit", blake2.Blake2b160),
    zig("blake2b-224", "BLAKE2b, 224-bit", blake2.Blake2b(224)),
    zig("blake2b-256", "BLAKE2b, 256-bit", blake2.Blake2b256),
    zig("blake2b-384", "BLAKE2b, 384-bit", blake2.Blake2b384),
    zig("blake2s", "BLAKE2s, 256-bit", blake2.Blake2s256),
    zig("blake2s-128", "BLAKE2s, 128-bit", blake2.Blake2s128),
    zig("blake2s-160", "BLAKE2s, 160-bit", blake2.Blake2s160),
    zig("blake2s-224", "BLAKE2s, 224-bit", blake2.Blake2s224),

    // Explicit names for CRC-64, xxHash and MurmurHash3; no bare `crc64`,
    // `xxhash` or `murmur3`.
    zig("adler32", "Adler-32 checksum (RFC 1950)", Adler32Digest),
    nativeStd("crc32", "CRC-32 (ISO 3309 / ITU-T)", CrcDigest(std.hash.Crc32)),
    zig("crc64-xz", "CRC-64-XZ (reflected ECMA-182)", CrcDigest(std.hash.crc.@"CRC-64/XZ")),
    zig("crc64-ecma", "CRC-64-ECMA-182", CrcDigest(std.hash.crc.@"CRC-64/ECMA-182")),
    zig("crc64-iso", "CRC-64-ISO", CrcDigest(std.hash.crc.@"CRC-64/GO-ISO")),
    zig("crc64-ms", "CRC-64-MS (Microsoft)", CrcDigest(std.hash.crc.@"CRC-64/MS")),
    zig("xxhash32", "xxHash32, 32-bit, seed 0 (non-cryptographic)", XxHashDigest(std.hash.XxHash32)),
    zig("xxhash64", "xxHash64, 64-bit, seed 0 (non-cryptographic)", XxHashDigest(std.hash.XxHash64)),
    zig("xxhash3", "xxHash3, 64-bit, seed 0 (non-cryptographic)", XxHash3Digest),
    .{ .name = "murmur3-32", .description = "MurmurHash3 x86-32, seed 0 (non-cryptographic)", .hash_length = 4, .zig = murmur3_32Impl() },
    zig("murmur3-128", "MurmurHash3 x64-128, seed 0 (non-cryptographic)", Murmur3_128Digest),
} ++ CRC32C_ENTRIES ++ [_]Entry{
    nativeStd("md5", "MD5, 128-bit (RFC 1321)", crypto.Md5),
    nativeStd("sha1", "SHA-1, 160-bit (FIPS 180-4)", crypto.Sha1),
    nativeStd("sha224", "SHA-224, 224-bit (FIPS 180-4)", sha2.Sha224),
    nativeStd("sha256", "SHA-256, 256-bit (FIPS 180-4)", sha2.Sha256),
    nativeStd("sha384", "SHA-384, 384-bit (FIPS 180-4)", sha2.Sha384),
    nativeStd("sha512", "SHA-512, 512-bit (FIPS 180-4)", sha2.Sha512),
    zig("sha512-224", "SHA-512/224 (FIPS 180-4)", sha2.Sha512_224),
    zig("sha512-256", "SHA-512/256 (FIPS 180-4)", sha2.Sha512_256),
    native("sm3", "SM3, 256-bit (GM/T 0004)", 32),
};

test "bindStandIns covers every catalog entry" {
    // Arrange
    const table = comptime bindStandIns();

    // Act
    const md5 = find(&table, "MD5").?;
    const tiger = find(&table, "tiger").?;
    var md5_out: [16]u8 = undefined;
    var tiger_out: [24]u8 = undefined;
    compute(md5, "abc", &md5_out);
    compute(tiger, "abc", &tiger_out);

    // Assert
    try std.testing.expectEqual(entries.len, table.len);
    try std.testing.expectEqualSlices(u8, &.{ 0x90, 0x01, 0x50, 0x98, 0x3c, 0xd2, 0x4f, 0xb0, 0xd6, 0x96, 0x3f, 0x7d, 0x28, 0xe1, 0x7f, 0x72 }, &md5_out);
    try std.testing.expectEqualSlices(u8, &@as([24]u8, @splat(0)), &tiger_out);
}
