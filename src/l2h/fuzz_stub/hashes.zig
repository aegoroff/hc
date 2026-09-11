//! Fuzz-only `hashes` stand-in: real digests via Zig std where available,
//! no OpenSSL/hc-crypto (those + `-fno-strip` SEGVs Zig 0.16).
//!
//! Algos without a std implementation still resolve by name (full catalog) but
//! produce an all-zero digest. File/dir/restore stay in `fuzz_stub/modes.zig`.

const std = @import("std");

pub const InitFn = *const fn (context: *anyopaque) callconv(.c) void;
pub const UpdateFn = *const fn (context: *anyopaque, input: [*]const u8, len: usize) callconv(.c) void;
pub const FinalFn = *const fn (context: *anyopaque, digest: [*]u8) callconv(.c) void;
pub const DigestFn = *const fn (digest: [*]u8, input: [*]const u8, len: usize) callconv(.c) void;

pub const HashDefinition = struct {
    name: []const u8,
    description: []const u8 = "",
    hash_length: usize,
    use_wide_string: bool = false,
    init: InitFn = noopInit,
    update: UpdateFn = noopUpdate,
    final: FinalFn = noopFinal,
    digest: DigestFn = noopDigest,
};

fn noopInit(_: *anyopaque) callconv(.c) void {}
fn noopUpdate(_: *anyopaque, _: [*]const u8, _: usize) callconv(.c) void {}
fn noopFinal(_: *anyopaque, _: [*]u8) callconv(.c) void {}
fn noopDigest(digest: [*]u8, _: [*]const u8, _: usize) callconv(.c) void {
    @memset(digest[0..64], 0);
}

const crypto = std.crypto.hash;
const blake2 = crypto.blake2;
const sha2 = crypto.sha2;
const sha3 = crypto.sha3;

const Keccak224 = sha3.Keccak(1600, 224, 0x01, 24);
const Keccak384 = sha3.Keccak(1600, 384, 0x01, 24);

/// Same names/`hash_length` as production `hashes.hashes` (+ always-on `crc32c`).
const defs = [_]HashDefinition{
    .{ .name = "blake2b", .hash_length = 64 },
    .{ .name = "blake2b-128", .hash_length = 16 },
    .{ .name = "blake2b-160", .hash_length = 20 },
    .{ .name = "blake2b-224", .hash_length = 28 },
    .{ .name = "blake2b-256", .hash_length = 32 },
    .{ .name = "blake2b-384", .hash_length = 48 },
    .{ .name = "blake2s", .hash_length = 32 },
    .{ .name = "blake2s-128", .hash_length = 16 },
    .{ .name = "blake2s-160", .hash_length = 20 },
    .{ .name = "blake2s-224", .hash_length = 28 },
    .{ .name = "blake3", .hash_length = 32 },
    .{ .name = "adler32", .hash_length = 4 },
    .{ .name = "crc32", .hash_length = 4 },
    .{ .name = "crc32c", .hash_length = 4 },
    .{ .name = "crc64-xz", .hash_length = 8 },
    .{ .name = "crc64-ecma", .hash_length = 8 },
    .{ .name = "crc64-iso", .hash_length = 8 },
    .{ .name = "crc64-ms", .hash_length = 8 },
    .{ .name = "edonr256", .hash_length = 32 },
    .{ .name = "edonr512", .hash_length = 64 },
    .{ .name = "gost", .hash_length = 32 },
    .{ .name = "streebog256", .hash_length = 32 },
    .{ .name = "streebog512", .hash_length = 64 },
    .{ .name = "haval-128-3", .hash_length = 16 },
    .{ .name = "haval-128-4", .hash_length = 16 },
    .{ .name = "haval-128-5", .hash_length = 16 },
    .{ .name = "haval-160-3", .hash_length = 20 },
    .{ .name = "haval-160-4", .hash_length = 20 },
    .{ .name = "haval-160-5", .hash_length = 20 },
    .{ .name = "haval-192-3", .hash_length = 24 },
    .{ .name = "haval-192-4", .hash_length = 24 },
    .{ .name = "haval-192-5", .hash_length = 24 },
    .{ .name = "haval-224-3", .hash_length = 28 },
    .{ .name = "haval-224-4", .hash_length = 28 },
    .{ .name = "haval-224-5", .hash_length = 28 },
    .{ .name = "haval-256-3", .hash_length = 32 },
    .{ .name = "haval-256-4", .hash_length = 32 },
    .{ .name = "haval-256-5", .hash_length = 32 },
    .{ .name = "md2", .hash_length = 16 },
    .{ .name = "md4", .hash_length = 16 },
    .{ .name = "md5", .hash_length = 16 },
    .{ .name = "murmur3-128", .hash_length = 16 },
    .{ .name = "murmur3-32", .hash_length = 4 },
    .{ .name = "ntlm", .hash_length = 16, .use_wide_string = true },
    .{ .name = "ripemd128", .hash_length = 16 },
    .{ .name = "ripemd160", .hash_length = 20 },
    .{ .name = "ripemd256", .hash_length = 32 },
    .{ .name = "ripemd320", .hash_length = 40 },
    .{ .name = "sha-3-224", .hash_length = 28 },
    .{ .name = "sha-3-256", .hash_length = 32 },
    .{ .name = "sha-3-384", .hash_length = 48 },
    .{ .name = "sha-3-512", .hash_length = 64 },
    .{ .name = "sha-3k-224", .hash_length = 28 },
    .{ .name = "sha-3k-256", .hash_length = 32 },
    .{ .name = "sha-3k-384", .hash_length = 48 },
    .{ .name = "sha-3k-512", .hash_length = 64 },
    .{ .name = "sha1", .hash_length = 20 },
    .{ .name = "sha224", .hash_length = 28 },
    .{ .name = "sha256", .hash_length = 32 },
    .{ .name = "sha384", .hash_length = 48 },
    .{ .name = "sha512", .hash_length = 64 },
    .{ .name = "sha512-224", .hash_length = 28 },
    .{ .name = "sha512-256", .hash_length = 32 },
    .{ .name = "shake128", .hash_length = 32 },
    .{ .name = "shake256", .hash_length = 64 },
    .{ .name = "sm3", .hash_length = 32 },
    .{ .name = "snefru128", .hash_length = 16 },
    .{ .name = "snefru256", .hash_length = 32 },
    .{ .name = "tiger", .hash_length = 24 },
    .{ .name = "tiger2", .hash_length = 24 },
    .{ .name = "tth", .hash_length = 24 },
    .{ .name = "whirlpool", .hash_length = 64 },
    .{ .name = "xxhash3", .hash_length = 8 },
    .{ .name = "xxhash32", .hash_length = 4 },
    .{ .name = "xxhash64", .hash_length = 8 },
};

pub fn getHash(name: []const u8) ?*const HashDefinition {
    for (&defs) |*h| {
        if (std.ascii.eqlIgnoreCase(h.name, name)) return h;
    }
    return null;
}

pub fn ensureOpenSslReady() void {}

fn digestStd(comptime Hash: type, input: []const u8, out: []u8) void {
    var dig: [Hash.digest_length]u8 = undefined;
    Hash.hash(input, &dig, .{});
    @memcpy(out[0..Hash.digest_length], &dig);
}

fn digestAdler32(input: []const u8, out: []u8) void {
    std.mem.writeInt(u32, out[0..4], std.hash.Adler32.hash(input), .big);
}

fn digestCrc(comptime Crc: type, comptime Int: type, input: []const u8, out: []u8) void {
    var state = Crc.init();
    state.update(input);
    std.mem.writeInt(Int, out[0..@sizeOf(Int)], state.final(), .big);
}

fn digestXx(comptime H: type, comptime Int: type, input: []const u8, out: []u8) void {
    std.mem.writeInt(Int, out[0..@sizeOf(Int)], H.hash(0, input), .big);
}

fn digestMurmur32(input: []const u8, out: []u8) void {
    std.mem.writeInt(u32, out[0..4], std.hash.Murmur3_32.hashWithSeed(input, 0), .big);
}

/// Real digests for std-backed algos; zeros for OpenSSL/sph/rhash-only names.
pub fn compute(h: *const HashDefinition, input: []const u8, out: []u8) void {
    @memset(out[0..h.hash_length], 0);
    if (std.ascii.eqlIgnoreCase(h.name, "md5")) {
        digestStd(crypto.Md5, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha1")) {
        digestStd(crypto.Sha1, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha224")) {
        digestStd(sha2.Sha224, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha256")) {
        digestStd(sha2.Sha256, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha384")) {
        digestStd(sha2.Sha384, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha512")) {
        digestStd(sha2.Sha512, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha512-224")) {
        digestStd(sha2.Sha512_224, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha512-256")) {
        digestStd(sha2.Sha512_256, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha-3-224")) {
        digestStd(sha3.Sha3_224, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha-3-256")) {
        digestStd(sha3.Sha3_256, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha-3-384")) {
        digestStd(sha3.Sha3_384, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha-3-512")) {
        digestStd(sha3.Sha3_512, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha-3k-224")) {
        digestStd(Keccak224, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha-3k-256")) {
        digestStd(sha3.Keccak256, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha-3k-384")) {
        digestStd(Keccak384, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "sha-3k-512")) {
        digestStd(sha3.Keccak512, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "shake128")) {
        digestStd(sha3.Shake128, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "shake256")) {
        digestStd(sha3.Shake256, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2b")) {
        digestStd(blake2.Blake2b512, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2b-128")) {
        digestStd(blake2.Blake2b128, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2b-160")) {
        digestStd(blake2.Blake2b160, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2b-224")) {
        digestStd(blake2.Blake2b(224), input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2b-256")) {
        digestStd(blake2.Blake2b256, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2b-384")) {
        digestStd(blake2.Blake2b384, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2s")) {
        digestStd(blake2.Blake2s256, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2s-128")) {
        digestStd(blake2.Blake2s128, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2s-160")) {
        digestStd(blake2.Blake2s160, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake2s-224")) {
        digestStd(blake2.Blake2s224, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "blake3")) {
        digestStd(crypto.Blake3, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "adler32")) {
        digestAdler32(input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "crc32")) {
        digestCrc(std.hash.crc.Crc32, u32, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "crc64-xz")) {
        digestCrc(std.hash.crc.Crc64Xz, u64, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "crc64-ecma")) {
        digestCrc(std.hash.crc.Crc64Ecma182, u64, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "crc64-iso")) {
        digestCrc(std.hash.crc.Crc64GoIso, u64, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "crc64-ms")) {
        digestCrc(std.hash.crc.Crc64Ms, u64, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "xxhash32")) {
        digestXx(std.hash.XxHash32, u32, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "xxhash64")) {
        digestXx(std.hash.XxHash64, u64, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "xxhash3")) {
        digestXx(std.hash.XxHash3, u64, input, out);
    } else if (std.ascii.eqlIgnoreCase(h.name, "murmur3-32")) {
        digestMurmur32(input, out);
    }
    // murmur3-128 / md2 / md4 / ntlm / ripemd* / whirlpool / sm3 / gost /
    // streebog* / tiger* / tth / snefru* / edonr* / haval* / crc32c: no std
    // impl without linking C crypto — leave zeros.
}

pub fn createStringDigest(h: *const HashDefinition, input: []const u8, out: []u8, gpa: std.mem.Allocator) !void {
    if (!h.use_wide_string) {
        compute(h, input, out);
        return;
    }
    // ntlm needs MD4(UTF-16LE); no MD4 in std — validate UTF-8 then zero.
    const wide = try std.unicode.utf8ToUtf16LeAlloc(gpa, input);
    defer gpa.free(wide);
    @memset(out[0..h.hash_length], 0);
}
