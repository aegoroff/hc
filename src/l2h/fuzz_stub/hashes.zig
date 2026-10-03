//! Fuzz-only `hashes` stand-in built from the shared catalog without C
//! (OpenSSL/hc-crypto + `-fno-strip` SEGVs Zig 0.16).
//!
//! Hashes with a Zig implementation give real digests; native-only ones
//! (tiger, gost, haval, ...) give all-zero digests. File/dir/restore stay in
//! `fuzz_stub/modes.zig`.

const catalog = @import("hash_catalog");

pub const HashDefinition = catalog.HashDefinition;
pub const compute = catalog.compute;
pub const createStringDigest = catalog.createStringDigest;

pub const hashes = catalog.bindStandIns();

pub fn getHash(name: []const u8) ?*const HashDefinition {
    return catalog.find(&hashes, name);
}

pub fn ensureOpenSslReady() void {}
