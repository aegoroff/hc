/*!
 * Digest callbacks for bf_core (Zig hc build). No APR.
 */
#include "bf_shim.h"

#include <string.h>

static bf_digest_fn g_digest;
static size_t g_hash_len;

void bf_shim_set(bf_digest_fn digest, size_t hash_len) {
    g_digest = digest;
    g_hash_len = hash_len;
}

size_t bf_shim_hash_len(void) {
    return g_hash_len;
}

int bf_compare_hash_attempt(void *hash, const void *pass, const uint32_t length) {
    uint8_t attempt[64];
    g_digest(attempt, pass, (size_t)length);
    return memcmp(attempt, hash, g_hash_len) == 0;
}
