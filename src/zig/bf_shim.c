/*!
 * Digest callbacks for bf_core (Zig hc build). No APR.
 */
#include "bf_shim.h"
#include "lib.h"

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

void bf_shim_set_output_suspended(int suspend) {
    g_lib_output_suspended = suspend;
}

int bf_compare_hash_attempt(void *hash, const void *pass, const uint32_t length) {
    uint8_t attempt[64];
    g_digest(attempt, pass, (size_t)length);
    return memcmp(attempt, hash, g_hash_len) == 0;
}

void bf_shim_hash_to_bytes(const char *hash_hex, uint8_t *out) {
    lib_hex_str_2_byte_array(hash_hex, out, g_hash_len);
}

int bf_compare_hash(const uint8_t *digest, const char *check_sum) {
    uint8_t bytes[64];
    if (g_hash_len > sizeof(bytes)) {
        return 0;
    }
    memset(bytes, 0, sizeof(bytes));
    lib_hex_str_2_byte_array(check_sum, bytes, g_hash_len);
    return memcmp(bytes, digest, g_hash_len) == 0;
}
