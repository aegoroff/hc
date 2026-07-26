/*!
 * Callbacks required by src/srclib/bf.c (normally provided by src/hc/hash.c).
 */
#include "bf.h"
#include "lib.h"

#include <string.h>

static void (*g_digest)(apr_byte_t *digest, const void *string, const apr_size_t input_len);
static apr_size_t g_hash_len;

void bf_shim_set(void (*digest)(apr_byte_t *digest, const void *string, const apr_size_t input_len),
                 apr_size_t hash_len) {
    g_digest = digest;
    g_hash_len = hash_len;
}

void bf_shim_set_output_suspended(int suspend) {
    g_lib_output_suspended = suspend;
}

int bf_compare_hash_attempt(void *hash, const void *pass, const uint32_t length) {
    apr_byte_t attempt[64];
    g_digest(attempt, pass, (apr_size_t)length);
    return memcmp(attempt, hash, g_hash_len) == 0;
}

void *bf_create_digest(const char *hash, apr_pool_t *p) {
    apr_byte_t *result = (apr_byte_t *)apr_pcalloc(p, g_hash_len);
    lib_hex_str_2_byte_array(hash, result, g_hash_len);
    return result;
}

int bf_compare_hash(apr_byte_t *digest, const char *check_sum) {
    apr_byte_t bytes[64];
    if (g_hash_len > sizeof(bytes)) {
        return 0;
    }
    memset(bytes, 0, sizeof(bytes));
    lib_hex_str_2_byte_array(check_sum, bytes, g_hash_len);
    return memcmp(bytes, digest, g_hash_len) == 0;
}
