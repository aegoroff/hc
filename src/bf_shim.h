#ifndef HC_BF_SHIM_H_
#define HC_BF_SHIM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*bf_digest_fn)(uint8_t *digest, const void *string, size_t input_len);

void bf_shim_set(bf_digest_fn digest, size_t hash_len);

/** Digest pass and memcmp against prepared hash bytes (used by bf_core). */
int bf_compare_hash_attempt(void *hash, const void *pass, const uint32_t length);

/** Parse hex hash into caller-provided buffer of `hash_len` bytes (from bf_shim_set). */
void bf_shim_hash_to_bytes(const char *hash_hex, uint8_t *out);

/** Compare binary digest to hex checksum string. */
int bf_compare_hash(const uint8_t *digest, const char *check_sum);

size_t bf_shim_hash_len(void);

#ifdef __cplusplus
}
#endif

#endif /* HC_BF_SHIM_H_ */
