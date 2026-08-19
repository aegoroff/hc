#ifndef TTH_H
#define TTH_H

#include <stddef.h>
#include <stdint.h>
#include "sph_tiger.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TTH_HASH_LENGTH 24
#define TTH_LEAF_SIZE 1024

/* algorithm context */
typedef struct tth_ctx {
	sph_tiger_context tiger; /* scratch hasher for leaf / node digests */
	unsigned char leaf[TTH_LEAF_SIZE]; /* pending leaf payload (no marker) */
	size_t leaf_len; /* bytes currently in leaf[] */
	uint64_t block_count; /* number of processed leaves */
	uint64_t stack[64 * 3]; /* pending node digests (24 bytes each) */
} tth_ctx;

void rhash_tth_init(tth_ctx *ctx);
void rhash_tth_update(tth_ctx *ctx, const unsigned char *msg, size_t size);
void rhash_tth_final(tth_ctx *ctx, unsigned char result[TTH_HASH_LENGTH]);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* TTH_H */
