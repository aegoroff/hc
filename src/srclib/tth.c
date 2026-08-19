/* tth.c - TTH (Tiger Tree Hash) over sphlib Tiger.
 *
 * Leaf:   Tiger(0x00 || up-to-1024 payload bytes)
 * Node:   Tiger(0x01 || left_digest || right_digest)
 *
 * Formerly used a private rhash Tiger (tiger_ctx) so markers could be poked
 * into its message buffer; sph_tiger exposes only init/update/close, so the
 * leaf payload is buffered here and the 0x00/0x01 prefix is fed explicitly.
 *
 * Copyright: 2007-2012 Aleksey Kravchenko <rhash.admin@gmail.com>
 * (tree logic); Tiger compression from sphlib (tiger.c).
 */

#include <string.h>
#include "tth.h"

static void tth_tiger(tth_ctx *ctx, unsigned char marker, const unsigned char *a, size_t a_len,
		      const unsigned char *b, size_t b_len, unsigned char out[TTH_HASH_LENGTH])
{
	sph_tiger_init(&ctx->tiger);
	sph_tiger(&ctx->tiger, &marker, 1);
	if (a_len)
		sph_tiger(&ctx->tiger, a, a_len);
	if (b_len)
		sph_tiger(&ctx->tiger, b, b_len);
	sph_tiger_close(&ctx->tiger, out);
}

/**
 * Initialize context before calculating hash.
 *
 * @param ctx context to initialize
 */
void rhash_tth_init(tth_ctx *ctx)
{
	ctx->leaf_len = 0;
	ctx->block_count = 0;
}

/**
 * Hash the current leaf (marker 0x00 + leaf[]) and fold it into the tree stack.
 *
 * @param ctx algorithm state
 */
static void rhash_tth_process_block(tth_ctx *ctx)
{
	uint64_t it;
	unsigned pos = 0;
	unsigned char msg[TTH_HASH_LENGTH];

	tth_tiger(ctx, 0x00, ctx->leaf, ctx->leaf_len, NULL, 0, msg);
	ctx->leaf_len = 0;

	for (it = 1; it & ctx->block_count; it <<= 1) {
		unsigned char right[TTH_HASH_LENGTH];
		memcpy(right, msg, TTH_HASH_LENGTH);
		tth_tiger(ctx, 0x01, (unsigned char *)(ctx->stack + pos), TTH_HASH_LENGTH, right,
			  TTH_HASH_LENGTH, msg);
		pos += 3;
	}
	memcpy(ctx->stack + pos, msg, TTH_HASH_LENGTH);
	ctx->block_count++;
}

/**
 * Calculate message hash.
 * Can be called repeatedly with chunks of the message to be hashed.
 *
 * @param ctx the algorithm context containing current hashing state
 * @param msg message chunk
 * @param size length of the message chunk
 */
void rhash_tth_update(tth_ctx *ctx, const unsigned char *msg, size_t size)
{
	while (size > 0) {
		size_t room = TTH_LEAF_SIZE - ctx->leaf_len;
		size_t n = size < room ? size : room;
		memcpy(ctx->leaf + ctx->leaf_len, msg, n);
		ctx->leaf_len += n;
		msg += n;
		size -= n;
		if (ctx->leaf_len == TTH_LEAF_SIZE)
			rhash_tth_process_block(ctx);
	}
}

/**
 * Store calculated hash into the given array.
 *
 * @param ctx the algorithm context containing current hashing state
 * @param result calculated hash in binary form
 */
void rhash_tth_final(tth_ctx *ctx, unsigned char result[TTH_HASH_LENGTH])
{
	uint64_t it = 1;
	unsigned pos = 0;
	unsigned char msg[TTH_HASH_LENGTH];
	const unsigned char *last_message;

	/* process the bytes left in the leaf buffer */
	if (ctx->leaf_len > 0 || ctx->block_count == 0)
		rhash_tth_process_block(ctx);

	for (; it < ctx->block_count && (it & ctx->block_count) == 0; it <<= 1)
		pos += 3;
	last_message = (unsigned char *)(ctx->stack + pos);

	for (it <<= 1; it <= ctx->block_count; it <<= 1) {
		/* merge TTH sums in the tree */
		pos += 3;
		if (it & ctx->block_count) {
			tth_tiger(ctx, 0x01, (unsigned char *)(ctx->stack + pos), TTH_HASH_LENGTH,
				  last_message, TTH_HASH_LENGTH, msg);
			last_message = msg;
		}
	}

	if (result)
		memcpy(result, last_message, TTH_HASH_LENGTH);
}
