/*!
 * Brute-force hot loops for the Zig hc build. No APR — buffers owned by Zig.
 */
#include "bf_core.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

#ifndef MAXBYTE
#define MAXBYTE 0xFF
#endif

#define CPU_MAX_ATTEMPT_COUNT_TO_FLUSH 200000

typedef struct brute_force_ctx_t {
    const unsigned char *dict_;
    size_t dict_len_;
    void *hash_to_find_;
    bf_hash_compare_fn pfn_hash_compare_;
} brute_force_ctx_t;

static brute_force_ctx_t g_ctx;
/* Zig compiles this with clang on all targets — use builtins, not windows.h
 * Interlocked* (which pulls winuser.h and breaks zig cc / translate-c). */
static uint32_t g_already_found = 0;
static uint64_t g_attempts = 0;

static int prbf_indexofchar(const unsigned char c, int *alphabet_hash);
static void prbf_create_dict_hash(int *alphabet_hash);
static void prbf_increment_attempts(uint64_t attempts);
static void prbf_update_thread_ix(bf_cpu_ctx_t *ctx);
static BOOL prbf_make_cpu_attempt(bf_cpu_ctx_t *ctx, int *alphabet_hash);
static BOOL prbf_make_cpu_attempt_wide(bf_cpu_ctx_t *ctx, int *alphabet_hash);

void bf_core_set_context(const unsigned char *dict, size_t dict_len, void *hash_to_find,
                         bf_hash_compare_fn compare) {
    g_ctx.dict_ = dict;
    g_ctx.dict_len_ = dict_len;
    g_ctx.hash_to_find_ = hash_to_find;
    g_ctx.pfn_hash_compare_ = compare;
}

void bf_core_reset(void) {
    __atomic_store_n(&g_already_found, 0u, __ATOMIC_SEQ_CST);
    __atomic_store_n(&g_attempts, (uint64_t)0, __ATOMIC_SEQ_CST);
}

BOOL bf_core_is_found(void) {
    return __atomic_load_n(&g_already_found, __ATOMIC_SEQ_CST) != 0;
}

void bf_core_set_found(BOOL found) {
    __atomic_store_n(&g_already_found, found ? 1u : 0u, __ATOMIC_SEQ_CST);
}

uint64_t bf_core_get_attempts(void) {
    return __atomic_load_n(&g_attempts, __ATOMIC_SEQ_CST);
}

void bf_core_add_attempts(uint64_t n) {
    prbf_increment_attempts(n);
}

static void prbf_increment_attempts(uint64_t attempts) {
    __atomic_fetch_add(&g_attempts, attempts, __ATOMIC_SEQ_CST);
}

static int prbf_already_found(void) {
    return (int)__atomic_load_n(&g_already_found, __ATOMIC_SEQ_CST);
}

static void prbf_mark_found(void) {
    bf_core_set_found(TRUE);
}

void bf_core_cpu_worker(bf_cpu_ctx_t *tc) {
    int alphabet_hash[MAXBYTE + 1];

    memset(alphabet_hash, -1, (MAXBYTE + 1) * sizeof(int));
    prbf_create_dict_hash(alphabet_hash);

    if (tc->use_wide_pass_) {
        if (prbf_make_cpu_attempt_wide(tc, alphabet_hash)) {
            return;
        }
    } else {
        if (prbf_make_cpu_attempt(tc, alphabet_hash)) {
            return;
        }
    }

    tc->pass_ = NULL;
    tc->wide_pass_ = NULL;
}

void bf_core_gpu_worker(gpu_tread_ctx_t *ctx) {
    /* Max prefix slots per launch = grid capacity (blocks * threads). */
    const uint64_t max_batch =
        (uint64_t)ctx->max_gpu_blocks_number_ * (uint64_t)ctx->max_threads_per_block_;
    ctx->variants_count_ = (size_t)max_batch;
    ctx->variants_size_ = 0;
    ctx->variant_ix_ = 0;

    if (!gpu_init_pipeline(ctx)) {
        return;
    }

    ctx->gpu_context_->pfn_prepare_(ctx->device_ix_, g_ctx.dict_, g_ctx.dict_len_,
                                    (const unsigned char *)g_ctx.hash_to_find_, ctx);

    const uint32_t dict_len = (uint32_t)g_ctx.dict_len_;
    if (dict_len == 0) {
        gpu_cleanup(ctx);
        return;
    }

    /* Classic GPU model: walk prefix lengths, kernel expands last cpi chars.
     * pass_length_ is the PREFIX length; full password = prefix + cpi. */
    const uint32_t pass_min = 3;
    const uint32_t decrease = 3u - (uint32_t)ctx->max_threads_decrease_factor_;
    if (ctx->passmax_ < decrease + pass_min) {
        gpu_cleanup(ctx);
        return;
    }
    const uint32_t prefix_max = ctx->passmax_ - decrease;
    const int cpi = ctx->comparisons_per_iteration_;

    uint64_t multiplicator = dict_len;
    if (cpi == 2) {
        multiplicator *= dict_len;
    }

    for (uint32_t plen = pass_min; plen <= prefix_max; ++plen) {
        if (prbf_already_found()) {
            break;
        }

        /* total = dict_len ^ plen, with overflow → treat as "huge" and walk until found. */
        uint64_t total = 1;
        BOOL overflow = FALSE;
        for (uint32_t i = 0; i < plen; ++i) {
            if (total > UINT64_MAX / dict_len) {
                overflow = TRUE;
                break;
            }
            total *= dict_len;
        }

        uint64_t start = 0;
        while (!prbf_already_found()) {
            uint64_t remaining = overflow ? max_batch : (total - start);
            if (!overflow && start >= total) {
                break;
            }
            uint32_t count = (uint32_t)((remaining > max_batch) ? max_batch : remaining);
            if (count == 0) {
                break;
            }

            ctx->pass_length_ = plen;
            ctx->index_start_ = start;
            ctx->batch_count_ = count;

            ctx->gpu_context_->pfn_run_(ctx, g_ctx.dict_len_, NULL, 0);
            gpu_synchronize(ctx);

            /* Same accounting as classic: V + V * D^cpi per launch. */
            prbf_increment_attempts((uint64_t)count + (uint64_t)count * multiplicator);

            if (ctx->found_in_the_thread_) {
                prbf_mark_found();
                break;
            }

            start += count;
            if (overflow && start < count) {
                /* wrapped — stop this length */
                break;
            }
        }

        if (ctx->found_in_the_thread_) {
            break;
        }
    }

    gpu_cleanup(ctx);
}

static int prbf_indexofchar(const unsigned char c, int *alphabet_hash) {
    return c ? alphabet_hash[c] : -1;
}

static void prbf_create_dict_hash(int *alphabet_hash) {
    for (size_t ix = 0; ix < g_ctx.dict_len_; ix++) {
        alphabet_hash[g_ctx.dict_[ix]] = (int)ix;
    }
}

static void prbf_update_thread_ix(bf_cpu_ctx_t *ctx) {
    if (ctx->work_thread_ < ctx->num_of_threads) {
        ++ctx->work_thread_;
    } else {
        ctx->work_thread_ = 1;
    }
}

static BOOL prbf_make_cpu_attempt(bf_cpu_ctx_t *ctx, int *alphabet_hash) {
    const uint32_t pass_min = ctx->passmin_;
    const uint32_t pass_len = ctx->passmax_;
    const uint32_t dict_len = (uint32_t)g_ctx.dict_len_;
    const unsigned char *dict = g_ctx.dict_;
    unsigned char *attempt = ctx->pass_;

    for (int ti = (int)pass_len - 1, li; ti > -1; ti--) {
        for (li = prbf_indexofchar(attempt[ti], alphabet_hash) + 1; li < (int)dict_len; ++li) {
            attempt[ti] = dict[li];

            if (ctx->work_thread_ == ctx->thread_num_) {
                if (prbf_already_found()) {
                    return FALSE;
                }

                size_t skip = 0;
                while (!attempt[0]) {
                    ++skip;
                    ++attempt;
                }

                ++ctx->num_of_attempts_;

                if (ctx->num_of_attempts_ > CPU_MAX_ATTEMPT_COUNT_TO_FLUSH) {
                    prbf_increment_attempts(ctx->num_of_attempts_);
                    ctx->num_of_attempts_ = 0;
                }

                if (pass_min <= pass_len - (uint32_t)skip &&
                    g_ctx.pfn_hash_compare_(g_ctx.hash_to_find_, attempt, pass_len - (uint32_t)skip)) {
                    prbf_mark_found();
                    ctx->pass_ += skip;
                    ctx->found_in_the_thread_ = TRUE;
                    return TRUE;
                }

                attempt -= skip;
            }

            prbf_update_thread_ix(ctx);

            for (int z = ti + 1; z < (int)pass_len; ++z) {
                if (attempt[z] != dict[dict_len - 1]) {
                    ti = (int)pass_len;
                    goto outerBreak;
                }
            }
        }
    outerBreak:
        if (li == (int)dict_len) {
            attempt[ti] = dict[0];
        }
    }

    return FALSE;
}

static BOOL prbf_make_cpu_attempt_wide(bf_cpu_ctx_t *ctx, int *alphabet_hash) {
    const uint32_t pass_min = ctx->passmin_;
    const uint32_t pass_len = ctx->passmax_;
    const uint32_t dict_len = (uint32_t)g_ctx.dict_len_;
    const unsigned char *dict = g_ctx.dict_;
    bf_wide_char_t *attempt = ctx->wide_pass_;

    for (int ti = (int)pass_len - 1, li; ti > -1; ti--) {
        for (li = prbf_indexofchar((unsigned char)attempt[ti], alphabet_hash) + 1; li < (int)dict_len; ++li) {
            attempt[ti] = (bf_wide_char_t)dict[li];

            if (ctx->work_thread_ == ctx->thread_num_) {
                if (prbf_already_found()) {
                    return FALSE;
                }

                size_t skip = 0;
                while (!attempt[0]) {
                    ++skip;
                    ++attempt;
                }

                ++ctx->num_of_attempts_;

                if (ctx->num_of_attempts_ > CPU_MAX_ATTEMPT_COUNT_TO_FLUSH) {
                    prbf_increment_attempts(ctx->num_of_attempts_);
                    ctx->num_of_attempts_ = 0;
                }

                if (pass_min <= pass_len - (uint32_t)skip &&
                    g_ctx.pfn_hash_compare_(g_ctx.hash_to_find_, attempt,
                                            (pass_len - (uint32_t)skip) * (uint32_t)sizeof(bf_wide_char_t))) {
                    prbf_mark_found();
                    ctx->wide_pass_ += skip;
                    ctx->found_in_the_thread_ = TRUE;
                    return TRUE;
                }

                attempt -= skip;
            }

            prbf_update_thread_ix(ctx);

            for (uint32_t z = (uint32_t)ti + 1; z < pass_len; ++z) {
                if (attempt[z] != (bf_wide_char_t)dict[dict_len - 1]) {
                    ti = (int)pass_len;
                    goto outerBreak;
                }
            }
        }
    outerBreak:
        if (li == (int)dict_len) {
            attempt[ti] = (bf_wide_char_t)dict[0];
        }
    }

    return FALSE;
}
