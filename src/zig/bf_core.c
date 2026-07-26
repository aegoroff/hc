/*!
 * Brute-force hot loops for the Zig hc build. No APR — buffers owned by Zig.
 */
#include "bf_core.h"

#include <limits.h>
#include <string.h>

#ifdef _MSC_VER
#include <windows.h>
#endif

#ifndef MAXBYTE
#define MAXBYTE 0xFF
#endif

#define SET_CURRENT(x) (x) + g_gpu_variant_ix *GPU_ATTEMPT_SIZE
#define CPU_MAX_ATTEMPT_COUNT_TO_FLUSH 200000

typedef struct brute_force_ctx_t {
    const unsigned char *dict_;
    size_t dict_len_;
    void *hash_to_find_;
    bf_hash_compare_fn pfn_hash_compare_;
} brute_force_ctx_t;

static brute_force_ctx_t g_ctx;
static volatile uint32_t g_already_found = 0;
static uint32_t g_gpu_variant_ix = 0;

#ifdef _MSC_VER
static uint64_t g_attempts;
#else
_Atomic static uint64_t g_attempts;
#endif

static int prbf_indexofchar(const unsigned char c, int *alphabet_hash);
static void prbf_create_dict_hash(int *alphabet_hash);
static void prbf_increment_attempts(uint64_t attempts);
static void prbf_update_thread_ix(bf_cpu_ctx_t *ctx);
static BOOL prbf_make_cpu_attempt(bf_cpu_ctx_t *ctx, int *alphabet_hash);
static BOOL prbf_make_cpu_attempt_wide(bf_cpu_ctx_t *ctx, int *alphabet_hash);
static BOOL prbf_make_gpu_attempt(gpu_tread_ctx_t *tc, int *alphabet_hash, uint32_t pass_len);
static BOOL prbf_compare_on_gpu(gpu_tread_ctx_t *ctx, const uint32_t variants_count, const uint32_t max_index);

void bf_core_set_context(const unsigned char *dict, size_t dict_len, void *hash_to_find,
                         bf_hash_compare_fn compare) {
    g_ctx.dict_ = dict;
    g_ctx.dict_len_ = dict_len;
    g_ctx.hash_to_find_ = hash_to_find;
    g_ctx.pfn_hash_compare_ = compare;
}

void bf_core_reset(void) {
    g_already_found = 0;
    g_gpu_variant_ix = 0;
#ifdef _MSC_VER
    g_attempts = 0;
#else
    g_attempts = 0;
#endif
}

BOOL bf_core_is_found(void) {
#ifdef _MSC_VER
    return (BOOL)InterlockedCompareExchange((volatile LONG *)&g_already_found, 0, 0);
#else
    return __atomic_load_n((uint32_t *)&g_already_found, __ATOMIC_SEQ_CST) != 0;
#endif
}

void bf_core_set_found(BOOL found) {
#ifdef _MSC_VER
    InterlockedExchange((volatile LONG *)&g_already_found, found ? 1 : 0);
#else
    __atomic_store_n((uint32_t *)&g_already_found, found ? 1u : 0u, __ATOMIC_SEQ_CST);
#endif
}

uint64_t bf_core_get_attempts(void) {
#ifdef _MSC_VER
    return (uint64_t)InterlockedCompareExchange64((volatile LONG64 *)&g_attempts, 0, 0);
#else
    return g_attempts;
#endif
}

void bf_core_add_attempts(uint64_t n) {
    prbf_increment_attempts(n);
}

static void prbf_increment_attempts(uint64_t attempts) {
#ifdef _MSC_VER
    InterlockedExchangeAdd64((volatile LONG64 *)&g_attempts, (LONG64)attempts);
#else
    g_attempts += attempts;
#endif
}

static int prbf_already_found(void) {
#ifdef _MSC_VER
    return (int)InterlockedCompareExchange((volatile LONG *)&g_already_found, 0, 0);
#else
    return (int)__atomic_load_n((uint32_t *)&g_already_found, __ATOMIC_SEQ_CST);
#endif
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
    ctx->variants_count_ = (size_t)ctx->max_gpu_blocks_number_ * (size_t)ctx->max_threads_per_block_;
    ctx->variants_size_ = ctx->variants_count_ * GPU_ATTEMPT_SIZE;

    /* variants_ must be allocated by the Zig orchestrator before spawn. */

    ctx->gpu_context_->pfn_prepare_(ctx->device_ix_, g_ctx.dict_, g_ctx.dict_len_,
                                    (const unsigned char *)g_ctx.hash_to_find_, ctx);

    int alphabet_hash[MAXBYTE + 1];
    memset(alphabet_hash, -1, (MAXBYTE + 1) * sizeof(int));
    prbf_create_dict_hash(alphabet_hash);

    uint32_t pass_min = 3;
    uint32_t decrease = 3 - (uint32_t)ctx->max_threads_decrease_factor_;
    uint32_t pass_len = ctx->passmax_ - decrease;

    for (uint32_t i = pass_min; i <= pass_len; ++i) {
        if (prbf_make_gpu_attempt(ctx, alphabet_hash, i)) {
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

static BOOL prbf_make_gpu_attempt(gpu_tread_ctx_t *ctx, int *alphabet_hash, uint32_t pass_len) {
    unsigned char *current = SET_CURRENT(ctx->variants_);
#if (defined(__STDC_LIB_EXT1__) && defined(__STDC_WANT_LIB_EXT1__)) ||                                                 \
    (defined(__STDC_SECURE_LIB__) && defined(__STDC_WANT_SECURE_LIB__))
    const size_t variants_size_in_bytes = ctx->variants_size_ * sizeof(unsigned char);
#endif

    const uint32_t dict_len = (uint32_t)g_ctx.dict_len_;
    const uint32_t variants_count = (uint32_t)ctx->variants_count_;
    const uint32_t max_index = variants_count - 1;
    const unsigned char *dict = g_ctx.dict_;
    unsigned char *attempt = ctx->attempt_;

    for (int ti = (int)pass_len - 1, li; ti > -1; ti--) {
        for (li = prbf_indexofchar(attempt[ti], alphabet_hash) + 1; li < (int)dict_len; ++li) {
            attempt[ti] = dict[li];

            if (!attempt[0]) {
                goto skip_attempt;
            }

#if (defined(__STDC_LIB_EXT1__) && defined(__STDC_WANT_LIB_EXT1__)) ||                                                 \
    (defined(__STDC_SECURE_LIB__) && defined(__STDC_WANT_SECURE_LIB__))
            const errno_t err = memcpy_s(current, variants_size_in_bytes, attempt, pass_len);
            if (err) {
                return FALSE;
            }
#else
            memcpy(current, attempt, pass_len);
#endif

            if (prbf_compare_on_gpu(ctx, variants_count, max_index)) {
                return TRUE;
            }

            current = SET_CURRENT(ctx->variants_);
        skip_attempt:
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

static BOOL prbf_compare_on_gpu(gpu_tread_ctx_t *ctx, const uint32_t variants_count, const uint32_t max_index) {
    if (g_gpu_variant_ix < max_index) {
        ++g_gpu_variant_ix;
    } else {
        g_gpu_variant_ix = 0;
        if (prbf_already_found()) {
            return TRUE;
        }

        ctx->gpu_context_->pfn_run_(ctx, g_ctx.dict_len_, ctx->variants_, ctx->variants_size_);

        uint64_t multiplicator = g_ctx.dict_len_;
        if (ctx->comparisons_per_iteration_ == 2) {
            multiplicator *= multiplicator;
        }
        uint64_t attempts_in_iteration = variants_count + variants_count * multiplicator;

        prbf_increment_attempts(attempts_in_iteration);

        if (ctx->found_in_the_thread_) {
            prbf_mark_found();
            return TRUE;
        }

        memset(ctx->variants_, 0, ctx->variants_size_ * sizeof(unsigned char));
    }
    return FALSE;
}
