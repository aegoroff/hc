/*!
 * \brief   The file contains brute force algorithm implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2025
 */

#include "bf.h"
#include "apr_atomic.h"
#include "apr_strings.h"
#include "apr_thread_proc.h"
#include "encoding.h"
#include "gpu.h"
#include "intl.h"
#include "output.h"
#include <math.h>
#ifndef _MSC_VER
#include <ctype.h>
#include <string.h>
#include <uchar.h>
char *_strrev(char *str);
#endif

/*
    bf_ - public members
    prbf_ - private members
    g_ - global data
*/

#define SET_CURRENT(x) (x) + g_gpu_variant_ix *GPU_ATTEMPT_SIZE
#define CPU_MAX_ATTEMPT_COUNT_TO_FLUSH 200000

#ifndef MAXBYTE
#define MAXBYTE 0xFF
#endif

#ifdef _MSC_VER
#define WIDE_CHAR wchar_t
#else
#define WIDE_CHAR char16_t
#endif

typedef struct brute_force_ctx_t {
    const unsigned char *dict_;
    size_t dict_len_;
    void *hash_to_find_;
    int (*pfn_hash_compare_)(void *hash, const void *pass, const uint32_t length);
} brute_force_ctx_t;

typedef struct tread_ctx_t {
    unsigned char *pass_;
    WIDE_CHAR *wide_pass_;
    size_t *chars_indexes_;
    uint64_t num_of_attempts_;
    size_t thread_num_;
    uint32_t passmin_;
    uint32_t passmax_;
    uint32_t pass_length_;
    uint32_t work_thread_;
    uint32_t num_of_threads;
    BOOL use_wide_pass_;
    BOOL found_in_the_thread_;
} tread_ctx_t;

static brute_force_ctx_t *g_brute_force_ctx;
static char *alphabet = DIGITS LOW_CASE UPPER_CASE;
static volatile apr_uint32_t g_already_found = FALSE;
static uint32_t g_gpu_variant_ix = 0;
static const unsigned char k_ascii_first = '!';
static const unsigned char k_ascii_last = '~';

#ifdef _MSC_VER
static uint64_t g_attempts;
#else
_Atomic static uint64_t g_attempts;
#endif

static const unsigned char *prbf_prepare_dictionary(const unsigned char *dict, apr_pool_t *pool);
static void *APR_THREAD_FUNC prbf_cpu_thread_func(apr_thread_t *thd, void *data);
static void *APR_THREAD_FUNC prbf_gpu_thread_func(apr_thread_t *thd, void *data);
static char *prbf_commify(char *numstr, apr_pool_t *pool);
static char *prbf_double_to_string(double value, apr_pool_t *pool);
static char *prbf_int64_to_string(uint64_t value, apr_pool_t *pool);
static const unsigned char *prbf_str_replace(const unsigned char *orig, const char *rep, const char *with,
                                             apr_pool_t *pool);
static BOOL prbf_make_gpu_attempt(gpu_tread_ctx_t *tc, int *alphabet_hash, uint32_t pass_len);
static BOOL prbf_compare_on_gpu(gpu_tread_ctx_t *ctx, const uint32_t variants_count, const uint32_t max_index);
static BOOL prbf_make_cpu_attempt(tread_ctx_t *ctx, int *alphabet_hash);
static BOOL prbf_make_cpu_attempt_wide(tread_ctx_t *ctx, int *alphabet_hash);
static void prbf_update_thread_ix(tread_ctx_t *ctx);
static int prbf_indexofchar(const unsigned char c, int *alphabet_hash);
static void prbf_create_dict_hash(int *alphabet_hash);
static void prbf_increment_attempts(uint64_t attempts);

void bf_crack_hash(const char *dict, const char *hash, const uint32_t passmin, uint32_t passmax, apr_size_t hash_length,
                   void (*pfn_digest_function)(apr_byte_t *digest, const void *string, const apr_size_t input_len),
                   const BOOL no_probe, const uint32_t num_of_threads, const BOOL use_wide_pass,
                   const BOOL has_gpu_implementation, gpu_context_t *gpu_context, apr_pool_t *pool) {
    char *str;

    apr_byte_t *digest = (apr_byte_t *)apr_pcalloc(pool, hash_length);

    lib_time_t time = {0};

    g_attempts = 0;

    // Empty string validation
    pfn_digest_function(digest, "", 0);

    passmax = passmax ? passmax : MAX_DEFAULT;

    if (bf_compare_hash(digest, hash)) {
        str = _("Empty string");
        lib_start_timer();
    } else {
        // Probing
        if (!no_probe) {
            const char *t = "123";
            const size_t max_time_msg_size = 63;
            if (use_wide_pass) {
#ifdef _MSC_VER
                wchar_t *s = enc_from_ansi_to_unicode(t, pool);
                pfn_digest_function(digest, s, wcslen(s) * sizeof(wchar_t));
#else
                char16_t* s = enc_from_ansi_to_wide_chars(t, pool);
                pfn_digest_function(digest, s, strlen(t) * sizeof(char16_t));
#endif
            } else {
                pfn_digest_function(digest, t, strlen(t));
            }

            const char *str123 = out_hash_to_string(digest, FALSE, hash_length, pool);

            lib_start_timer();

            bf_brute_force(1, MAX_DEFAULT, alphabet, str123, bf_create_digest, num_of_threads, use_wide_pass, FALSE,
                           gpu_context, pool);

            lib_stop_timer();
            time = lib_read_elapsed_time();
            const double ratio = g_attempts / time.seconds;

            g_attempts = 0;

            const size_t max_count = 256;
#ifdef __STDC_WANT_SECURE_LIB__
            const size_t len = strnlen_s(prbf_prepare_dictionary(dict, pool), max_count);
#else
            const size_t len = strnlen(prbf_prepare_dictionary(dict, pool), max_count);
#endif
            const double max_attempts = pow(len, passmax);
            lib_time_t max_time = lib_normalize_time(max_attempts / ratio);
            char *max_time_msg = (char *)apr_pcalloc(pool, max_time_msg_size + 1);
            lib_time_to_string(&max_time, max_time_msg);
            lib_printf(_("May take approximatelly: %s (%s attempts)"), max_time_msg,
                       prbf_double_to_string(max_attempts, pool));
        }

        // Main run
        lib_start_timer();
        str = bf_brute_force(passmin, passmax, dict, hash, bf_create_digest, num_of_threads, use_wide_pass,
                             has_gpu_implementation, gpu_context, pool);
    }

    bf_output_timings(pool);

    if (str != NULL) {
        char *ansi = enc_from_utf8_to_ansi(str, pool);
        lib_printf(_("Initial string is: %s"), ansi == NULL ? str : ansi);
    } else {
        lib_printf(_("Nothing found"));
    }
    lib_new_line();
}

void bf_output_timings(apr_pool_t *pool) {
    lib_stop_timer();

    lib_time_t time = lib_read_elapsed_time();
    const double speed = g_attempts > 0 && time.total_seconds > 0 ? g_attempts / time.total_seconds : 0;
    char *speed_str = prbf_double_to_string(speed, pool);
    lib_new_line();
    lib_printf(_("Attempts: %s Time "), prbf_int64_to_string(g_attempts, pool));
    lib_printf(FULL_TIME_FMT, time.hours, time.minutes, time.seconds);
    lib_printf(_(" Speed: %s attempts/second"), speed_str);
    lib_new_line();
}

char *bf_brute_force(const uint32_t passmin, const uint32_t passmax, const char *dict, const char *hash,
                     void *(*pfn_hash_prepare)(const char *h, apr_pool_t *pool), uint32_t num_of_threads,
                     const BOOL use_wide_pass, BOOL has_gpu_implementation, gpu_context_t *gpu_context,
                     apr_pool_t *pool) {
    apr_threadattr_t *thd_attr = NULL;
    apr_status_t rv;
    size_t i = 0;
    unsigned char *pass = NULL;

    g_already_found = FALSE;

    if (passmax > INT_MAX / sizeof(int)) {
        lib_printf(_("Max string length is too big: %lu"), passmax);
        return NULL;
    }

    /* If max password length less then 4 GPU not needed */
    has_gpu_implementation = has_gpu_implementation && passmax > 3;
    if (has_gpu_implementation) {
        if (!gpu_can_use_gpu()) {
            gpu_versions_t driver_ver = gpu_number_to_version(gpu_driver_version());
            gpu_versions_t runtime_ver = gpu_number_to_version(gpu_runtime_version());
            if (driver_ver.major > 0) {
                lib_printf(
                    "\nGPU present but driver's CUDA version %lu.%lu less then required %lu.%lu. So use only CPU",
                    driver_ver.major, driver_ver.minor, runtime_ver.major, runtime_ver.minor);
            }

            has_gpu_implementation = false;
        }
    }

    /* If has GPU implementation the max performance mode if one CPU thread */
    if (has_gpu_implementation) {
        num_of_threads = 1;
    }

    device_props_t *gpu_props = (device_props_t *)apr_pcalloc(pool, sizeof(device_props_t));

    g_brute_force_ctx = (brute_force_ctx_t *)apr_pcalloc(pool, sizeof(brute_force_ctx_t));
    g_brute_force_ctx->hash_to_find_ = pfn_hash_prepare(hash, pool);
    g_brute_force_ctx->pfn_hash_compare_ = bf_compare_hash_attempt;
    g_brute_force_ctx->dict_ = prbf_prepare_dictionary(dict, pool);
    g_brute_force_ctx->dict_len_ = strlen(g_brute_force_ctx->dict_);

    apr_thread_t **thd_arr = (apr_thread_t **)apr_pcalloc(pool, sizeof(apr_thread_t *) * num_of_threads);
    tread_ctx_t **thd_ctx = (tread_ctx_t **)apr_pcalloc(pool, sizeof(tread_ctx_t *) * num_of_threads);

    /* The default thread attribute: detachable */
    apr_threadattr_create(&thd_attr, pool);

    if (g_brute_force_ctx->dict_len_ <= num_of_threads) {
        num_of_threads = g_brute_force_ctx->dict_len_;
    }

    for (; i < num_of_threads; ++i) {
        thd_ctx[i] = (tread_ctx_t *)apr_pcalloc(pool, sizeof(tread_ctx_t));
        thd_ctx[i]->passmin_ = passmin;
        thd_ctx[i]->passmax_ = passmax;
        thd_ctx[i]->work_thread_ = 1;
        thd_ctx[i]->thread_num_ = i + 1;
        thd_ctx[i]->pass_ = (unsigned char *)apr_pcalloc(pool, sizeof(unsigned char) * ((size_t)passmax + 1));
        thd_ctx[i]->wide_pass_ = (WIDE_CHAR *)apr_pcalloc(pool, sizeof(WIDE_CHAR) * ((size_t)passmax + 1));
        thd_ctx[i]->chars_indexes_ = (size_t *)apr_pcalloc(pool, (size_t)passmax * sizeof(size_t));
        thd_ctx[i]->pass_length_ = passmin;
        thd_ctx[i]->num_of_threads = num_of_threads;
        thd_ctx[i]->use_wide_pass_ = use_wide_pass;
        rv = apr_thread_create(&thd_arr[i], thd_attr, prbf_cpu_thread_func, thd_ctx[i], pool);
    }

    if (has_gpu_implementation) {
        gpu_get_props(gpu_props);

        if (!gpu_props->device_count) {
            goto wait_cpu_threads;
        }

        const uint32_t num_of_gpu_threads = gpu_props->device_count;
        gpu_tread_ctx_t **gpu_thd_ctx =
            (gpu_tread_ctx_t **)apr_pcalloc(pool, sizeof(gpu_tread_ctx_t *) * num_of_gpu_threads);
        apr_thread_t **gpu_thd_arr = (apr_thread_t **)apr_pcalloc(pool, sizeof(apr_thread_t *) * num_of_gpu_threads);

        for (i = 0; i < num_of_gpu_threads; ++i) {
            gpu_thd_ctx[i] = (gpu_tread_ctx_t *)apr_pcalloc(pool, sizeof(gpu_tread_ctx_t));
            gpu_thd_ctx[i]->passmin_ = passmin;
            gpu_thd_ctx[i]->passmax_ = passmax;
            gpu_thd_ctx[i]->attempt_ = (unsigned char *)apr_pcalloc(pool, sizeof(unsigned char) * GPU_ATTEMPT_SIZE);
            gpu_thd_ctx[i]->result_ = (unsigned char *)apr_pcalloc(pool, sizeof(unsigned char) * GPU_ATTEMPT_SIZE);
            gpu_thd_ctx[i]->pass_length_ = passmin;
            gpu_thd_ctx[i]->max_gpu_blocks_number_ = gpu_props->max_blocks_number;
            gpu_thd_ctx[i]->multiprocessor_count_ = gpu_props->multiprocessor_count;
            gpu_thd_ctx[i]->max_threads_per_block_ =
                gpu_props->max_threads_per_block / gpu_context->max_threads_decrease_factor_;
            gpu_thd_ctx[i]->device_ix_ = i;
            gpu_thd_ctx[i]->gpu_context_ = gpu_context;
            gpu_thd_ctx[i]->use_wide_pass_ = use_wide_pass;
            gpu_thd_ctx[i]->max_threads_decrease_factor_ = gpu_context->max_threads_decrease_factor_;
            gpu_thd_ctx[i]->comparisons_per_iteration_ = gpu_context->comparisons_per_iteration_;
            gpu_thd_ctx[i]->pool_ = pool;
            rv = apr_thread_create(&gpu_thd_arr[i], thd_attr, prbf_gpu_thread_func, gpu_thd_ctx[i], pool);
        }

        for (i = 0; i < num_of_gpu_threads; ++i) {
            rv = apr_thread_join(&rv, gpu_thd_arr[i]);

            if (gpu_thd_ctx[i]->found_in_the_thread_) {
                pass = gpu_thd_ctx[i]->result_;
            }
        }

        // Stop CPU threads even if they found nothing
        apr_atomic_set32(&g_already_found, TRUE);
    }
wait_cpu_threads:
    for (i = 0; i < num_of_threads; ++i) {
        rv = apr_thread_join(&rv, thd_arr[i]);

        g_attempts += thd_ctx[i]->num_of_attempts_;

        if (thd_ctx[i]->use_wide_pass_) {
            if (thd_ctx[i]->wide_pass_ != NULL) {
#ifdef _MSC_VER
                pass = (unsigned char *)enc_from_unicode_to_ansi(thd_ctx[i]->wide_pass_, pool);
#else
                pass = (unsigned char *)enc_wide_chars_to_ansi(thd_ctx[i]->wide_pass_, pool);
#endif
            }
        } else {
            if (thd_ctx[i]->pass_ != NULL) {
                pass = thd_ctx[i]->pass_;
            }
        }
    }

    return (char *)pass;
}

static int prbf_indexofchar(const unsigned char c, int *alphabet_hash) { return c ? alphabet_hash[c] : -1; }

static void prbf_create_dict_hash(int *alphabet_hash) {
    // fill ABC hash
    for (size_t ix = 0; ix < g_brute_force_ctx->dict_len_; ix++) {
        alphabet_hash[g_brute_force_ctx->dict_[ix]] = ix;
    }
}

static void prbf_increment_attempts(uint64_t attempts) {
#ifdef _MSC_VER
    InterlockedExchangeAdd64(&g_attempts, attempts);
#else
    // Bad but APR don't implement atomic 64 bit functions
    g_attempts += attempts;
#endif
}

/**
 * CPU thread entry point
 */
void *APR_THREAD_FUNC prbf_cpu_thread_func(apr_thread_t *thd, void *data) {
    tread_ctx_t *tc = (tread_ctx_t *)data;

    int alphabet_hash[MAXBYTE + 1];

    memset(alphabet_hash, -1, (MAXBYTE + 1) * sizeof(int));

    prbf_create_dict_hash(alphabet_hash);

    if (tc->use_wide_pass_) {
        if (prbf_make_cpu_attempt_wide(tc, alphabet_hash)) {
            goto result;
        }
    } else {
        if (prbf_make_cpu_attempt(tc, alphabet_hash)) {
            goto result;
        }
    }

    tc->pass_ = NULL;
    tc->wide_pass_ = NULL;
result:
    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}

/**
 * GPU thread entry point
 */
void *APR_THREAD_FUNC prbf_gpu_thread_func(apr_thread_t *thd, void *data) {
    gpu_tread_ctx_t *ctx = (gpu_tread_ctx_t *)data;

    ctx->variants_count_ = ctx->max_gpu_blocks_number_ * ctx->max_threads_per_block_;
    ctx->variants_size_ = ctx->variants_count_ * GPU_ATTEMPT_SIZE;

    size_t variants_size_in_bytes = ctx->variants_size_ * sizeof(unsigned char);
    ctx->variants_ = (unsigned char *)apr_pcalloc(ctx->pool_, variants_size_in_bytes);

    ctx->gpu_context_->pfn_prepare_(ctx->device_ix_, g_brute_force_ctx->dict_, g_brute_force_ctx->dict_len_,
                                    g_brute_force_ctx->hash_to_find_, ctx);

    int alphabet_hash[MAXBYTE + 1];

    memset(alphabet_hash, -1, (MAXBYTE + 1) * sizeof(int));

    prbf_create_dict_hash(alphabet_hash);

    uint32_t pass_min = 3;
    uint32_t decrease = 3 - ctx->max_threads_decrease_factor_;

    // The max length passmax munus decrease the last decrease numbers will be added within GPU kernel
    uint32_t pass_len = ctx->passmax_ - decrease;

    for (uint32_t i = pass_min; i <= pass_len; ++i) {
        if (prbf_make_gpu_attempt(ctx, alphabet_hash, i)) {
            break;
        }
    }

    gpu_cleanup(ctx);

    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}

BOOL prbf_make_gpu_attempt(gpu_tread_ctx_t *ctx, int *alphabet_hash, uint32_t pass_len) {
    unsigned char *current = SET_CURRENT(ctx->variants_);
#if (defined(__STDC_LIB_EXT1__) && defined(__STDC_WANT_LIB_EXT1__)) ||                                                 \
    (defined(__STDC_SECURE_LIB__) && defined(__STDC_WANT_SECURE_LIB__))
    const size_t variants_size_in_bytes = ctx->variants_size_ * sizeof(unsigned char);
#endif

    const uint32_t dict_len = g_brute_force_ctx->dict_len_;
    const uint32_t variants_count = ctx->variants_count_;
    const uint32_t max_index = variants_count - 1;
    const unsigned char *dict = g_brute_force_ctx->dict_;
    unsigned char *attempt = ctx->attempt_;

    // ti - text index (on probing it's the index of the attempt's last char)
    // li - ABC index

    // start rotating chars from the back and forward

    for (int ti = pass_len - 1, li; ti > -1; ti--) {
        for (li = prbf_indexofchar(attempt[ti], alphabet_hash) + 1; li < dict_len; ++li) {
            attempt[ti] = dict[li];

            if (!attempt[0]) {
                goto skip_attempt;
            }

            // Probe attempt

            // Copy variant
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
            // rotate to the right
            for (int z = ti + 1; z < pass_len; ++z) {
                if (attempt[z] != dict[dict_len - 1]) {
                    ti = pass_len;
                    goto outerBreak;
                }
            }
        }
    outerBreak:
        if (li == dict_len) {
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
        if (apr_atomic_read32(&g_already_found)) {
            return TRUE;
        }

        // run real GPU code here
        ctx->gpu_context_->pfn_run_(ctx, g_brute_force_ctx->dict_len_, ctx->variants_, ctx->variants_size_);

        uint64_t multiplicator = g_brute_force_ctx->dict_len_;
        if (ctx->comparisons_per_iteration_ == 2) {
            multiplicator *= multiplicator;
        }
        uint64_t attempts_in_iteration = variants_count + variants_count * multiplicator;

        prbf_increment_attempts(attempts_in_iteration);

        if (ctx->found_in_the_thread_) {
            apr_atomic_set32(&g_already_found, TRUE);
            return TRUE;
        }

        memset(ctx->variants_, 0, ctx->variants_size_ * sizeof(unsigned char));
    }
    return FALSE;
}

static void prbf_update_thread_ix(tread_ctx_t *ctx) {
    if (ctx->work_thread_ < ctx->num_of_threads) {
        ++ctx->work_thread_;
    } else {
        ctx->work_thread_ = 1;
    }
}

BOOL prbf_make_cpu_attempt(tread_ctx_t *ctx, int *alphabet_hash) {
    const uint32_t pass_min = ctx->passmin_;
    const uint32_t pass_len = ctx->passmax_;
    const uint32_t dict_len = g_brute_force_ctx->dict_len_;
    const unsigned char *dict = g_brute_force_ctx->dict_;
    unsigned char *attempt = ctx->pass_;

    // ti - text index (on probing it's the index of the attempt's last char)
    // li - ABC index

    // start rotating chars from the back and forward
    for (int ti = pass_len - 1, li; ti > -1; ti--) {
        for (li = prbf_indexofchar(attempt[ti], alphabet_hash) + 1; li < dict_len; ++li) {
            attempt[ti] = dict[li];

            // Probe attempt
            if (ctx->work_thread_ == ctx->thread_num_) {
                if (apr_atomic_read32(&g_already_found)) {
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

                if (pass_min <= pass_len - skip &&
                    g_brute_force_ctx->pfn_hash_compare_(g_brute_force_ctx->hash_to_find_, attempt, pass_len - skip)) {
                    apr_atomic_set32(&g_already_found, TRUE);
                    ctx->pass_ += skip;
                    return TRUE;
                }

                attempt -= skip;
            }

            prbf_update_thread_ix(ctx);

            // rotate to the right
            for (int z = ti + 1; z < pass_len; ++z) {
                if (attempt[z] != dict[dict_len - 1]) {
                    ti = pass_len;
                    goto outerBreak;
                }
            }
        }
    outerBreak:
        if (li == dict_len) {
            attempt[ti] = dict[0];
        }
    }

    return FALSE;
}

BOOL prbf_make_cpu_attempt_wide(tread_ctx_t *ctx, int *alphabet_hash) {
    const uint32_t pass_min = ctx->passmin_;
    const uint32_t pass_len = ctx->passmax_;
    const uint32_t dict_len = g_brute_force_ctx->dict_len_;
    const unsigned char *dict = g_brute_force_ctx->dict_;
    WIDE_CHAR *attempt = ctx->wide_pass_;

    // ti - text index (on probing it's the index of the attempt's last char)
    // li - ABC index

    // start rotating chars from the back and forward
    for (int ti = pass_len - 1, li; ti > -1; ti--) {
        for (li = prbf_indexofchar(attempt[ti], alphabet_hash) + 1; li < dict_len; ++li) {
            attempt[ti] = dict[li];

            // Probe attempt
            if (ctx->work_thread_ == ctx->thread_num_) {
                if (apr_atomic_read32(&g_already_found)) {
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

                if (pass_min <= pass_len - skip &&
                    g_brute_force_ctx->pfn_hash_compare_(g_brute_force_ctx->hash_to_find_, attempt,
                                                         (pass_len - skip) * sizeof(WIDE_CHAR))) {
                    apr_atomic_set32(&g_already_found, TRUE);
                    ctx->wide_pass_ += skip;
                    return TRUE;
                }

                attempt -= skip;
            }

            prbf_update_thread_ix(ctx);

            // rotate to the right
            for (uint32_t z = ti + 1; z < pass_len; ++z) {
                if (attempt[z] != dict[dict_len - 1]) {
                    ti = pass_len;
                    goto outerBreak;
                }
            }
        }
    outerBreak:
        if (li == dict_len) {
            attempt[ti] = dict[0];
        }
    }

    return FALSE;
}

const unsigned char *prbf_prepare_dictionary(const unsigned char *dict, apr_pool_t *pool) {
    const char *digits_class = strstr((char *)dict, DIGITS_TPL);
    const char *low_case_class = strstr((char *)dict, LOW_CASE_TPL);
    const char *upper_case_class = strstr((char *)dict, UPPER_CASE_TPL);
    const char *all_ascii_class = strstr((char *)dict, ASCII_TPL);
    const unsigned char *result = (const unsigned char *)dict;
    size_t dict_len;
    int chars_hash[MAXBYTE + 1];

    memset(chars_hash, 0, (MAXBYTE + 1) * sizeof(int));

    if (all_ascii_class) {
        dict_len = (k_ascii_last - k_ascii_first) + 1;
        unsigned char *tmp = (unsigned char *)apr_pcalloc(pool, (dict_len + 1) * sizeof(unsigned char));
        size_t i = 0;
        for (unsigned char sym = k_ascii_first; sym <= k_ascii_last; sym++) {
            tmp[i++] = sym;
        }
        result = tmp;
        return result;
    }

    if (digits_class) {
        result = prbf_str_replace(dict, DIGITS_TPL, DIGITS, pool);
    }
    if (low_case_class) {
        result = prbf_str_replace(result, LOW_CASE_TPL, LOW_CASE, pool);
    }
    if (upper_case_class) {
        result = prbf_str_replace(result, UPPER_CASE_TPL, UPPER_CASE, pool);
    }

    dict_len = strlen((char *)result);
    unsigned char *result_without_duplicates =
        (unsigned char *)apr_pcalloc(pool, (dict_len + 1) * sizeof(unsigned char));

    size_t ir = 0;
    for (size_t i = 0; i < dict_len; ++i) {
        const unsigned char c = result[i];
        if (!chars_hash[c]) {
            result_without_duplicates[ir++] = c;
            chars_hash[c] = 1;
        }
    }

    return result_without_duplicates;
}

char *prbf_double_to_string(double value, apr_pool_t *pool) {
    const double rounded = round(value);
    const int digits = lib_count_digits_in(rounded);
    const size_t new_size = digits + digits / 3 + 1;

    char *result = (char *)apr_pcalloc(pool, sizeof(char) * new_size);
    lib_sprintf(result, "%.0f", value);
    lib_sprintf(result, "%s", prbf_commify(result, pool));
    return result;
}

char *prbf_int64_to_string(uint64_t value, apr_pool_t *pool) {
    const int digits = lib_count_digits_in(value);
    const size_t new_size = digits + digits / 3 + 1;

    char *result = (char *)apr_pcalloc(pool, sizeof(char) * new_size);
    lib_sprintf(result, "%llu", value);
    lib_sprintf(result, "%s", prbf_commify(result, pool));
    return result;
}

char *prbf_commify(char *numstr, apr_pool_t *pool) {
    char *ret = numstr;
    const char separator = ' ';

    char *wk = _strrev(apr_pstrdup(pool, numstr));

    char *p = strchr(wk, '.');
    if (p) {
        // include '.'
        while (wk != p) // skip until '.'
            *numstr++ = *wk++;
        *numstr++ = *wk++;
    }
    for (int i = 1; *wk; ++i) {
        if (isdigit(*wk)) {
            *numstr++ = *wk++;
            if (isdigit(*wk) && i % 3 == 0)
                *numstr++ = separator;
        } else {
            break;
        }
    }
    // ReSharper disable once CppUsingResultOfAssignmentAsCondition
    // ReSharper disable once CppPossiblyErroneousEmptyStatements
    while (*numstr++ = *wk++)
        ;
    return _strrev(ret);
}

const unsigned char *prbf_str_replace(const unsigned char *orig, const char *rep, const char *with, apr_pool_t *pool) {
    const unsigned char *result; // the return string
    const unsigned char *ins;    // the next insert point
    unsigned char *tmp;          // varies
    size_t len_rep;              // length of rep (the string to remove)
    size_t len_with;             // length of with (the string to replace rep with)
    size_t len_front;            // distance between rep and end of last rep
    size_t count;                // number of replacements
    size_t result_len;

    // sanity checks and initialization
    if (!orig || !rep) {
        return NULL;
    }
    len_rep = strlen(rep);
    if (len_rep == 0) {
        return orig;
    }
    if (!with) {
        with = "";
    }
    len_with = strlen(with);

    // count the number of replacements needed
    ins = orig;
    for (count = 0; tmp = strstr(ins, rep); ++count) {
        ins = tmp + len_rep;
    }

    result_len = strlen(orig) + (len_with - len_rep) * count + 1;
    result = tmp = (unsigned char *)apr_pcalloc(pool, result_len * sizeof(unsigned char));

    if (!result) {
        return orig;
    }

    // first time through the loop, all the variable are set correctly
    // from here on,
    //    tmp points to the end of the result string
    //    ins points to the next occurrence of rep in orig
    //    orig points to the remainder of orig after "end of rep"
    while (count--) {
        ins = strstr(orig, rep);
        len_front = ins - orig;
#ifdef __STDC_WANT_SECURE_LIB__
        strncpy_s(tmp, (len_front + 1) * sizeof(char), orig, len_front);
        tmp += len_front;
        strcpy_s(tmp, (len_with + 1) * sizeof(char), with);
        tmp += len_with;
#else
        tmp = strncpy(tmp, orig, len_front) + len_front;
        tmp = strcpy(tmp, with) + len_with;
#endif

        orig += len_front + len_rep; // move to next "end of rep"
    }
#ifdef __STDC_WANT_SECURE_LIB__
    strcpy_s(tmp, (strlen(orig) + 1) * sizeof(char), orig);
#else
    strcpy(tmp, orig);
#endif

    return result;
}

#ifndef _MSC_VER
char *_strrev(char *str) {
    char *p1 = str;
    char *p2;
    const size_t max_len = 256;

    if (!*str)
        return str;
    for (p2 = str + strnlen(str, max_len) - 1; p2 > p1; --p2, ++p1) {
        *p1 ^= *p2;
        *p2 ^= *p1;
        *p1 ^= *p2;
    }
    return str;
}
#endif
