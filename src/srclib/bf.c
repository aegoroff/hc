/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains brute force algorithm implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "targetver.h"
#include <math.h>
#include "apr_strings.h"
#include "apr_thread_proc.h"
#include "apr_atomic.h"
#include "bf.h"
#include "output.h"
#include "encoding.h"
#include "gpu.h"
#include "sha1.h"
#include "intl.h"

/*
    bf_ - public members
    prbf_ - private members
    g_ - global data
*/

typedef struct brute_force_ctx_t {
    const char* dict_;
    size_t dict_len_;
    void* hash_to_find_;
    int (*pfn_hash_compare_)(void* hash, const void* pass, const uint32_t length);
} brute_force_ctx_t;

typedef struct tread_ctx_t {
    uint32_t passmin_;
    uint32_t passmax_;
    uint32_t pass_length_;
    uint32_t thread_num_;
    char* pass_;
    wchar_t* wide_pass_;
    size_t* chars_indexes_;
    uint64_t num_of_attempts_;
    uint32_t num_of_threads;
    BOOL use_wide_pass_;
    BOOL found_in_the_thread_;
} tread_ctx_t;

static brute_force_ctx_t* g_brute_force_ctx;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;
static volatile apr_uint32_t already_found = FALSE;
static int g_gpu_variant_ix = 0;

static int prbf_make_attempt(const uint32_t pos, const size_t max_index, tread_ctx_t* tc);
static int prbf_make_gpu_attempt(gpu_tread_ctx_t* tc, int* alphabet_hash);
static const char* prbf_prepare_dictionary(const char* dict, apr_pool_t* pool);
static void* APR_THREAD_FUNC prbf_make_attempt_thread_func(apr_thread_t* thd, void* data);
static void* APR_THREAD_FUNC prbf_gpu_thread_func(apr_thread_t* thd, void* data);
static char* prbf_commify(char* numstr, apr_pool_t* pool);
static char* prbf_double_to_string(double value, apr_pool_t* pool);
static char* prbf_int64_to_string(uint64_t value, apr_pool_t* pool);
static const char* prbf_str_replace(const char* orig, const char* rep, const char* with, apr_pool_t* pool);

void bf_crack_hash(const char* dict,
                   const char* hash,
                   uint32_t passmin,
                   uint32_t passmax,
                   apr_size_t hash_length,
                   void (*pfn_digest_function)(apr_byte_t* digest, const void* string, const apr_size_t input_len),
                   BOOL no_probe,
                   uint32_t num_of_threads,
                   BOOL use_wide_pass,
                   BOOL has_gpu_implementation,
                   apr_pool_t* pool) {
    char* str;

    apr_byte_t* digest = (apr_byte_t*)apr_pcalloc(pool, hash_length);
    uint64_t attempts = 0;
    lib_time_t time;

    // Empty string validation
    pfn_digest_function(digest, "", 0);

    passmax = passmax ? passmax : MAX_DEFAULT;

    if(bf_compare_hash(digest, hash)) {
        str = _("Empty string");
        lib_start_timer();
    } else {
        // Probing
        size_t max_time_msg_size = 63;
        const char* t = "123";

        if(!no_probe) {
            if(use_wide_pass) {
                wchar_t* s = enc_from_ansi_to_unicode(t, pool);
                pfn_digest_function(digest, s, wcslen(s) * sizeof(wchar_t));
            } else {
                pfn_digest_function(digest, t, strlen(t));
            }

            const char* str123 = out_hash_to_string(digest, FALSE, hash_length, pool);

            lib_start_timer();

            bf_brute_force(1,
                           MAX_DEFAULT,
                           alphabet,
                           str123,
                           &attempts,
                           bf_create_digest,
                           num_of_threads,
                           use_wide_pass,
                           has_gpu_implementation,
                           pool);

            lib_stop_timer();
            time = lib_read_elapsed_time();
            double ratio = attempts / time.seconds;

            attempts = 0;

            const double max_attempts = pow(strlen(prbf_prepare_dictionary(dict, pool)), passmax);
            lib_time_t max_time = lib_normalize_time(max_attempts / ratio);
            char* max_time_msg = (char*)apr_pcalloc(pool, max_time_msg_size + 1);
            lib_time_to_string(&max_time, max_time_msg);
            lib_printf(_("May take approximatelly: %s (%s attempts)"), max_time_msg, prbf_double_to_string(max_attempts, pool));
        }

        // Main run
        lib_start_timer();
        str = bf_brute_force(passmin, passmax, dict, hash, &attempts, bf_create_digest, num_of_threads, use_wide_pass, has_gpu_implementation, pool);
    }

    lib_stop_timer();
    time = lib_read_elapsed_time();
    const double speed = attempts > 0 && time.total_seconds > 0 ? attempts / time.total_seconds : 0;
    char* speed_str = prbf_double_to_string(speed, pool);
    lib_new_line();
    lib_printf(_("Attempts: %s Time "), prbf_int64_to_string(attempts, pool));
    lib_printf(FULL_TIME_FMT, time.hours, time.minutes, time.seconds);
    lib_printf(_(" Speed: %s attempts/second"), speed_str);
    lib_new_line();
    if(str != NULL) {
        char* ansi = enc_from_utf8_to_ansi(str, pool);
        lib_printf(_("Initial string is: %s"), ansi == NULL ? str : ansi);
    } else {
        lib_printf(_("Nothing found"));
    }
    lib_new_line();
}

char* bf_brute_force(const uint32_t passmin,
                     const uint32_t passmax,
                     const char* dict,
                     const char* hash,
                     uint64_t* attempts,
                     void* (* pfn_hash_prepare)(const char* h, apr_pool_t* pool),
                     uint32_t num_of_threads,
                     BOOL use_wide_pass,
                     BOOL has_gpu_implementation,
                     apr_pool_t* pool) {
    apr_threadattr_t* thd_attr = NULL;
    apr_status_t rv;
    size_t i = 0;
    char* pass = NULL;
    

    already_found = FALSE;

    if(passmax > INT_MAX / sizeof(int)) {
        lib_printf(_("Max string length is too big: %lu"), passmax);
        return NULL;
    }

    device_props_t* gpu_props = (device_props_t*)apr_pcalloc(pool, sizeof(device_props_t));

    g_brute_force_ctx = (brute_force_ctx_t*)apr_pcalloc(pool, sizeof(brute_force_ctx_t));
    g_brute_force_ctx->hash_to_find_ = pfn_hash_prepare(hash, pool);
    g_brute_force_ctx->pfn_hash_compare_ = bf_compare_hash_attempt;
    g_brute_force_ctx->dict_ = prbf_prepare_dictionary(dict, pool);
    g_brute_force_ctx->dict_len_ = strlen(g_brute_force_ctx->dict_);

    apr_thread_t** thd_arr = (apr_thread_t**)apr_pcalloc(pool, sizeof(apr_thread_t*) * num_of_threads);
    tread_ctx_t** thd_ctx = (tread_ctx_t**)apr_pcalloc(pool, sizeof(tread_ctx_t*) * num_of_threads);

    /* The default thread attribute: detachable */
    apr_threadattr_create(&thd_attr, pool);

    if(g_brute_force_ctx->dict_len_ <= num_of_threads) {
        num_of_threads = g_brute_force_ctx->dict_len_;
    }

    /* If max password length less then 4 GPU not needed */
    has_gpu_implementation = has_gpu_implementation && passmax > 3;

    if(has_gpu_implementation) {
        gpu_get_props(gpu_props);

        if (gpu_props->device_count) {
            num_of_threads -= gpu_props->device_count;
        }
    }
    
    for(; i < num_of_threads; ++i) {
        thd_ctx[i] = (tread_ctx_t*)apr_pcalloc(pool, sizeof(tread_ctx_t));
        thd_ctx[i]->passmin_ = passmin;
        thd_ctx[i]->passmax_ = passmax;
        thd_ctx[i]->thread_num_ = i + 1;
        thd_ctx[i]->pass_ = (char*)apr_pcalloc(pool, sizeof(char)* ((size_t)passmax + 1));
        thd_ctx[i]->wide_pass_ = (wchar_t*)apr_pcalloc(pool, sizeof(wchar_t)* ((size_t)passmax + 1));
        thd_ctx[i]->chars_indexes_ = (size_t*)apr_pcalloc(pool, (size_t)passmax * sizeof(size_t));
        thd_ctx[i]->pass_length_ = passmin;
        thd_ctx[i]->num_of_threads = num_of_threads;
        thd_ctx[i]->use_wide_pass_ = use_wide_pass;
        rv = apr_thread_create(&thd_arr[i], thd_attr, prbf_make_attempt_thread_func, thd_ctx[i], pool);
    }

    if (has_gpu_implementation && gpu_props->device_count) {
        gpu_tread_ctx_t** gpu_thd_ctx = (gpu_tread_ctx_t**)apr_pcalloc(pool, sizeof(gpu_tread_ctx_t*) * gpu_props->device_count);
        apr_thread_t** gpu_thd_arr = (apr_thread_t**)apr_pcalloc(pool, sizeof(apr_thread_t*) * gpu_props->device_count);
        apr_threadattr_t* gpu_thd_attr = NULL;
        apr_threadattr_create(&gpu_thd_attr, pool);

        for (i = 0; i < gpu_props->device_count; ++i) {
            gpu_thd_ctx[i] = (gpu_tread_ctx_t*)apr_pcalloc(pool, sizeof(gpu_tread_ctx_t));
            gpu_thd_ctx[i]->passmin_ = passmin;
            gpu_thd_ctx[i]->passmax_ = passmax;
            gpu_thd_ctx[i]->attempt_ = (char*)apr_pcalloc(pool, sizeof(char)* ((size_t)passmax + 1));
            gpu_thd_ctx[i]->result_ = (char*)apr_pcalloc(pool, sizeof(char)* ((size_t)passmax + 1));
            gpu_thd_ctx[i]->pass_length_ = passmin;
            gpu_thd_ctx[i]->pool_ = pool;
            gpu_thd_ctx[i]->max_gpu_blocks_number_ = gpu_props->max_blocks_number * 2; // two times more then max device blocks number
            gpu_thd_ctx[i]->max_threads_per_block_ = gpu_props->max_threads_per_block;
            rv = apr_thread_create(&gpu_thd_arr[i], gpu_thd_attr, prbf_gpu_thread_func, gpu_thd_ctx[i], pool);
        }

        for (i = 0; i < gpu_props->device_count; ++i) {
            rv = apr_thread_join(&rv, gpu_thd_arr[i]);

            (*attempts) += gpu_thd_ctx[i]->num_of_attempts_;

            if(gpu_thd_ctx[i]->found_in_the_thread_) {
                pass = gpu_thd_ctx[i]->result_;
            }
        }
    }

    for(i = 0; i < num_of_threads; ++i) {
        rv = apr_thread_join(&rv, thd_arr[i]);

        (*attempts) += thd_ctx[i]->num_of_attempts_;

        if (thd_ctx[i]->use_wide_pass_) {
            if (thd_ctx[i]->wide_pass_ != NULL) {
                pass = enc_from_unicode_to_ansi(thd_ctx[i]->wide_pass_, pool);
            }
        }
        else {
            if (thd_ctx[i]->pass_ != NULL) {
                pass = thd_ctx[i]->pass_;
            }
        }
    }
    
    return pass;
}

/**
 * Thread entry point
 */
void* APR_THREAD_FUNC prbf_make_attempt_thread_func(apr_thread_t* thd, void* data) {
    tread_ctx_t* tc = (tread_ctx_t*)data;

    const size_t max_index = g_brute_force_ctx->dict_len_ - 1;

    for(; tc->pass_length_ <= tc->passmax_ - 2; ++tc->pass_length_) {
        if(prbf_make_attempt(0, max_index, tc)) {
            goto result;
        }

        if(apr_atomic_read32(&already_found)) {
            break;
        }
    }
    tc->pass_ = NULL;
    tc->wide_pass_ = NULL;
result:
    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}

int indexofchar(const char c, int* alphabet_hash) {
    return c ? alphabet_hash[c] : -1;
}

/**
 * GPU thread entry point
 */
void* APR_THREAD_FUNC prbf_gpu_thread_func(apr_thread_t* thd, void* data) {
    gpu_tread_ctx_t* tc = (gpu_tread_ctx_t*)data;
    int alphabet_hash[MAXBYTE + 1];

    // fill ABC hash
    for (int ix = 0; ix < g_brute_force_ctx->dict_len_; ix++) {
        alphabet_hash[g_brute_force_ctx->dict_[ix]] = ix;
    }
    
    tc->variants_size_ = (tc->max_gpu_blocks_number_ * tc->max_threads_per_block_) * MAX_DEFAULT;
    tc->variants_ = (char*)apr_pcalloc(tc->pool_, tc->variants_size_ * sizeof(char));

    sha1_on_gpu_prepare((unsigned char*)g_brute_force_ctx->dict_, g_brute_force_ctx->dict_len_, g_brute_force_ctx->hash_to_find_);

    for (; tc->pass_length_ <= tc->passmax_ - 1; ++tc->pass_length_) {
        if (prbf_make_gpu_attempt(tc, alphabet_hash)) {
            break;
        }

        if(tc->found_in_the_thread_) {
            apr_atomic_set32(&already_found, TRUE);
        }
    }

    sha1_on_gpu_cleanup();
    
    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}

int prbf_make_gpu_attempt(gpu_tread_ctx_t* tc, int* alphabet_hash) {
    // ti - text index
    // li - ABC index

    // start rotating chars from the back and forward
    for (int ti = tc->pass_length_ - 1, li; ti > -1; ti--) {
        for (li = indexofchar(tc->attempt_[ti], alphabet_hash) + 1; li < g_brute_force_ctx->dict_len_; li++) {
            // test
            tc->attempt_[ti] = g_brute_force_ctx->dict_[li];
            ++tc->num_of_attempts_;

            // Probe attempt
            memcpy_s(tc->variants_ + g_gpu_variant_ix * MAX_DEFAULT, tc->pass_length_, tc->attempt_, tc->pass_length_);
            if (g_gpu_variant_ix < tc->variants_size_ / MAX_DEFAULT) {
                ++g_gpu_variant_ix;
            }
            else {
                g_gpu_variant_ix = 0;
                if (apr_atomic_read32(&already_found)) {
                    return TRUE;
                }
                sha1_run_on_gpu(tc, g_brute_force_ctx->dict_len_, (unsigned char*)tc->variants_, tc->variants_size_);
            }

            if (tc->found_in_the_thread_) {
                apr_atomic_set32(&already_found, TRUE);
                return TRUE;
            }

            // rotate to the right
            for (int z = ti + 1; z < tc->pass_length_; z++) {
                if (tc->attempt_[z] != g_brute_force_ctx->dict_[g_brute_force_ctx->dict_len_ - 1]) {
                    ti = tc->pass_length_;
                    goto outerBreak;
                }
            }
        }
    outerBreak:
        if (li == g_brute_force_ctx->dict_len_)
            tc->attempt_[ti] = g_brute_force_ctx->dict_[0];
    }

    return FALSE;
}

int prbf_make_attempt(const uint32_t pos, const size_t max_index, tread_ctx_t* tc) {
    size_t i = 0;
    int found;

    for(; i <= max_index; ++i) {
        tc->chars_indexes_[pos] = i;

        if(pos == tc->pass_length_ - 1) {
            uint32_t j = 0;

            // Generate attempt
            while(j < tc->pass_length_) {
                const size_t dict_position = tc->chars_indexes_[j];

                if(
                    j > 0 ||
                    tc->num_of_threads == 1 || // single threaded brute force
                    (tc->thread_num_ == 1 && dict_position % tc->num_of_threads != 0) ||
                    (tc->thread_num_ - 1) + (uint32_t)floor(dict_position / tc->num_of_threads) * tc->num_of_threads == dict_position
                ) {
                    if(tc->use_wide_pass_) {
                        tc->wide_pass_[j] = g_brute_force_ctx->dict_[dict_position];
                    } else {
                        tc->pass_[j] = g_brute_force_ctx->dict_[dict_position];
                    }
                } else {
                    return FALSE;
                }
                ++j;
            }
            if(apr_atomic_read32(&already_found)) {
                break;
            }
            ++(tc->num_of_attempts_);

            // Probe attempt
            if(tc->use_wide_pass_) {
                found = g_brute_force_ctx->pfn_hash_compare_(g_brute_force_ctx->hash_to_find_, tc->wide_pass_, tc->pass_length_ * sizeof(wchar_t));
            } else {
                found = g_brute_force_ctx->pfn_hash_compare_(g_brute_force_ctx->hash_to_find_, tc->pass_, tc->pass_length_);
            }
            if(found) {
                apr_atomic_set32(&already_found, TRUE);
                return TRUE;
            }
        } else {
            // All attempts with length = pos done. Increment max attempt length
            if(prbf_make_attempt(pos + 1, max_index, tc)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}

const char* prbf_prepare_dictionary(const char* dict, apr_pool_t* pool) {
    const char* digits_class = strstr(dict, DIGITS_TPL);
    const char* low_case_class = strstr(dict, LOW_CASE_TPL);
    const char* upper_case_class = strstr(dict, UPPER_CASE_TPL);
    const char* result = dict;

    if(!digits_class && !low_case_class && !upper_case_class) {
        return dict;
    }
    if(digits_class) {
        result = prbf_str_replace(dict, DIGITS_TPL, DIGITS, pool);
    }
    if(low_case_class) {
        result = prbf_str_replace(result, LOW_CASE_TPL, LOW_CASE, pool);
    }
    if(upper_case_class) {
        result = prbf_str_replace(result, UPPER_CASE_TPL, UPPER_CASE, pool);
    }
    return result;
}

char* prbf_double_to_string(double value, apr_pool_t* pool) {
    const double rounded = round(value);
    const int digits = lib_count_digits_in(rounded);
    size_t new_size = digits + digits / 3 + 1;

    char* result = (char*)apr_pcalloc(pool, sizeof(char) * new_size);
    lib_sprintf(result, "%.0f", value);
    lib_sprintf(result, "%s", prbf_commify(result, pool));
    return result;
}

char* prbf_int64_to_string(uint64_t value, apr_pool_t* pool) {
    int digits = lib_count_digits_in(value);
    size_t new_size = digits + digits / 3 + 1;

    char* result = (char*)apr_pcalloc(pool, sizeof(char) * new_size);
    lib_sprintf(result, "%llu", value);
    lib_sprintf(result, "%s", prbf_commify(result, pool));
    return result;
}

char* prbf_commify(char* numstr, apr_pool_t* pool) {
    char* ret = numstr;
    const char separator = ' ';

    char* wk = _strrev(apr_pstrdup(pool, numstr));

    char* p = strchr(wk, '.');
    if(p) {//include '.' 
        while(wk != p)//skip until '.'
            *numstr++ = *wk++;
        *numstr++ = *wk++;
    }
    for(int i = 1; *wk; ++i) {
        if(isdigit(*wk)) {
            *numstr++ = *wk++;
            if(isdigit(*wk) && i % 3 == 0)
                *numstr++ = separator;
        } else {
            break;
        }
    }
    // ReSharper disable once CppUsingResultOfAssignmentAsCondition
    // ReSharper disable once CppPossiblyErroneousEmptyStatements
    while(*numstr++ = *wk++);
    return _strrev(ret);
}

const char* prbf_str_replace(const char* orig, const char* rep, const char* with, apr_pool_t* pool) {
    const char* result; // the return string
    const char* ins; // the next insert point
    char* tmp; // varies
    size_t len_rep; // length of rep (the string to remove)
    size_t len_with; // length of with (the string to replace rep with)
    size_t len_front; // distance between rep and end of last rep
    size_t count; // number of replacements
    size_t result_len;

    // sanity checks and initialization
    if(!orig || !rep) {
        return NULL;
    }
    len_rep = strlen(rep);
    if(len_rep == 0) {
        return orig;
    }
    if(!with) {
        with = "";
    }
    len_with = strlen(with);

    // count the number of replacements needed
    ins = orig;
    for(count = 0; tmp = strstr(ins, rep); ++count) {
        ins = tmp + len_rep;
    }

    result_len = strlen(orig) + (len_with - len_rep) * count + 1;
    result = tmp = (char*)apr_pcalloc(pool, result_len * sizeof(char));

    if(!result) {
        return orig;
    }

    // first time through the loop, all the variable are set correctly
    // from here on,
    //    tmp points to the end of the result string
    //    ins points to the next occurrence of rep in orig
    //    orig points to the remainder of orig after "end of rep"
    while(count--) {
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
