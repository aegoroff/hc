/*!
 * \brief   The file contains brute force algorithm implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include "targetver.h"
#include <math.h>
#include "apr_strings.h"
#include "apr_thread_proc.h"
#include "apr_atomic.h"
#include "bf.h"
#include "output.h"
#include "encoding.h"

typedef struct brute_force_ctx_t {
    const char* dict;
    void*       desired;
    int(*PfnHashCompare)(void* hash, const void* pass, const uint32_t length);
} brute_force_ctx_t;

brute_force_ctx_t* ctx;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;
volatile apr_uint32_t already_found = FALSE;

typedef struct tread_ctx_t {
    uint32_t passmin_;
    uint32_t passmax_;
    uint32_t length_;
    int num_;
    char* pass_;
    wchar_t* wide_pass_;
    size_t* indexes_;
    uint64_t num_of_attempts_;
    uint32_t num_of_threads;
    BOOL use_wide_pass_;
} tread_ctx_t;

int bfp_make_attempt(const uint32_t pos, const size_t max_index, tread_ctx_t* tc);
const char* bfp_prepare_dictionary(const char* dict);
void* APR_THREAD_FUNC bfp_make_attempt_thread_func(apr_thread_t* thd, void* data);
char* bfp_commify(char* numstr, apr_pool_t* pool);
char* bfp_to_string(double value, apr_pool_t* pool);

void bf_crack_hash(const char* dict,
               const char* hash,
               uint32_t passmin,
               uint32_t passmax,
               apr_size_t hashLength,
               void (*pfn_digest_function)(apr_byte_t* digest, const void* string, const apr_size_t input_len),
               BOOL noProbe,
               uint32_t num_of_threads,
               BOOL use_wide_pass,
               apr_pool_t* pool) {
    char* str = NULL;

    apr_byte_t* digest = (apr_byte_t*)apr_pcalloc(pool, hashLength);
    uint64_t attempts = 0;
    lib_time_t time = {0};
    double speed = 0.0;
    char* speedStr = NULL;


    // Empty string validation
    pfn_digest_function(digest, "", 0);

    passmax = passmax ? passmax : MAX_DEFAULT;

    if(bf_compare_hash(digest, hash)) {
        str = "Empty string";
        lib_start_timer();
    }
    else {
        char* maxTimeMsg = NULL;
        size_t maxTimeMsgSz = 63;
        double ratio = 0;
        double maxAttepts = 0;
        lib_time_t maxTime = {0};
        const char* str1234 = NULL;
        const char* t = "123";

        if(!noProbe) {
            if(use_wide_pass) {
                wchar_t* s = enc_from_ansi_to_unicode(t, pool);
                pfn_digest_function(digest, s, wcslen(s) * sizeof(wchar_t));
            }
            else {
                pfn_digest_function(digest, t, strlen(t));
            }

            str1234 = out_hash_to_string(digest, FALSE, hashLength, pool);

            lib_start_timer();

            bf_brute_force(1,
                       MAX_DEFAULT,
                       alphabet,
                       str1234,
                       &attempts,
                       bf_create_digest,
                       num_of_threads,
                       use_wide_pass,
                       pool);

            lib_stop_timer();
            time = lib_read_elapsed_time();
            ratio = attempts / time.seconds;

            attempts = 0;

            maxAttepts = pow(strlen(bfp_prepare_dictionary(dict)), passmax);
            maxTime = lib_normalize_time(maxAttepts / ratio);
            maxTimeMsg = (char*)apr_pcalloc(pool, maxTimeMsgSz + 1);
            lib_time_to_string(maxTime, maxTimeMsgSz, maxTimeMsg);
            lib_printf("May take approximatelly: %s (%.0f attempts)", maxTimeMsg, maxAttepts);
        }
        lib_start_timer();
        str = bf_brute_force(passmin, passmax, dict, hash, &attempts, bf_create_digest, num_of_threads, use_wide_pass, pool);
    }

    lib_stop_timer();
    time = lib_read_elapsed_time();
    speed = attempts > 0 && time.total_seconds > 0 ? attempts / time.total_seconds : 0;
    speedStr = bfp_to_string(speed, pool);
    lib_printf(NEW_LINE "Attempts: %llu Time " FULL_TIME_FMT " Speed: %s attempts/second",
                      attempts,
                      time.hours,
                      time.minutes,
                      time.seconds,
                      speedStr);
    lib_new_line();
    if(str != NULL) {
        char* ansi = enc_from_utf8_to_ansi(str, pool);
        lib_printf("Initial string is: %s", ansi == NULL ? str : ansi);
    }
    else {
        lib_printf("Nothing found");
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
                 apr_pool_t* pool) {
    apr_thread_t** thd_arr;
    tread_ctx_t** thd_ctx = NULL;
    apr_threadattr_t* thd_attr = NULL;
    apr_status_t rv;
    size_t i = 0;
    char* pass = NULL;

    already_found = FALSE;

    if(passmax > INT_MAX / sizeof(int)) {
        lib_printf("Max string length is too big: %lu", passmax);
        return NULL;
    }

    ctx = (brute_force_ctx_t*)apr_pcalloc(pool, sizeof(brute_force_ctx_t));
    ctx->desired = pfn_hash_prepare(hash, pool);
    ctx->PfnHashCompare = bf_compare_hash_attempt;
    ctx->dict = bfp_prepare_dictionary(dict);

    thd_arr = (apr_thread_t**)apr_pcalloc(pool, sizeof(apr_thread_t*) * num_of_threads);
    thd_ctx = (tread_ctx_t**)apr_pcalloc(pool, sizeof(tread_ctx_t*) * num_of_threads);

    /* The default thread attribute: detachable */
    apr_threadattr_create(&thd_attr, pool);

    if(strlen(ctx->dict) <= num_of_threads) {
        num_of_threads = strlen(ctx->dict);
    }

    for(; i < num_of_threads; ++i) {
        thd_ctx[i] = (tread_ctx_t*)apr_pcalloc(pool, sizeof(tread_ctx_t));
        thd_ctx[i]->passmin_ = passmin;
        thd_ctx[i]->passmax_ = passmax;
        thd_ctx[i]->num_ = i + 1;
        thd_ctx[i]->pass_ = (char*)apr_pcalloc(pool, sizeof(char)* ((size_t)passmax + 1));
        thd_ctx[i]->wide_pass_ = (wchar_t*)apr_pcalloc(pool, sizeof(wchar_t)* ((size_t)passmax + 1));
        thd_ctx[i]->indexes_ = (size_t*)apr_pcalloc(pool, (size_t)passmax * sizeof(size_t));
        thd_ctx[i]->length_ = passmin;
        thd_ctx[i]->num_of_threads = num_of_threads;
        thd_ctx[i]->use_wide_pass_ = use_wide_pass;
        rv = apr_thread_create(&thd_arr[i], thd_attr, bfp_make_attempt_thread_func, thd_ctx[i], pool);
    }

    for(i = 0; i < num_of_threads; ++i) {
        rv = apr_thread_join(&rv, thd_arr[i]);
    }

    for(i = 0; i < num_of_threads; ++i) {
        (*attempts) += thd_ctx[i]->num_of_attempts_;

        if(thd_ctx[i]->use_wide_pass_) {
            if(thd_ctx[i]->wide_pass_ != NULL) {
                pass = enc_from_unicode_to_ansi(thd_ctx[i]->wide_pass_, pool);
            }
        }
        else {
            if(thd_ctx[i]->pass_ != NULL) {
                pass = thd_ctx[i]->pass_;
            }
        }
    }
    return pass;
}

/**
 * Thread entry point
 */
void* APR_THREAD_FUNC bfp_make_attempt_thread_func(apr_thread_t* thd, void* data) {
    size_t maxIndex = 0;
    tread_ctx_t* tc = (tread_ctx_t*)data;

    maxIndex = strlen(ctx->dict) - 1;

    for(; tc->length_ <= tc->passmax_; ++tc->length_) {
        if(bfp_make_attempt(0, maxIndex, tc)) {
            goto result;
        }
        else if(apr_atomic_read32(&already_found)) {
            break;
        }
    }
    tc->pass_ = NULL;
    tc->wide_pass_ = NULL;
result:
    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}

int bfp_make_attempt(const uint32_t pos, const size_t max_index, tread_ctx_t* tc) {
    size_t i = 0;
    int found;

    for(; i <= max_index; ++i) {
        tc->indexes_[pos] = i;

        if(pos == tc->length_ - 1) {
            uint32_t j = 0;
            while(j < tc->length_) {
                size_t dictPosition = tc->indexes_[j];

                if(
                    j > 0 ||
                    tc->num_of_threads == 1 || // single threaded brute force
                    tc->num_ == 1 && dictPosition % tc->num_of_threads != 0 ||
                    (tc->num_ - 1) + floor(dictPosition / tc->num_of_threads) * tc->num_of_threads == dictPosition
                ) {
                    if(tc->use_wide_pass_) {
                        tc->wide_pass_[j] = ctx->dict[dictPosition];
                    }
                    else {
                        tc->pass_[j] = ctx->dict[dictPosition];
                    }
                }
                else {
                    return FALSE;
                }
                ++j;
            }
            if(apr_atomic_read32(&already_found)) {
                break;
            }
            ++(tc->num_of_attempts_);

            if(tc->use_wide_pass_) {
                found = ctx->PfnHashCompare(ctx->desired, tc->wide_pass_, tc->length_ * sizeof(wchar_t));
            }
            else {
                found = ctx->PfnHashCompare(ctx->desired, tc->pass_, tc->length_);
            }
            if(found) {
                apr_atomic_set32(&already_found, TRUE);
                return TRUE;
            }
        }
        else {
            if(bfp_make_attempt(pos + 1, max_index, tc)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}

const char* bfp_prepare_dictionary(const char* dict) {
    const char* digits_class;
    const char* low_case_class;
    const char* upper_case_class;

    digits_class = strstr(dict, DIGITS_TPL);
    low_case_class = strstr(dict, LOW_CASE_TPL);
    upper_case_class = strstr(dict, UPPER_CASE_TPL);

    if(!digits_class && !low_case_class && !upper_case_class) {
        return dict;
    }
    if(digits_class && low_case_class && upper_case_class) {
        return DIGITS LOW_CASE UPPER_CASE;
    }
    if(!digits_class && low_case_class && upper_case_class) {
        return LOW_CASE UPPER_CASE;
    }
    if(digits_class && !low_case_class && upper_case_class) {
        return DIGITS UPPER_CASE;
    }
    if(digits_class && low_case_class && !upper_case_class) {
        return DIGITS LOW_CASE;
    }
    if(digits_class && !low_case_class && !upper_case_class) {
        return DIGITS;
    }
    if(!digits_class && !low_case_class && upper_case_class) {
        return UPPER_CASE;
    }
    if(!digits_class && low_case_class && !upper_case_class) {
        return LOW_CASE;
    }

    return dict;
}

char* bfp_to_string(double value, apr_pool_t* pool) {
    char* result;
    double rounded = round(value);
    int digits = lib_count_digits_in(rounded);
    size_t newSize = digits + (digits / 3) + 1;

    result = (char*)apr_pcalloc(pool, sizeof(char) * newSize);
    sprintf_s(result, newSize, "%.0f", value);
    sprintf_s(result, newSize, "%s", bfp_commify(result, pool));
    return result;
}

char* bfp_commify(char* numstr, apr_pool_t* pool) {
    char* wk, * p, * ret = numstr;
    int i;

    wk = _strrev(apr_pstrdup(pool, numstr));

    p = strchr(wk, '.');
    if(p) {//include '.' 
        while(wk != p)//skip until '.'
            *numstr++ = *wk++;
        *numstr++ = *wk++;
    }
    for(i = 1; *wk; ++i) {
        if(isdigit(*wk)) {
            *numstr++ = *wk++;
            if(isdigit(*wk) && i % 3 == 0)
                *numstr++ = ',';
        }
        else {
            break;
        }
    }
    while(*numstr++ = *wk++);
    return _strrev(ret);
}
