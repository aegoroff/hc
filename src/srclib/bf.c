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

/*
    bf_ - public members
    prbf_ - private members
*/

typedef struct brute_force_ctx_t {
    const char* dict;
    void* desired;
    int (*pfn_hash_compare_)(void* hash, const void* pass, const uint32_t length);
} brute_force_ctx_t;

static brute_force_ctx_t* ctx;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;
static volatile apr_uint32_t already_found = FALSE;

typedef struct tread_ctx_t {
    uint32_t passmin_;
    uint32_t passmax_;
    uint32_t length_;
    uint32_t num_;
    char* pass_;
    wchar_t* wide_pass_;
    size_t* indexes_;
    uint64_t num_of_attempts_;
    uint32_t num_of_threads;
    BOOL use_wide_pass_;
} tread_ctx_t;

static int prbf_make_attempt(const uint32_t pos, const size_t max_index, tread_ctx_t* tc);
static const char* prbf_prepare_dictionary(const char* dict, apr_pool_t* pool);
static void* APR_THREAD_FUNC prbf_make_attempt_thread_func(apr_thread_t* thd, void* data);
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
        size_t max_time_msg_size = 63;
        const char* t = "123";

        if(!no_probe) {
            if(use_wide_pass) {
                wchar_t* s = enc_from_ansi_to_unicode(t, pool);
                pfn_digest_function(digest, s, wcslen(s) * sizeof(wchar_t));
            } else {
                pfn_digest_function(digest, t, strlen(t));
            }

            const char* str1234 = out_hash_to_string(digest, FALSE, hash_length, pool);

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
            double ratio = attempts / time.seconds;

            attempts = 0;

            const double max_attempts = pow(strlen(prbf_prepare_dictionary(dict, pool)), passmax);
            lib_time_t max_time = lib_normalize_time(max_attempts / ratio);
            char* max_time_msg = (char*)apr_pcalloc(pool, max_time_msg_size + 1);
            lib_time_to_string(&max_time, max_time_msg);
            lib_printf(_("May take approximatelly: %s (%s attempts)"), max_time_msg, prbf_double_to_string(max_attempts, pool));
        }
        lib_start_timer();
        str = bf_brute_force(passmin, passmax, dict, hash, &attempts, bf_create_digest, num_of_threads, use_wide_pass, pool);
    }

    lib_stop_timer();
    time = lib_read_elapsed_time();
    double speed = attempts > 0 && time.total_seconds > 0 ? attempts / time.total_seconds : 0;
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

    ctx = (brute_force_ctx_t*)apr_pcalloc(pool, sizeof(brute_force_ctx_t));
    ctx->desired = pfn_hash_prepare(hash, pool);
    ctx->pfn_hash_compare_ = bf_compare_hash_attempt;
    ctx->dict = prbf_prepare_dictionary(dict, pool);

    apr_thread_t** thd_arr = (apr_thread_t**)apr_pcalloc(pool, sizeof(apr_thread_t*) * num_of_threads);
    tread_ctx_t** thd_ctx = (tread_ctx_t**)apr_pcalloc(pool, sizeof(tread_ctx_t*) * num_of_threads);

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
        rv = apr_thread_create(&thd_arr[i], thd_attr, prbf_make_attempt_thread_func, thd_ctx[i], pool);
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
        } else {
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
void* APR_THREAD_FUNC prbf_make_attempt_thread_func(apr_thread_t* thd, void* data) {
    tread_ctx_t* tc = (tread_ctx_t*)data;

    const size_t max_index = strlen(ctx->dict) - 1;

    for(; tc->length_ <= tc->passmax_; ++tc->length_) {
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

int prbf_make_attempt(const uint32_t pos, const size_t max_index, tread_ctx_t* tc) {
    size_t i = 0;
    int found;

    for(; i <= max_index; ++i) {
        tc->indexes_[pos] = i;

        if(pos == tc->length_ - 1) {
            uint32_t j = 0;
            while(j < tc->length_) {
                const size_t dict_position = tc->indexes_[j];

                if(
                    j > 0 ||
                    tc->num_of_threads == 1 || // single threaded brute force
                    (tc->num_ == 1 && dict_position % tc->num_of_threads != 0) ||
                    (tc->num_ - 1) + (uint32_t)floor(dict_position / tc->num_of_threads) * tc->num_of_threads == dict_position
                ) {
                    if(tc->use_wide_pass_) {
                        tc->wide_pass_[j] = ctx->dict[dict_position];
                    } else {
                        tc->pass_[j] = ctx->dict[dict_position];
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

            if(tc->use_wide_pass_) {
                found = ctx->pfn_hash_compare_(ctx->desired, tc->wide_pass_, tc->length_ * sizeof(wchar_t));
            } else {
                found = ctx->pfn_hash_compare_(ctx->desired, tc->pass_, tc->length_);
            }
            if(found) {
                apr_atomic_set32(&already_found, TRUE);
                return TRUE;
            }
        } else {
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
