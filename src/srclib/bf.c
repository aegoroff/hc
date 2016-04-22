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

BruteForceContext* ctx;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;
volatile apr_uint32_t alreadyFound = FALSE;

typedef struct ThreadContext {
    uint32_t Passmin;
    uint32_t Passmax;
    uint32_t Length;
    int Num;
    char* Pass;
    wchar_t* WidePass;
    size_t* Indexes;
    uint64_t NumOfAttempts;
    uint32_t NumOfThreads;
    BOOL UseWidePass;
} ThreadContext;

int MakeAttempt(const uint32_t pos, const size_t maxIndex, ThreadContext* tc);
const char* PrepareDictionary(const char* dict);
void* APR_THREAD_FUNC MakeAttemptThreadFunc(apr_thread_t* thd, void* data);
char* Commify(char* numstr, apr_pool_t* pool);
char* ToString(double value, apr_pool_t* pool);

void CrackHash(const char* dict,
               const char* hash,
               uint32_t passmin,
               uint32_t passmax,
               apr_size_t hashLength,
               void (*digestFunction)(apr_byte_t* digest, const void* string, const apr_size_t inputLen),
               BOOL noProbe,
               uint32_t numOfThreads,
               BOOL useWidePass,
               apr_pool_t* pool) {
    char* str = NULL;

    apr_byte_t* digest = (apr_byte_t*)apr_pcalloc(pool, hashLength);
    uint64_t attempts = 0;
    lib_time_t time = {0};
    double speed = 0.0;
    char* speedStr = NULL;


    // Empty string validation
    digestFunction(digest, "", 0);

    passmax = passmax ? passmax : MAX_DEFAULT;

    if(CompareHash(digest, hash)) {
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
            if(useWidePass) {
                wchar_t* s = enc_from_ansi_to_unicode(t, pool);
                digestFunction(digest, s, wcslen(s) * sizeof(wchar_t));
            }
            else {
                digestFunction(digest, t, strlen(t));
            }

            str1234 = out_hash_to_string(digest, FALSE, hashLength, pool);

            lib_start_timer();

            BruteForce(1,
                       MAX_DEFAULT,
                       alphabet,
                       str1234,
                       &attempts,
                       CreateDigest,
                       numOfThreads,
                       useWidePass,
                       pool);

            lib_stop_timer();
            time = lib_read_elapsed_time();
            ratio = attempts / time.seconds;

            attempts = 0;

            maxAttepts = pow(strlen(PrepareDictionary(dict)), passmax);
            maxTime = lib_normalize_time(maxAttepts / ratio);
            maxTimeMsg = (char*)apr_pcalloc(pool, maxTimeMsgSz + 1);
            lib_time_to_string(maxTime, maxTimeMsgSz, maxTimeMsg);
            lib_printf("May take approximatelly: %s (%.0f attempts)", maxTimeMsg, maxAttepts);
        }
        lib_start_timer();
        str = BruteForce(passmin, passmax, dict, hash, &attempts, CreateDigest, numOfThreads, useWidePass, pool);
    }

    lib_stop_timer();
    time = lib_read_elapsed_time();
    speed = attempts > 0 && time.total_seconds > 0 ? attempts / time.total_seconds : 0;
    speedStr = ToString(speed, pool);
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

char* BruteForce(const uint32_t passmin,
                 const uint32_t passmax,
                 const char* dict,
                 const char* hash,
                 uint64_t* attempts,
                 void* (* PfnHashPrepare)(const char* hash, apr_pool_t* pool),
                 uint32_t numOfThreads,
                 BOOL useWidePass,
                 apr_pool_t* pool) {
    apr_thread_t** thd_arr = NULL;
    ThreadContext** thd_ctx = NULL;
    apr_threadattr_t* thd_attr = NULL;
    apr_status_t rv;
    int i = 0;
    char* pass = NULL;

    alreadyFound = FALSE;

    if(passmax > INT_MAX / sizeof(int)) {
        lib_printf("Max string length is too big: %lu", passmax);
        return NULL;
    }

    ctx = (BruteForceContext*)apr_pcalloc(pool, sizeof(BruteForceContext));
    ctx->Desired = PfnHashPrepare(hash, pool);
    ctx->PfnHashCompare = CompareHashAttempt;
    ctx->Dict = PrepareDictionary(dict);

    thd_arr = (apr_thread_t**)apr_pcalloc(pool, sizeof(apr_thread_t*) * numOfThreads);
    thd_ctx = (ThreadContext**)apr_pcalloc(pool, sizeof(ThreadContext*) * numOfThreads);

    /* The default thread attribute: detachable */
    apr_threadattr_create(&thd_attr, pool);

    if(strlen(ctx->Dict) <= numOfThreads) {
        numOfThreads = strlen(ctx->Dict);
    }

    for(; i < numOfThreads; ++i) {
        thd_ctx[i] = (ThreadContext*)apr_pcalloc(pool, sizeof(ThreadContext));
        thd_ctx[i]->Passmin = passmin;
        thd_ctx[i]->Passmax = passmax;
        thd_ctx[i]->Num = i + 1;
        thd_ctx[i]->Pass = (char*)apr_pcalloc(pool, sizeof(char)* ((size_t)passmax + 1));
        thd_ctx[i]->WidePass = (wchar_t*)apr_pcalloc(pool, sizeof(wchar_t)* ((size_t)passmax + 1));
        thd_ctx[i]->Indexes = (size_t*)apr_pcalloc(pool, (size_t)passmax * sizeof(size_t));
        thd_ctx[i]->Length = passmin;
        thd_ctx[i]->NumOfThreads = numOfThreads;
        thd_ctx[i]->UseWidePass = useWidePass;
        rv = apr_thread_create(&thd_arr[i], thd_attr, MakeAttemptThreadFunc, thd_ctx[i], pool);
    }

    for(i = 0; i < numOfThreads; ++i) {
        rv = apr_thread_join(&rv, thd_arr[i]);
    }

    for(i = 0; i < numOfThreads; ++i) {
        (*attempts) += thd_ctx[i]->NumOfAttempts;

        if(thd_ctx[i]->UseWidePass) {
            if(thd_ctx[i]->WidePass != NULL) {
                pass = enc_from_unicode_to_ansi(thd_ctx[i]->WidePass, pool);
            }
        }
        else {
            if(thd_ctx[i]->Pass != NULL) {
                pass = thd_ctx[i]->Pass;
            }
        }
    }
    return pass;
}

/**
 * Thread entry point
 */
void* APR_THREAD_FUNC MakeAttemptThreadFunc(apr_thread_t* thd, void* data) {
    size_t maxIndex = 0;
    ThreadContext* tc = (ThreadContext*)data;

    maxIndex = strlen(ctx->Dict) - 1;

    for(; tc->Length <= tc->Passmax; ++tc->Length) {
        if(MakeAttempt(0, maxIndex, tc)) {
            goto result;
        }
        else if(apr_atomic_read32(&alreadyFound)) {
            break;
        }
    }
    tc->Pass = NULL;
    tc->WidePass = NULL;
result:
    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}

int MakeAttempt(const uint32_t pos, const size_t maxIndex, ThreadContext* tc) {
    size_t i = 0;
    int found = 0;

    for(; i <= maxIndex; ++i) {
        tc->Indexes[pos] = i;

        if(pos == tc->Length - 1) {
            uint32_t j = 0;
            while(j < tc->Length) {
                size_t dictPosition = tc->Indexes[j];

                if(
                    j > 0 ||
                    tc->NumOfThreads == 1 || // single threaded brute force
                    tc->Num == 1 && dictPosition % tc->NumOfThreads != 0 ||
                    (tc->Num - 1) + floor(dictPosition / tc->NumOfThreads) * tc->NumOfThreads == dictPosition
                ) {
                    if(tc->UseWidePass) {
                        tc->WidePass[j] = ctx->Dict[dictPosition];
                    }
                    else {
                        tc->Pass[j] = ctx->Dict[dictPosition];
                    }
                }
                else {
                    return FALSE;
                }
                ++j;
            }
            if(apr_atomic_read32(&alreadyFound)) {
                break;
            }
            ++(tc->NumOfAttempts);

            if(tc->UseWidePass) {
                found = ctx->PfnHashCompare(ctx->Desired, tc->WidePass, tc->Length * sizeof(wchar_t));
            }
            else {
                found = ctx->PfnHashCompare(ctx->Desired, tc->Pass, tc->Length);
            }
            if(found) {
                apr_atomic_set32(&alreadyFound, TRUE);
                return TRUE;
            }
        }
        else {
            if(MakeAttempt(pos + 1, maxIndex, tc)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}

const char* PrepareDictionary(const char* dict) {
    const char* digitsClass = NULL;
    const char* lowCaseClass = NULL;
    const char* upperCaseClass = NULL;

    digitsClass = strstr(dict, DIGITS_TPL);
    lowCaseClass = strstr(dict, LOW_CASE_TPL);
    upperCaseClass = strstr(dict, UPPER_CASE_TPL);

    if(!digitsClass && !lowCaseClass && !upperCaseClass) {
        return dict;
    }
    if(digitsClass && lowCaseClass && upperCaseClass) {
        return DIGITS LOW_CASE UPPER_CASE;
    }
    if(!digitsClass && lowCaseClass && upperCaseClass) {
        return LOW_CASE UPPER_CASE;
    }
    if(digitsClass && !lowCaseClass && upperCaseClass) {
        return DIGITS UPPER_CASE;
    }
    if(digitsClass && lowCaseClass && !upperCaseClass) {
        return DIGITS LOW_CASE;
    }
    if(digitsClass && !lowCaseClass && !upperCaseClass) {
        return DIGITS;
    }
    if(!digitsClass && !lowCaseClass && upperCaseClass) {
        return UPPER_CASE;
    }
    if(!digitsClass && lowCaseClass && !upperCaseClass) {
        return LOW_CASE;
    }

    return dict;
}

char* ToString(double value, apr_pool_t* pool) {
    char* result = NULL;
    double rounded = round(value);
    int digits = lib_count_digits_in(rounded);
    size_t newSize = digits + (digits / 3) + 1;

    result = (char*)apr_pcalloc(pool, sizeof(char) * newSize);
    sprintf_s(result, newSize, "%.0f", value);
    sprintf_s(result, newSize, "%s", Commify(result, pool));
    return result;
}

char* Commify(char* numstr, apr_pool_t* pool) {
    char* wk, * wks, * p, * ret = numstr;
    int i;

    wk = _strrev(apr_pstrdup(pool, numstr));
    wks = wk;

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
