/*!
 * \brief   The file contains brute force algorithm implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
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
    size_t* Indexes;
    uint64_t NumOfAttempts;
    uint32_t NumOfThreads;
} ThreadContext;

int MakeAttempt(const uint32_t pos, const size_t maxIndex, ThreadContext* tc);
const char* PrepareDictionary(const char* dict);
void* APR_THREAD_FUNC MakeAttemptThreadFunc(apr_thread_t *thd, void *data);

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_size_t  hashLength,
               void (*digestFunction)(apr_byte_t* digest, const char* string, const apr_size_t inputLen),
               BOOL noProbe,
               uint32_t numOfThreads,
               apr_pool_t* pool)
{
    char* str = NULL;

    apr_byte_t* digest = (apr_byte_t*)apr_pcalloc(pool, hashLength);
    uint64_t attempts = 0;
    Time time = { 0 };
    double speed = 0.0;
    char* speedStr = NULL;
    size_t speedStrSize = 64;


    // Empty string validation
    digestFunction(digest, "", 0);

    passmax = passmax ? passmax : MAX_DEFAULT;

    if (CompareHash(digest, hash)) {
        str = "Empty string";
        StartTimer();
    } else {
        char* maxTimeMsg = NULL;
        size_t maxTimeMsgSz = 63;
        double ratio = 0;
        double maxAttepts = 0;
        Time maxTime = { 0 };
        const char* str1234 = NULL;
        const char* t = "123";

        if (!noProbe) {
            digestFunction(digest, t, strlen(t));
            str1234 = HashToString(digest, FALSE, hashLength, pool);

            StartTimer();

            BruteForce(1,
                       MAX_DEFAULT,
                       alphabet,
                       str1234,
                       &attempts,
                       CreateDigest,
                       numOfThreads,
                       pool);

            StopTimer();
            time = ReadElapsedTime();
            ratio = attempts / time.seconds;

            attempts = 0;

            maxAttepts = pow(strlen(PrepareDictionary(dict)), passmax);
            maxTime = NormalizeTime(maxAttepts / ratio);
            maxTimeMsg = (char*)apr_pcalloc(pool, maxTimeMsgSz + 1);
            TimeToString(maxTime, maxTimeMsgSz, maxTimeMsg);
            CrtPrintf("May take approximatelly: %s (%.0f attempts)", maxTimeMsg, maxAttepts);
        }
        StartTimer();
        alreadyFound = FALSE;
        str = BruteForce(passmin, passmax, dict, hash, &attempts, CreateDigest, numOfThreads, pool);
    }

    StopTimer();
    time = ReadElapsedTime();
    speed = attempts / time.total_seconds;
    speedStr = (char*)apr_pcalloc(pool, speedStrSize);
    ToString(speed, speedStr, speedStrSize);
    CrtPrintf(NEW_LINE "Attempts: %llu Time " FULL_TIME_FMT " Speed: %s attempts/second",
              attempts,
              time.hours,
              time.minutes,
              time.seconds,
              speedStr);
    NewLine();
    if (str != NULL) {
        char* ansi = FromUtf8ToAnsi(str, pool);
        CrtPrintf("Initial string is: %s", ansi == NULL ? str : ansi);
    } else {
        CrtPrintf("Nothing found");
    }
    NewLine();
}

char* BruteForce(const uint32_t    passmin,
                 const uint32_t    passmax,
                 const char*       dict,
                 const char*       hash,
                 uint64_t*         attempts,
                 void* (* PfnHashPrepare)(const char* hash, apr_pool_t* pool),
                 uint32_t numOfThreads,
                 apr_pool_t*       pool)
{
    apr_thread_t** thd_arr = NULL;
    ThreadContext** thd_ctx = NULL;
    apr_threadattr_t* thd_attr = NULL;
    apr_status_t rv;
    int i = 0;
    char* pass = NULL;


    if (passmax > INT_MAX / sizeof(int)) {
        CrtPrintf("Max string length is too big: %lu", passmax);
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

    for (; i < numOfThreads; ++i) {
        thd_ctx[i] = (ThreadContext*)apr_pcalloc(pool, sizeof(ThreadContext));
        thd_ctx[i]->Passmin = passmin;
        thd_ctx[i]->Passmax = passmax;
        thd_ctx[i]->Num = i + 1;
        thd_ctx[i]->Pass = (char*)apr_pcalloc(pool, sizeof(char)* ((size_t)passmax + 1));
        thd_ctx[i]->Indexes = (size_t*)apr_pcalloc(pool, (size_t)passmax * sizeof(size_t));
        thd_ctx[i]->Length = passmin;
        thd_ctx[i]->NumOfThreads = numOfThreads;
        rv = apr_thread_create(&thd_arr[i], thd_attr, MakeAttemptThreadFunc, thd_ctx[i], pool);
    }

    for (i = 0; i < numOfThreads; ++i) {
        rv = apr_thread_join(&rv, thd_arr[i]);
    }

    for (i = 0; i < numOfThreads; ++i) {
        (*attempts) += thd_ctx[i]->NumOfAttempts;
        if (thd_ctx[i]->Pass != NULL) {
            pass = thd_ctx[i]->Pass;
        }
    }
    return pass;
}

/**
 * Thread entry point
 */
void* APR_THREAD_FUNC MakeAttemptThreadFunc(apr_thread_t *thd, void *data)
{
    size_t maxIndex = 0;
    ThreadContext* tc = (ThreadContext*)data;

    maxIndex = strlen(ctx->Dict) - 1;

    for (; tc->Length <= tc->Passmax; ++tc->Length) {
        if (MakeAttempt(0, maxIndex, tc)) {
            goto result;
        } else if (apr_atomic_read32(&alreadyFound)) {
            break;
        }
    }
    tc->Pass = NULL;
result:
    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}

int MakeAttempt(const uint32_t pos, const size_t maxIndex, ThreadContext* tc)
{
    size_t i = 0;

    for (; i <= maxIndex; ++i) {
        tc->Indexes[pos] = i;

        if (pos == tc->Length - 1) {
            uint32_t j = 0;
            while (j < tc->Length) {
                size_t dictPosition = tc->Indexes[j];
                if (j > 0 || tc->NumOfThreads == 1 || tc->Num > 1 && dictPosition % tc->Num == 0 || tc->Num == 1 && dictPosition % tc->NumOfThreads != 0){
                    tc->Pass[j] = ctx->Dict[dictPosition];
                } else {
                    return FALSE;
                }
                ++j;
            }
            if (apr_atomic_read32(&alreadyFound)) {
                break;
            }
            ++(tc->NumOfAttempts);
            
            if (ctx->PfnHashCompare(ctx->Desired, tc->Pass, tc->Length)) {
                apr_atomic_set32(&alreadyFound, TRUE);
                return TRUE;
            }
        } else {
            if (MakeAttempt(pos + 1, maxIndex, tc)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}

const char* PrepareDictionary(const char* dict)
{
    const char* digitsClass = NULL;
    const char* lowCaseClass = NULL;
    const char* upperCaseClass = NULL;
    
    digitsClass = strstr(dict, DIGITS_TPL);
    lowCaseClass = strstr(dict, LOW_CASE_TPL);
    upperCaseClass = strstr(dict, UPPER_CASE_TPL);

    if (!digitsClass && !lowCaseClass && !upperCaseClass) {
        return dict;
    }
    if (digitsClass && lowCaseClass && upperCaseClass) {
        return DIGITS LOW_CASE UPPER_CASE;
    }
    if (!digitsClass && lowCaseClass && upperCaseClass) {
        return LOW_CASE UPPER_CASE;
    }
    if (digitsClass && !lowCaseClass && upperCaseClass) {
        return DIGITS UPPER_CASE;
    }
    if (digitsClass && lowCaseClass && !upperCaseClass) {
        return DIGITS LOW_CASE;
    }
    if (digitsClass && !lowCaseClass && !upperCaseClass) {
        return DIGITS;
    }
    if (!digitsClass && !lowCaseClass && upperCaseClass) {
        return UPPER_CASE;
    }
    if (!digitsClass && lowCaseClass && !upperCaseClass) {
        return LOW_CASE;
    }

    return dict;
}