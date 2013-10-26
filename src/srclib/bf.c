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
#include "bf.h"
#include "output.h"
#include "encoding.h"

uint32_t length;
uint64_t noOfAttempts;
BruteForceContext* ctx;
size_t*        indexes;
char*       pass;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;


int MakeAttempt(const uint32_t pos, const size_t maxIndex);
const char* PrepareDictionary(const char* dict);

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_size_t  hashLength,
               void (*digestFunction)(apr_byte_t* digest, const char* string, const apr_size_t inputLen),
               BOOL noProbe,
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
        str = BruteForce(passmin, passmax, dict, hash, &attempts, CreateDigest, pool);
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
                 apr_pool_t*       pool)
{
    size_t maxIndex = 0;
    noOfAttempts = 0;

    if (passmax > INT_MAX / sizeof(int)) {
        CrtPrintf("Max string length is too big: %lu", passmax);
        return NULL;
    }

    pass = (char*)apr_pcalloc(pool, sizeof(char) * ((size_t)passmax + 1));
    if (pass == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, passmax + 1, __FILE__, __LINE__);
        return NULL;
    }
    indexes = (size_t*)apr_pcalloc(pool, (size_t)passmax * sizeof(size_t));
    if (indexes == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, (size_t)passmax * sizeof(int), __FILE__, __LINE__);
        return NULL;
    }

    ctx = (BruteForceContext*)apr_pcalloc(pool, sizeof(BruteForceContext));
    ctx->Desired = PfnHashPrepare(hash, pool);
    ctx->PfnHashCompare = CompareHashAttempt;
    ctx->Dict = PrepareDictionary(dict);
    maxIndex = strlen(ctx->Dict) - 1;
    length = passmin;
    for (; length <= passmax; ++length) {
        if (MakeAttempt(0, maxIndex)) {

            goto result;
        }
    }
    pass = NULL;
result:
    *attempts = noOfAttempts;
    return pass;
}

int MakeAttempt(const uint32_t pos, const size_t maxIndex)
{
    size_t i = 0;

    for (; i <= maxIndex; ++i) {
        indexes[pos] = i;

        if (pos == length - 1) {
            uint32_t j = 0;
            for (; j < length; ++j) {
                pass[j] = ctx->Dict[indexes[j]];
            }
            ++noOfAttempts;

            if (ctx->PfnHashCompare(ctx->Desired, pass, length)) {
                return TRUE;
            }
        } else {
            if (MakeAttempt(pos + 1, maxIndex)) {
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