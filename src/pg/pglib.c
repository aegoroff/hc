/*!
 * \brief   The file contains common solution library implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
*/

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef WIN32
#include <windows.h>
#endif
#include "pglib.h"


#define BIG_FILE_FORMAT "%.2f %s (%llu %s)" // greater or equal 1 Kb
#define SMALL_FILE_FORMAT "%llu %s" // less then 1 Kb
#define SEC_FMT "%.3f sec"
#define MIN_FMT "%d min "
#define HOURS_FMT "%d h "
#define SECONDS_PER_HOUR 3600
#define SECONDS_PER_MINUTE 60

// Defining min number values that causes number to prime ratio specifed
#define NUM_TO_PRIME_RATIO_5 1000
#define NUM_TO_PRIME_RATIO_10 65000
#define NUM_TO_PRIME_RATIO_12 500000
#define NUM_TO_PRIME_RATIO_15 9600000
#define NUM_TO_PRIME_RATIO_17 71000000
#define NUM_TO_PRIME_RATIO_18 200000000
#define NUM_TO_PRIME_RATIO_19 800000000
#define NUM_TO_PRIME_RATIO_20 2000000000

static char *sizes[] = {
    "bytes",
    "Kb",
    "Mb",
    "Gb",
    "Tb",
    "Pb",
    "Eb",
    "Zb",
    "Yb",
    "Bb",
    "GPb"
};

static double span = 0.0;
#ifdef WIN32
static LARGE_INTEGER freq = { 0 };
static LARGE_INTEGER time1 = { 0 };
static LARGE_INTEGER time2 = { 0 };

#else
static clock_t c0 = 0;
static clock_t c1 = 0;
#endif

void PrintSize(unsigned long long size)
{
    FileSize normalized = NormalizeSize(size);
    CrtPrintf(normalized.unit ? BIG_FILE_FORMAT : SMALL_FILE_FORMAT,
              normalized.value, sizes[normalized.unit], size, sizes[SizeUnitBytes]);
}

FileSize NormalizeSize(unsigned long long size)
{
    FileSize result = { 0 };
    result.unit = size == 0 ? SizeUnitBytes : floor(log(size) / log(BINARY_THOUSAND));
    if (result.unit == SizeUnitBytes) {
        result.value.sizeInBytes = size;
    } else {
        result.value.size = size / pow(BINARY_THOUSAND, floor(result.unit));
    }
    return result;
}

size_t CalculateMemorySize(size_t maxNum)
{
    size_t sz = 0;
    sz = maxNum < NUM_TO_PRIME_RATIO_5 ? maxNum : maxNum / 5;
    sz = maxNum < NUM_TO_PRIME_RATIO_10 ? sz : maxNum / 10;
    sz = maxNum < NUM_TO_PRIME_RATIO_12 ? sz : maxNum / 12;
    sz = maxNum < NUM_TO_PRIME_RATIO_15 ? sz : maxNum / 15;
    sz = maxNum < NUM_TO_PRIME_RATIO_17 ? sz : maxNum / 17;
    sz = maxNum < NUM_TO_PRIME_RATIO_18 ? sz : maxNum / 18;
    sz = maxNum < NUM_TO_PRIME_RATIO_19 ? sz : maxNum / 19;
    sz = maxNum < NUM_TO_PRIME_RATIO_20 ? sz : maxNum / 20;
    return sz;
}

int CrtPrintf(const char *format, ...)
{
    va_list params;
    int result = 0;
    va_start(params, format);
#ifdef __STDC_WANT_SECURE_LIB__
    result = vfprintf_s(stdout, format, params);
#else
    result = vfprintf(stdout, format, params);
#endif
    va_end(params);
    return result;
}

int CrtFprintf(FILE * file, const char *format, ...)
{
    va_list params;
    int result = 0;
    va_start(params, format);
#ifdef __STDC_WANT_SECURE_LIB__
    result = vfprintf_s(file, format, params);
#else
    result = vfprintf(file, format, params);
#endif
    va_end(params);
    return result;
}

unsigned int htoi(const char *ptr, int size)
{
    unsigned int value = 0;
    char ch = 0;
    int count = 0;

    if (ptr == NULL) {
        return value;
    }

    ch = *ptr;
    while (ch == ' ' || ch == '\t') {
        ch = *(++ptr);
        ++count;
    }

    for (;;) {
        if (count >= size) {
            return value;
        } else if (ch >= '0' && ch <= '9') {
            value = (value << 4) + (ch - '0');
        } else if (ch >= 'A' && ch <= 'F') {
            value = (value << 4) + (ch - 'A' + 10);
        } else if (ch >= 'a' && ch <= 'f') {
            value = (value << 4) + (ch - 'a' + 10);
        } else {
            return value;
        }
        ch = *(++ptr);
        ++count;
    }
}

int NextPermutation(int n, int *pIndexes)
{
    int k = n - 1;
    int t = 0;

    while (k > 0 && pIndexes[k] > pIndexes[k + 1]) {
        --k;
    }
    if (!k) {
        return 1;
    }
    t = k + 1;
    while (t < n && pIndexes[t + 1] > pIndexes[k]) {
        ++t;
    }
    pIndexes[k] ^= pIndexes[t] ^= pIndexes[k] ^= pIndexes[t];
    t = 0;
    while (t < (n - k) >> 1) {
        pIndexes[n - t] ^= pIndexes[k + 1 + t] ^= pIndexes[n - t] ^= pIndexes[k + 1 + t];
        ++t;
    }
    return 0;
}

void ReverseString(char *s, unsigned int left, unsigned int right)
{
    unsigned int i = 0;
    unsigned int j = 0;

    if (left >= right || right >= strlen(s)) {
        return;
    }

    for (i = left, j = right; i < j; ++i, --j) {
        *(s + i) ^= *(s + j);
        *(s + j) ^= *(s + i);
        *(s + i) ^= *(s + j);
    }
}

Time NormalizeTime(double seconds)
{
    Time result = { 0 };
    result.hours = seconds / SECONDS_PER_HOUR;
    result.minutes = ((unsigned long long)seconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    result.seconds = ((unsigned long long)seconds % SECONDS_PER_HOUR) % SECONDS_PER_MINUTE;
    result.seconds +=
        seconds - (result.hours * SECONDS_PER_HOUR + result.minutes * SECONDS_PER_MINUTE + result.seconds);
    return result;
}

void PrintTime(Time time)
{
    if (time.hours) {
        CrtPrintf(HOURS_FMT, time.hours);
    }
    if (time.minutes) {
        CrtPrintf(MIN_FMT, time.minutes);
    }
    CrtPrintf(SEC_FMT, time.seconds);
}

void NewLine(void)
{
    CrtPrintf("\n");
}

void StartTimer(void)
{
#ifdef WIN32
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time1);
#else
    c0 = clock();
#endif
}

void StopTimer(void)
{
#ifdef WIN32
    QueryPerformanceCounter(&time2);
    span = (double)(time2.QuadPart - time1.QuadPart) / (double)freq.QuadPart;
#else
    c1 = clock();
    span = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
#endif
}

Time ReadElapsedTime(void)
{
    return NormalizeTime(span);
}
