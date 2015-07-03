/*!
 * \brief   The file contains common solution library implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2015
 */

#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef WIN32
#include <windows.h>
#endif
#include "lib.h"


#define BIG_FILE_FORMAT "%.2f %s (%llu %s)" // greater or equal 1 Kb
#define SMALL_FILE_FORMAT "%llu %s" // less then 1 Kb
#define SEC_FMT "%.3f sec"
#define MIN_FMT "%u min "
#define HOURS_FMT "%u hr "
#define DAYS_FMT "%u days "
#define YEARS_FMT "%u years "
#define SECONDS_PER_YEAR 31536000
#define SECONDS_PER_DAY 86400
#define SECONDS_PER_HOUR 3600
#define SECONDS_PER_MINUTE 60
#define INT64_BITS_COUNT 64

static char* sizes[] = {
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
static LARGE_INTEGER freq = {0};
static LARGE_INTEGER time1 = {0};
static LARGE_INTEGER time2 = {0};

#else
static clock_t c0 = 0;
static clock_t c1 = 0;
#endif

uint32_t GetProcessorCount(void) {
#ifdef WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (uint32_t)sysinfo.dwNumberOfProcessors;
#else
    return (uint32_t)sysconf( _SC_NPROCESSORS_ONLN );
#endif 
}

void PrintSize(uint64_t size) {
    FileSize normalized = NormalizeSize(size);
    CrtPrintf(normalized.unit ? BIG_FILE_FORMAT : SMALL_FILE_FORMAT, //-V510
              normalized.value, sizes[normalized.unit], size, sizes[SizeUnitBytes]);
}

void SizeToString(uint64_t size, size_t strSize, char* str) {
    FileSize normalized = NormalizeSize(size);

    if(str == NULL) {
        return;
    }
    sprintf_s(str, strSize, normalized.unit ? BIG_FILE_FORMAT : SMALL_FILE_FORMAT, //-V510
              normalized.value, sizes[normalized.unit], size, sizes[SizeUnitBytes]);
}

uint32_t htoi(const char* ptr, int size) {
    uint32_t value = 0;
    char ch = 0;
    int count = 0;

    if(ptr == NULL || size <= 0) {
        return value;
    }

    ch = ptr[count];
    for(;;) {
        if(ch == ' ' || ch == '\t') {
            goto nextChar;
        }
        if((ch >= '0') && (ch <= '9')) {
            value = (value << 4) + (ch - '0');
        }
        else if((ch >= 'A') && (ch <= 'F')) {
            value = (value << 4) + (ch - 'A' + 10);
        }
        else if((ch >= 'a') && (ch <= 'f')) {
            value = (value << 4) + (ch - 'a' + 10);
        }
        else {
            return value;
        }
    nextChar:
        if(++count >= size) {
            return value;
        }
        ch = ptr[count];
    }
}

void HexStrintToByteArray(const char* str, uint8_t* bytes, size_t sz) {
    size_t i = 0;
    size_t to = MIN(sz, strlen(str) / BYTE_CHARS_SIZE);

    for(; i < to; i++) {
        bytes[i] = (uint8_t)htoi(str + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}

uint64_t ilog(uint64_t x) {
    uint64_t y = 0;
    uint64_t n = INT64_BITS_COUNT;
    int c = INT64_BITS_COUNT / 2;

    do {
        y = x >> c;
        if(y != 0) {
            n -= c;
            x = y;
        }
        c >>= 1;
    }
    while(c != 0);
    n -= x >> (INT64_BITS_COUNT - 1);
    return (INT64_BITS_COUNT - 1) - (n - x);
}

FileSize NormalizeSize(uint64_t size) {
    FileSize result = {0};
    result.unit = size == 0 ? SizeUnitBytes : ilog(size) / ilog(BINARY_THOUSAND);
    if(result.unit == SizeUnitBytes) {
        result.value.sizeInBytes = size;
    }
    else {
        result.value.size = size / pow(BINARY_THOUSAND, result.unit);
    }
    return result;
}

int CrtPrintf(__format_string const char* format, ...) {
    va_list params = NULL;
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

int CrtFprintf(FILE* file, __format_string const char* format, ...) {
    va_list params = NULL;
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

Time NormalizeTime(double seconds) {
    Time result = {0};
    double tmp = 0;

    result.total_seconds = seconds;
    result.years = seconds / SECONDS_PER_YEAR;
    result.days = ((uint64_t)seconds % SECONDS_PER_YEAR) / SECONDS_PER_DAY;
    result.hours = (((uint64_t)seconds % SECONDS_PER_YEAR) % SECONDS_PER_DAY) / SECONDS_PER_HOUR;
    result.minutes = ((uint64_t)seconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    result.seconds = ((uint64_t)seconds % SECONDS_PER_HOUR) % SECONDS_PER_MINUTE;
    tmp = result.seconds;
    result.seconds +=
            seconds -
            ((double)(result.years * SECONDS_PER_YEAR) + (double)(result.days * SECONDS_PER_DAY) + (double)(result.hours * SECONDS_PER_HOUR) + (double)(result.minutes * SECONDS_PER_MINUTE) + result.seconds);
    if(result.seconds > 60) {
        result.seconds = tmp; // HACK
    }
    return result;
}

void TimeToString(Time time, size_t strSize, char* str) {
    if((str == NULL) || (strSize == 0)) {
        return;
    }

    if(time.years) {
        sprintf_s(str, strSize, YEARS_FMT DAYS_FMT HOURS_FMT MIN_FMT SEC_FMT, time.years, time.days, time.hours, time.minutes, time.seconds);
        return;
    }
    if(time.days) {
        sprintf_s(str, strSize, DAYS_FMT HOURS_FMT MIN_FMT SEC_FMT, time.days, time.hours, time.minutes, time.seconds);
        return;
    }
    if(time.hours) {
        sprintf_s(str, strSize, HOURS_FMT MIN_FMT SEC_FMT, time.hours, time.minutes, time.seconds);
        return;
    }
    if(time.minutes) {
        sprintf_s(str, strSize, MIN_FMT SEC_FMT, time.minutes, time.seconds);
        return;
    }
    sprintf_s(str, strSize, SEC_FMT, time.seconds);
}

void NewLine(void) {
    CrtPrintf(NEW_LINE);
}

void StartTimer(void) {
#ifdef WIN32
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time1);
#else
    c0 = clock();
#endif
}

void StopTimer(void) {
#ifdef WIN32
    QueryPerformanceCounter(&time2);
    span = (double)(time2.QuadPart - time1.QuadPart) / (double)freq.QuadPart;
#else
    c1 = clock();
    span = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
#endif
}

Time ReadElapsedTime(void) {
    return NormalizeTime(span);
}

int CountDigitsIn(double x) {
    int result = 0;
    long long n = x;
    do {
        ++result;
        n /= 10;
    }
    while(n > 0);
    return result;
}

const char* GetFileName(const char* path) {
    const char* filename = NULL;
    if(path == NULL) {
        return path;
    }
    filename = strrchr(path, '\\');

    if(filename == NULL) {
        filename = path;
    }
    else {
        filename++;
    }
    return filename;
}
