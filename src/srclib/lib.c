/*!
 * \brief   The file contains common solution library implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <stdarg.h>
#include <string.h>
#include <math.h>

#ifdef _MSC_VER

#include <Windows.h>

#else

#include <time.h>

#ifdef __APPLE_CC__

#include <zconf.h>

#else

#include <sys/sysinfo.h>

#endif
#endif

#include "lib.h"

/* Non-zero while stdout output from lib_printf should be swallowed (set by the
   Zig test driver via bf_shim_set_output_suspended). See lib_printf. */
int g_lib_output_suspended = 0;

/*
   lib_ - public members
   prdlib_ - private members
*/

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

// forwards
static uint64_t prlib_ilog(uint64_t x);

static char* lib_sizes[] = {
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

static double lib_span = 0.0;

#ifdef _MSC_VER
static LARGE_INTEGER lib_freq = {0};
static LARGE_INTEGER lib_time1 = {0};
static LARGE_INTEGER lib_time2 = {0};

#else
#define BILLION 1E9

static struct timespec lib_start = {0};
static struct timespec lib_finish = {0};
#endif

uint32_t lib_get_processor_count(void) {
#ifdef _MSC_VER
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (uint32_t) sysinfo.dwNumberOfProcessors;
#elif __APPLE_CC__
    return (uint32_t) sysconf(_SC_NPROCESSORS_ONLN);
#else
    return (uint32_t) get_nprocs();
#endif
}

void lib_print_size(uint64_t size) {
    const lib_file_size_t normalized = lib_normalize_size(size);

    if (normalized.unit) {
        lib_printf(BIG_FILE_FORMAT, normalized.size, lib_sizes[normalized.unit], normalized.size_in_bytes,
                   lib_sizes[size_unit_bytes]);
    } else {
        lib_printf(SMALL_FILE_FORMAT, normalized.size_in_bytes, lib_sizes[size_unit_bytes]);
    }
}

void lib_size_to_string(uint64_t size, char *str, size_t str_size) {
    const lib_file_size_t normalized = lib_normalize_size(size);

    if (str == NULL || str_size == 0) {
        return;
    }
    if (normalized.unit) {
        lib_snprintf(str, str_size, BIG_FILE_FORMAT, normalized.size, lib_sizes[normalized.unit],
                     normalized.size_in_bytes, lib_sizes[size_unit_bytes]);
    } else {
        lib_snprintf(str, str_size, SMALL_FILE_FORMAT, normalized.size_in_bytes, lib_sizes[size_unit_bytes]);
    }
}

uint32_t lib_htoi(const char *ptr, int size) {
    uint32_t value = 0;
    while (size-- > 0 && ptr != NULL) {
        if (*ptr >= '0' && *ptr <= '9') {
            value = (value << 4U) + (*ptr - '0');
        } else if (*ptr >= 'A' && *ptr <= 'F') {
            value = (value << 4U) + ((*ptr - 'A') + 10);
        } else if (*ptr >= 'a' && *ptr <= 'f') {
            value = (value << 4U) + ((*ptr - 'a') + 10);
        } else if (value > 0) {
            return value;
        }
        ++ptr;
    }
    return value;
}

void lib_hex_str_2_byte_array(const char* str, uint8_t* bytes, size_t sz) {
    size_t i = 0;
    const size_t to = MIN(sz, strlen(str) / BYTE_CHARS_SIZE);

    for(; i < to; i++) {
        bytes[i] = (uint8_t) lib_htoi(str + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}

uint64_t prlib_ilog(uint64_t x) {
    uint64_t n = INT64_BITS_COUNT;
    uint32_t c = INT64_BITS_COUNT / 2;

    do {
        const uint64_t y = x >> c;
        if(y != 0) {
            n -= c;
            x = y;
        }
        c >>= 1U;
    } while(c != 0);
    n -= x >> (INT64_BITS_COUNT - 1U);
    return (INT64_BITS_COUNT - 1) - (n - x);
}

lib_file_size_t lib_normalize_size(uint64_t size) {
    lib_file_size_t result = {0};
    result.unit = size == 0 ? size_unit_bytes : prlib_ilog(size) / prlib_ilog(BINARY_THOUSAND);
    result.size_in_bytes = size;
    if(result.unit != size_unit_bytes) {
        result.size = size / pow(BINARY_THOUSAND, result.unit);
    }
    // ReSharper disable once CppSomeObjectMembersMightNotBeInitialized
    return result;
}

#ifdef _MSC_VER

int lib_printf(__format_string const char* format, ...) {
#else

int lib_printf(const char* format, ...) {
#endif
    /* When suspended, swallow all stdout output. Used by the Zig test driver:
       the brute-force C path prints probe/timings/result here via vfprintf on
       fd 1, which is the same fd zig's --listen=- test IPC multiplexes on, so
       unsuppressed C output desyncs the protocol. The release binary never sets
       this (output is always wanted there). */
    if (g_lib_output_suspended) {
        return 0;
    }
    va_list params;
    int result;
    va_start(params, format);
#ifdef __STDC_WANT_SECURE_LIB__
    result = vfprintf_s(stdout, format, params);
#else
    result = vfprintf(stdout, format, params);
    fflush(stdout);
#endif
    va_end(params);
    return result;
}

#ifdef _MSC_VER

int lib_fprintf(FILE* file, __format_string const char* format, ...) {
#else

int lib_fprintf(FILE* file, const char* format, ...) {
#endif
    va_list params;
    int result;
    va_start(params, format);
#ifdef __STDC_WANT_SECURE_LIB__
    result = vfprintf_s(file, format, params);
#else
    result = vfprintf(file, format, params);
#endif
    va_end(params);
    return result;
}

#ifdef _MSC_VER

int lib_snprintf(char* buffer, size_t size, __format_string const char* format, ...) {
#else

int lib_snprintf(char* buffer, size_t size, const char* format, ...) {
#endif
    va_list params;
    int result;

    if(buffer == NULL || size == 0 || format == NULL) {
        return -1;
    }

    va_start(params, format);
#ifdef __STDC_WANT_SECURE_LIB__
    result = vsnprintf_s(buffer, size, _TRUNCATE, format, params);
#else
    result = vsnprintf(buffer, size, format, params);
#endif
    va_end(params);
    return result;
}

#ifdef _MSC_VER

int lib_wcsnprintf(wchar_t* buffer, size_t size, __format_string const wchar_t* format, ...) {
#else

int lib_wcsnprintf(wchar_t* buffer, size_t size, const wchar_t* format, ...) {
#endif
    va_list params;
    int result;

    if(buffer == NULL || size == 0 || format == NULL) {
        return -1;
    }

    va_start(params, format);
#ifdef __STDC_WANT_SECURE_LIB__
    result = _vsnwprintf_s(buffer, size, _TRUNCATE, format, params);
#else
    result = vswprintf(buffer, size, format, params);
#endif
    va_end(params);
    return result;
}


lib_time_t lib_normalize_time(double seconds) {
    lib_time_t result = {0};

    result.total_seconds = seconds;
    result.years = seconds / SECONDS_PER_YEAR;
    result.days = ((uint64_t) seconds % SECONDS_PER_YEAR) / SECONDS_PER_DAY;
    result.hours = (((uint64_t) seconds % SECONDS_PER_YEAR) % SECONDS_PER_DAY) / SECONDS_PER_HOUR;
    result.minutes = ((uint64_t) seconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    result.seconds = ((uint64_t) seconds % SECONDS_PER_HOUR) % SECONDS_PER_MINUTE;
    double tmp = result.seconds;
    result.seconds +=
            seconds -
            ((double) (result.years * SECONDS_PER_YEAR) + (double) (result.days * SECONDS_PER_DAY) +
             (double) (result.hours
                       * SECONDS_PER_HOUR) + (double) (result.minutes * SECONDS_PER_MINUTE) + result.seconds);
    if(result.seconds > 60) {
        result.seconds = tmp; // HACK
    }
    return result;
}

void lib_time_to_string(const lib_time_t* time, char* str, size_t str_size) {
    if(str == NULL || str_size == 0) {
        return;
    }

    if(time->years) {
        lib_snprintf(str, str_size, YEARS_FMT DAYS_FMT HOURS_FMT MIN_FMT SEC_FMT, time->years, time->days,
                     time->hours, time->minutes, time->seconds);
        return;
    }
    if(time->days) {
        lib_snprintf(str, str_size, DAYS_FMT HOURS_FMT MIN_FMT SEC_FMT, time->days, time->hours, time->minutes,
                     time->seconds);
        return;
    }
    if(time->hours) {
        lib_snprintf(str, str_size, HOURS_FMT MIN_FMT SEC_FMT, time->hours, time->minutes, time->seconds);
        return;
    }
    if(time->minutes) {
        lib_snprintf(str, str_size, MIN_FMT SEC_FMT, time->minutes, time->seconds);
        return;
    }
    lib_snprintf(str, str_size, SEC_FMT, time->seconds);
}

void lib_new_line(void) {
    lib_printf(NEW_LINE);
}

void lib_start_timer(void) {
#ifdef _MSC_VER
    QueryPerformanceFrequency(&lib_freq);
    QueryPerformanceCounter(&lib_time1);
#else
    clock_gettime(CLOCK_REALTIME, &lib_start);
#endif
}

void lib_stop_timer(void) {
#ifdef _MSC_VER
    QueryPerformanceCounter(&lib_time2);
    lib_span = (double) (lib_time2.QuadPart - lib_time1.QuadPart) / (double) lib_freq.QuadPart;
#else
    clock_gettime(CLOCK_REALTIME, &lib_finish);
    lib_span = (lib_finish.tv_sec - lib_start.tv_sec) + (lib_finish.tv_nsec - lib_start.tv_nsec) / BILLION;
#endif
}

lib_time_t lib_read_elapsed_time(void) {
    return lib_normalize_time(lib_span);
}

int lib_count_digits_in(double x) {
    int result = 0;
    long long n = x;
    do {
        ++result;
        n /= 10;
    } while(n > 0);
    return result;
}

const char* lib_get_file_name(const char* path) {
    if(path == NULL) {
        return path;
    }

#ifdef _WIN32
    const int path_sep = '\\';
#else
    const int path_sep = '/';
#endif

    const char* filename = strrchr(path, path_sep);

    if(filename == NULL) {
        filename = path;
    } else {
        filename++;
    }
    return filename;
}

char* lib_ltrim(char* str, const char* seps) {
    size_t totrim = 0;

    if(str == NULL) {
        return str;
    }

    if(seps == NULL) {
        seps = "\t\n\v\f\r ";
    }

    totrim = strspn(str, seps);
    if(totrim > 0) {
        size_t len = strlen(str);
        if(totrim == len) {
            str[0] = '\0';
        } else {
            memmove(str, str + totrim, len + 1 - totrim);
        }
    }

    return str;
}

char* lib_rtrim(char* str, const char* seps) {
    if(str == NULL) {
        return str;
    }

    if(seps == NULL) {
        seps = "\t\n\v\f\r ";
    }

    size_t i = strlen(str) - 1;
    while(i < SIZE_MAX && strchr(seps, str[i]) != NULL) {
        str[i] = '\0';
        i--;
    }

    return str;
}

char* lib_trim(char* str, const char* seps) {
    return lib_ltrim(lib_rtrim(str, seps), seps);
}
