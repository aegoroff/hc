// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains common solution library interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#ifndef LINQ2HASH_LIB_H_
#define LINQ2HASH_LIB_H_

#include <stdio.h>
#include "types.h"

 /* internationalization support via gettext/libintl */
#ifdef USE_GETTEXT
# include <libgnuintl.h>
# define _(str) gettext(str)
# ifdef _WIN32
#  define LOCALEDIR "./"
# else /* _WIN32 */
#  define LOCALEDIR "/usr/share/locale"
# endif /* _WIN32 */
#else
# define _(str) (str)
#endif /* USE_GETTEXT */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BYTE_CHARS_SIZE
#define BYTE_CHARS_SIZE 2   // byte representation string length
#endif

#define BINARY_THOUSAND 1024
#define FULL_TIME_FMT "%02u:%02u:%.3f"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifdef WIN32
#define NEW_LINE "\n"
#else
#define NEW_LINE "\n"
#endif

#define COPYRIGHT_FMT_TRAIL NEW_LINE "Copyright (C) 2009-2017 Alexander Egorov. All rights reserved." NEW_LINE NEW_LINE
#ifdef _WIN64
    #define COPYRIGHT_FMT NEW_LINE "%s x64" COPYRIGHT_FMT_TRAIL
#else
    #define COPYRIGHT_FMT NEW_LINE "%s x86" COPYRIGHT_FMT_TRAIL
#endif

#define ALLOCATION_FAIL_FMT "Failed to allocate %Iu bytes"
#define ALLOCATION_FAILURE_MESSAGE ALLOCATION_FAIL_FMT " in: %s:%d" NEW_LINE

typedef enum {
    size_unit_bytes = 0,
    size_unit_kbytes = 1,
    size_unit_mbytes = 2,
    size_unit_gbytes = 3,
    size_unit_tbytes = 4,
    size_unit_pbytes = 5,
    size_unit_ebytes = 6,
    size_unit_zbytes = 7,
    size_unit_ybytes = 8,
    size_unit_bbytes = 9,
    size_unit_gpbytes = 10
} size_unit_t;

typedef struct lib_file_size {
    size_unit_t unit;
    // Union of either size in bytes or size it KBytes, MBytes etc.
    union {
        double   size;
        uint64_t size_in_bytes;
    } value;
} lib_file_size_t;

typedef struct lib_time {
    uint32_t years;
    uint32_t days;
    uint32_t hours;
    uint32_t minutes;
    double   seconds;
    double   total_seconds;
} lib_time_t;

#ifdef __STDC_WANT_SECURE_LIB__
extern int lib_printf(__format_string const char* format, ...);
#else
extern int lib_printf(const char* format, ...);
#endif

#ifdef __STDC_WANT_SECURE_LIB__
extern int lib_fprintf(FILE* file, __format_string const char* format, ...);
#else
extern int lib_fprintf(FILE* file, const char* format, ...);
#endif

#ifdef __STDC_WANT_SECURE_LIB__
extern int lib_sprintf(char* buffer, __format_string const char* format, ...);
#else
extern int lib_sprintf(char* buffer, const char* format, ...);
#endif

extern void lib_print_size(uint64_t size);

extern lib_file_size_t lib_normalize_size(uint64_t size);

/*!
 * Prints new line into stdout
 */
extern void lib_new_line(void);


/**
 * \brief converts time in seconds into structure that can be easly interpreted into appropriate form
 * \param seconds time in seconds
 * \return time in second converted into lib_time_t structure
 */
extern lib_time_t lib_normalize_time(double seconds);

extern void lib_start_timer(void);
extern void lib_stop_timer(void);
extern lib_time_t lib_read_elapsed_time(void);
extern void lib_size_to_string(uint64_t size, char* str);
extern void lib_time_to_string(lib_time_t time, char* str);
extern void lib_hex_str_2_byte_array(const char* str, uint8_t* bytes, size_t sz);
extern uint32_t lib_htoi(const char* ptr, int size);
extern uint32_t lib_get_processor_count(void);
extern int lib_count_digits_in(double x);
extern const char* lib_get_file_name(const char *path);


#ifdef __cplusplus
}
#endif
#endif // LINQ2HASH_LIB_H_
