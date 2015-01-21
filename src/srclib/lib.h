/*!
 * \brief   The file contains common solution library interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#ifndef HC_LIB_H_
#define HC_LIB_H_

#include <stdio.h>
#include "types.h"

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

#ifdef WIN32
#define NEW_LINE "\n"
#else
#define NEW_LINE "\n"
#endif

#define COPYRIGHT_FMT_TRAIL NEW_LINE "Copyright (C) 2009-2015 Alexander Egorov. All rights reserved." NEW_LINE NEW_LINE
#ifdef _WIN64
    #define COPYRIGHT_FMT NEW_LINE "%s x64" COPYRIGHT_FMT_TRAIL
#else
    #define COPYRIGHT_FMT NEW_LINE "%s x86" COPYRIGHT_FMT_TRAIL
#endif

#define ALLOCATION_FAIL_FMT "Failed to allocate %Iu bytes"
#define ALLOCATION_FAILURE_MESSAGE ALLOCATION_FAIL_FMT " in: %s:%d" NEW_LINE

typedef enum {
    SizeUnitBytes = 0,
    SizeUnitKBytes = 1,
    SizeUnitMBytes = 2,
    SizeUnitGBytes = 3,
    SizeUnitTBytes = 4,
    SizeUnitPBytes = 5,
    SizeUnitEBytes = 6,
    SizeUnitZBytes = 7,
    SizeUnitYBytes = 8,
    SizeUnitBBytes = 9,
    SizeUnitGPBytes = 10
} SizeUnit;

typedef struct FileSize {
    SizeUnit unit;
    // Union of either size in bytes or size it KBytes, MBytes etc.
    union {
        double   size;
        uint64_t sizeInBytes;
    } value;
} FileSize;

typedef struct Time {
    uint32_t years;
    uint32_t days;
    uint32_t hours;
    uint32_t minutes;
    double   seconds;
    double   total_seconds;
} Time;

#ifdef __STDC_WANT_SECURE_LIB__
extern int CrtPrintf(__format_string const char* format, ...);
#else
extern int CrtPrintf(const char* format, ...);
#endif

#ifdef __STDC_WANT_SECURE_LIB__
extern int CrtFprintf(FILE* file, __format_string const char* format, ...);
#else
extern int CrtFprintf(FILE* file, const char* format, ...);
#endif

extern void PrintSize(uint64_t size);

extern FileSize NormalizeSize(uint64_t size);

/*!
 * Prints new line into stdout
 */
extern void NewLine(void);

extern Time NormalizeTime(double seconds);

extern void StartTimer(void);
extern void StopTimer(void);
extern Time ReadElapsedTime(void);
extern void SizeToString(uint64_t size, size_t strSize, char* str);
extern void TimeToString(Time time, size_t strSize, char* str);
extern void HexStrintToByteArray(const char* str, uint8_t* bytes, size_t sz);
extern uint32_t htoi(const char* ptr, int size);
extern uint32_t GetProcessorCount(void);
extern int CountDigitsIn(double x);
extern const char* GetFileName(const char *path);


#ifdef __cplusplus
}
#endif
#endif // HC_LIB_H_
