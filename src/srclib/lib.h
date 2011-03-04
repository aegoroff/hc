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

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BINARY_THOUSAND 1024
#define FULL_TIME_FMT "%02u:%02u:%.3f"

#define COPYRIGHT_FMT_TRAIL "\nCopyright (C) 2009-2011 Alexander Egorov. All rights reserved.\n\n"
#ifdef _WIN64
    #define COPYRIGHT_FMT "\n%s x64" COPYRIGHT_FMT_TRAIL
#else
    #define COPYRIGHT_FMT "\n%s x86" COPYRIGHT_FMT_TRAIL
#endif

#define ALLOCATION_FAIL_FMT "Failed to allocate %lu bytes"
#define ALLOCATION_FAILURE_MESSAGE ALLOCATION_FAIL_FMT " in: %s:%d\n"

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
    uint32_t hours;
    uint32_t minutes;
    double   seconds;
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

extern uint32_t htoi(const char* ptr, int size);

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


#ifdef __cplusplus
}
#endif
#endif // HC_LIB_H_
