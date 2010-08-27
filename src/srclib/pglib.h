/*!
 * \brief   The file contains common solution library interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#ifndef HC_LIB_H_
#define HC_LIB_H_

#ifdef __cplusplus
extern "C" {
#endif

#define BINARY_THOUSAND 1024
#define FULL_TIME_FMT "%02u:%02u:%.3f"
#define COPYRIGHT_FMT "\n%s\nCopyright (C) 2009-2010 Alexander Egorov. All rights reserved.\n\n"
#define ALLOCATION_FAIL_FMT "Failed to allocate %lu bytes"

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
        double             size;
        unsigned long long sizeInBytes;
    } value;
} FileSize;

typedef struct Time {
    unsigned long hours;
    unsigned int  minutes;
    double        seconds;
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

extern void PrintSize(unsigned long long size);

extern FileSize NormalizeSize(unsigned long long size);

extern unsigned int htoi(const char* ptr, int size);

/*!
 * Prints new line into stdout
 */
extern void NewLine(void);

extern Time NormalizeTime(double seconds);
extern void PrintTime(Time seconds);

extern void StartTimer(void);
extern void StopTimer(void);
extern Time ReadElapsedTime(void);


#ifdef __cplusplus
}
#endif
#endif // HC_LIB_H_
