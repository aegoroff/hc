/*
 * Copyright 2009 Alexander Egorov
 */

#ifndef PG_PGLIB_H_
#define PG_PGLIB_H_

#ifdef __cplusplus
extern "C" {
#endif

#define BINARY_THOUSAND 1024
#define FULL_TIME_FMT "%02d:%02d:%.3f"

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

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
            double size;
            unsigned long long sizeInBytes;
        } value;
    } FileSize;

    typedef struct Time {
        unsigned long hours;
        unsigned int minutes;
        double seconds;
    } Time;

    /*!
     * Calculates temp memory buffer size in size_t elements.
     * So to alloc buffer in bytes just multiply return value to sizeof(size_t)
     */
    extern size_t CalculateMemorySize(size_t maxNum);

#ifdef __STDC_WANT_SECURE_LIB__
    extern int CrtPrintf(__format_string const char *format, ...);
#else
    extern int CrtPrintf(const char *format, ...);
#endif

#ifdef __STDC_WANT_SECURE_LIB__
    extern int CrtFprintf(FILE * file, __format_string const char *format, ...);
#else
    extern int CrtFprintf(FILE * file, const char *format, ...);
#endif

    extern void PrintSize(unsigned long long size);

    extern FileSize NormalizeSize(unsigned long long size);

    extern unsigned int htoi(const char *ptr, int size);

    extern void ReverseString(char *s, unsigned int left, unsigned int right);

    /*!
     * Prints new line into stdout
     */
    extern void NewLine(void);

    extern int NextPermutation(int n, int *pIndexes);

    extern Time NormalizeTime(double seconds);
    extern void PrintTime(double seconds);


#ifdef __cplusplus
}
#endif
#endif // PG_PGLIB_H_
