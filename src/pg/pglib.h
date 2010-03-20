/*
 * Copyright 2009 Alexander Egorov
 */

#ifndef PG_PGLIB_H_
#define PG_PGLIB_H_

#ifdef __cplusplus
extern "C" {
#endif

#define BINARY_THOUSAND 1024

/**
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
extern int CrtFprintf(FILE* file, __format_string const char *format, ...);
#else
extern int CrtFprintf(FILE* file, const char *format, ...);
#endif

void PrintSize(unsigned long long size);

extern unsigned int htoi (const char *ptr, int size);

extern void ReverseString(char* s, unsigned int left, unsigned int right);

int NextPermutation(int n, int* pIndexes);


#ifdef __cplusplus
}
#endif

#endif  // PG_PGLIB_H_
