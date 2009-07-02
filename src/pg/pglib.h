#ifndef PG_H_
#define PG_H_

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif  // PG_H_
