#include <stdio.h>
#include <stdarg.h>
#include "pglib.h"

// Defining min number values that causes number to prime ratio specifed
#define NUM_TO_PRIME_RATIO_5 1000
#define NUM_TO_PRIME_RATIO_10 65000
#define NUM_TO_PRIME_RATIO_12 500000
#define NUM_TO_PRIME_RATIO_15 9600000
#define NUM_TO_PRIME_RATIO_17 71000000
#define NUM_TO_PRIME_RATIO_18 200000000
#define NUM_TO_PRIME_RATIO_19 800000000
#define NUM_TO_PRIME_RATIO_20 2000000000

size_t CalculateMemorySize(size_t maxNum) {
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

int CrtPrintf(const char *format, ...) {
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
