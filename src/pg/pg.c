/*
 * Copyright 2009 Alexander Egorov
 */

#include <stdio.h>
#include <io.h>

#ifdef WIN32
#ifndef _WIN32_WINNT    // Allow use of features specific to Windows XP or later.
#define _WIN32_WINNT 0x0501 // Change this to the appropriate value to target other versions of Windows.
#endif

#include <windows.h>
#endif

#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "pglib.h"

void PrintCopyright(void)
{
    CrtPrintf("\nPrimes Generator\nCopyright (C) 2009 Alexander Egorov.  All rights reserved.\n\n");
}

void PrintUsage(void)
{
    CrtPrintf("usage: pg <max number> [filename or full path]\n");
}

int main(int argc, char *argv[])
{
    FILE *file = NULL;
    size_t ixCurr = 0;  // current found index
    size_t *prime = NULL;
    size_t i = 0;
    size_t j = 0;
    size_t sz = 0;
    double span = 0;
    size_t num = 0;
    size_t szResult = 0;
    Time time = { 0 };

#ifdef WIN32
    LARGE_INTEGER freq = { 0 };
    LARGE_INTEGER time1 = { 0 };
    LARGE_INTEGER time2 = { 0 };
#else
    clock_t c0 = 0;
    clock_t c1 = 0;
#endif

    PrintCopyright();

    if (argc < 2) {
        PrintUsage();
        return EXIT_FAILURE;
    }
#ifdef __STDC_WANT_SECURE_LIB__
    sscanf_s(argv[1], "%d", &num);
#else
    sscanf(argv[1], "%d", &num);
#endif

#ifdef WIN32
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time1);
#else
    c0 = clock();
#endif

    if (argc > 2) {
#ifdef __STDC_WANT_SECURE_LIB__
        fopen_s(&file, argv[2], "w+");
#else
        file = fopen(argv[2], "w+");
#endif
    } else {
        file = stdout;
    }

    sz = CalculateMemorySize(num);
    prime = (size_t *) calloc(sz, sizeof(size_t));
    if (prime == NULL) {
        CrtPrintf("Cannot allocate %li bytes", sz * sizeof(size_t));
        if (argc > 2) {
            fclose(file);
        }
        return EXIT_FAILURE;
    }

    *prime = 2; // the first prime
    i = 3;  // The second prime
    while (prime[ixCurr] <= num && i < UINT_MAX) {
        for (j = 0; j <= ixCurr; ++j) {
            // This check must be first! Otherwise we will be two times slower
            if (prime[j] > sqrt(i)) {
                prime[++ixCurr] = i;    // IMPORTANT: prefix ++ not postfix !!!
                break;
            }
            if ((i % prime[j]) == 0) {
                break;
            }
        }
        ++i;
    }

    i = 0;
    while (i < ixCurr) {
        fprintf(file, "%i\n", prime[i]);
        ++i;
    }
    free(prime);
    if (argc > 2) {
        fflush(file);
        szResult = _filelength(file->_file);
        fclose(file);
    }
#ifdef WIN32
    QueryPerformanceCounter(&time2);
    span = (double)(time2.QuadPart - time1.QuadPart) / (double)freq.QuadPart;
#else
    c1 = clock();
    span = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
#endif
    time = NormalizeTime(span);

    CrtPrintf
        ("\nMax number:\t\t\t%li\nExecution time:\t\t\t" FULL_TIME_FMT
         "\nPrimes found:\t\t\t%i\nThe number to found ratio:\t%g\n", num, time.hours, time.minutes, time.seconds,
         i - 1, num / (double)i);

    if (argc > 2) {
        CrtPrintf("Result file:\t\t\t%s\nResult file size:\t\t", argv[2]);
        PrintSize(szResult);
        CrtPrintf("\n");
    }
    return EXIT_SUCCESS;
}
