/*!
 * \brief   The file contains prime generator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
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

#define INPUT_FMT "%d"
#define FILE_OPEN_FMT "w+"
#define APP_NAME "Primes Generator"

void PrintCopyright(void)
{
    CrtPrintf(COPYRIGHT_FMT, APP_NAME);
}

void PrintUsage(void)
{
    CrtPrintf("usage: pg <max number> [filename or full path]\n");
}

int main(int argc, char* argv[])
{
    FILE* file = NULL;
    size_t ixCurr = 0;  // current found index
    size_t* prime = NULL;
    size_t i = 0;
    size_t j = 0;
    size_t sz = 0;
    size_t num = 0;
    size_t szResult = 0;
    Time time = { 0 };

    PrintCopyright();

    if (argc < 2) {
        PrintUsage();
        return EXIT_FAILURE;
    }
#ifdef __STDC_WANT_SECURE_LIB__
    sscanf_s(argv[1], INPUT_FMT, &num);
#else
    sscanf(argv[1], INPUT_FMT, &num);
#endif

    StartTimer();

    if (argc > 2) {
#ifdef __STDC_WANT_SECURE_LIB__
        fopen_s(&file, argv[2], FILE_OPEN_FMT);
#else
        file = fopen(argv[2], FILE_OPEN_FMT);
#endif
    } else {
        file = stdout;
    }

    sz = CalculateMemorySize(num);
    prime = (size_t*)calloc(sz, sizeof(size_t));
    if (prime == NULL) {
        CrtPrintf(ALLOCATION_FAIL_FMT, sz * sizeof(size_t));
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
        fprintf(file, "%lu\n", prime[i]);
        ++i;
    }
    free(prime);
    if (argc > 2) {
        fflush(file);
        szResult = _filelength(file->_file);
        fclose(file);
    }
    StopTimer();
    time = ReadElapsedTime();

    CrtPrintf
        ("\nMax number:\t\t\t%d\nExecution time:\t\t\t" FULL_TIME_FMT
        "\nPrimes found:\t\t\t%d\nThe number to found ratio:\t%g\n",
        num,
        time.hours,
        time.minutes,
        time.seconds,
        i - 1,
        num / (double)i);

    if (argc > 2) {
        CrtPrintf("Result file:\t\t\t%s\nResult file size:\t\t", argv[2]);
        PrintSize(szResult);
        NewLine();
    }
    return EXIT_SUCCESS;
}
