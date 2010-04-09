// bf.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static char *alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

int MakeAttempt(int pos, int length, const char *pDict, int *indexes, char *pass, const char *desired,
                unsigned long long *attempts, int maxIndex)
{
    int i = 0;
    int j = 0;

    for (; i <= maxIndex; ++i) {
        indexes[pos] = i;

        if (pos == length - 1) {
            for (j = 0; j < length; ++j) {
                pass[j] = pDict[indexes[j]];
            }
            ++*attempts;
            /* you may replace this function with the encryption function
               and return in same manner if the encryption string matches
               the one we are cracking */
            if (strcmp(desired, pass) == 0) {
                return 1;
            }
        } else {
            if (MakeAttempt(pos + 1, length, pDict, indexes, pass, desired, attempts, maxIndex)) {
                return 1;
            }
        }
    }
    return 0;
}

/*!
* Caller must free memory allocated for the result in case of any
*/
char *BruteForce(int passmin, int passmax, const char *pDict, const char *desired, unsigned long long *attempts)
{
    char *pass = (char *)malloc(passmax + 1);
    int *indexes = (int *)malloc(passmax * sizeof(int));
    int passLength = passmin;
    int maxIndex = strlen(pDict) - 1;

    memset(pass, 0, passmax + 1);   // IMPORTANT

    for (; passLength <= passmax; ++passLength) {
        if (MakeAttempt(0, passLength, pDict, indexes, pass, desired, attempts, maxIndex)) {
            goto cleanup;
        }
    }
    // Nothing found
    free(pass);
    pass = NULL;
cleanup:
    free(indexes);
    return pass;
}

int main(int argc, char *argv[])
{
    char *pass = NULL;
    char *desired = NULL;   // it must be hash but it's plain string to demonstrate concept
    clock_t start = 0;
    clock_t end = 0;
    double elapsed = 0.0;
    unsigned long long count = 0;

    if (argc != 2) {
        printf("\nProof of concept brute force\nby Alexander Egorov\n");
        printf("\nUsage: %s <pass to crack>\n\n", argv[0]);
        return 0;
    }

    start = clock();
    desired = argv[1];
    printf("\nAttempting to brute force \"%s\"\n", desired);
    printf("Should take approximatelly around %.0f trys\n\n", pow((double)strlen(alphabet), (double)strlen(desired)));

    if (pass = BruteForce(1, strlen(desired), alphabet, desired, &count)) {
        printf("The correct password is \"%s\"", pass);
        free(pass);
    } else {
        printf("Hard luck\n");
    }

    printf("\nNo of trys was %llu\n", count);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time elapsed = %.3f seconds\n", elapsed);

    if (elapsed >= 1) {
        printf("Trys per second was %.3f\n", count / elapsed);
    }
    printf("\n");
    return 0;
}

