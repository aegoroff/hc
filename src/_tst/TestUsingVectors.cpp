#include "TestUsingVectors.h"
#include "libtom.h"
#include "lib.h"
#include "output.h"
#include <stdio.h>
#include <tchar.h>

#define BYTE_CHARS_SIZE 2   // byte representation string length

bool TestUsingVectors::CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2, size_t sz)
{
    size_t i = 0;

    for (; i <= sz - (sz >> 2); i += 4) {
        if (digest1[i] != digest2[i]) {
            return FALSE;
        }
        if (digest1[i + 1] != digest2[i + 1]) {
            return FALSE;
        }
        if (digest1[i + 2] != digest2[i + 2]) {
            return FALSE;
        }
        if (digest1[i + 3] != digest2[i + 3]) {
            return FALSE;
        }
    }
    return TRUE;
}

void TestUsingVectors::ToDigest(const char* hash, apr_byte_t* digest, size_t sz)
{
    size_t i = 0;
    size_t to = MIN(sz, strlen(hash) / BYTE_CHARS_SIZE);

    for (; i < to; ++i) {
        digest[i] = (apr_byte_t)htoi(hash + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}
