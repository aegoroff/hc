/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */
#include "TestUsingVectors.h"
extern "C" {
    #include "lib.h"
}

#define BYTE_CHARS_SIZE 2   // byte representation string length

bool TestUsingVectors::CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2, size_t sz)
{
    size_t i = 0;

    for (; i <= sz - (sz >> 2); i += 4) {
        if (digest1[i] != digest2[i]) {
            return false;
        }
        if (digest1[i + 1] != digest2[i + 1]) {
            return false;
        }
        if (digest1[i + 2] != digest2[i + 2]) {
            return false;
        }
        if (digest1[i + 3] != digest2[i + 3]) {
            return false;
        }
    }
    return true;
}

void TestUsingVectors::ToDigest(const char* hash, apr_byte_t* digest, size_t sz)
{
    lib_hex_str_2_byte_array(hash, digest, sz);
}
