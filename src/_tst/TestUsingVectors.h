// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains test of TestUsingVectors class interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#pragma once


#define SZ_WHIRLPOOL    64
#define SZ_SHA512       64
#define SZ_SHA384       48
#define SZ_RIPEMD320    40
#define SZ_SHA256       32
#define SZ_RIPEMD256    32
#define SZ_SHA224       28
#define SZ_TIGER192     24
#define SZ_SHA1         20
#define SZ_RIPEMD160    20
#define SZ_RIPEMD128    16
#define SZ_MD5          16
#define SZ_MD4          16
#define SZ_MD2          16

#include "gtest.h"

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
    #include "apr.h"
    #include "apr_pools.h"

    static apr_pool_t* pool_;

    class TestUsingVectors : public ::testing::Test {
        public:
        static bool CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2, size_t sz);

        protected:
        static void ToDigest(const char* hash, apr_byte_t* digest, size_t sz);

        static void TearDownTestCase() {
            apr_pool_destroy(pool_);
            apr_terminate();
        }

        static void SetUpTestCase() {
            auto argc = 1;

            const char* const argv[] = {"1"};

            auto status = apr_app_initialize(&argc, (const char *const **)&argv, NULL);

            if(status != APR_SUCCESS) {
                throw status;
            }
            apr_pool_create(&pool_, NULL);
        }
    };


#ifdef __cplusplus
}
#endif
