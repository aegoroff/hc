/*!
 * \brief   The file contains brute force test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#pragma once

#include "gtest.h"
#include <tchar.h>
#include <apr_pools.h>

#ifdef __cplusplus


extern "C" {
#endif

static apr_pool_t* pool_;

class BruteForceTest : public ::testing::Test {
    public:
    protected:
        virtual void SetUp() override;
        virtual void TearDown() override;

    static void TearDownTestCase() {
        apr_pool_destroy(pool_);
        apr_terminate();
    }

    static void SetUpTestCase() {
        auto argc = 1;

        const char* const argv[] = { "1" };

        auto status = apr_app_initialize(&argc, (const char *const **)&argv, nullptr);

        if (status != APR_SUCCESS) {
            throw status;
        }
        apr_pool_create(&pool_, NULL);
    }
};

#ifdef __cplusplus
}
#endif