/*!
 * \brief   The file contains brute force test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#pragma once

#include <gtest/gtest.h>
#ifdef _MSC_VER
#include <tchar.h>
#endif
#include <apr_pools.h>
#include "hashes.h"

static apr_pool_t* pool_;

class BruteForceTest : public ::testing::TestWithParam<const char*> {

protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
public:

    static void TearDownTestSuite() {
        apr_pool_destroy(pool_);
        apr_terminate();
    }

    static void SetUpTestSuite() {
        auto argc = 1;

        const char* const argv[] = { "1" };

        auto status = apr_app_initialize(&argc, (const char *const **)&argv, nullptr);

        if(status != APR_SUCCESS) {
            throw status;
        }
        apr_pool_create(&pool_, NULL);
        hsh_initialize_hashes(pool_);
    }
};
