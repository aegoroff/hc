/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains brute force test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#pragma once

#include "gtest.h"
#include <tchar.h>
#include <apr_pools.h>
#include "hashes.h"

static apr_pool_t* pool_;

class BruteForceTest : public ::testing::TestWithParam<const char*> {

protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
public:

    static void TearDownTestCase() {
        apr_pool_destroy(pool_);
        apr_terminate();
    }

    static void SetUpTestCase() {
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
