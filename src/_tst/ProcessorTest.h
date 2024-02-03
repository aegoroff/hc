/*!
 * \brief   The file contains processor test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-08-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#pragma once

#include "gtest.h"
#ifdef _MSC_VER
#include <tchar.h>
#endif
#include <apr_file_info.h>
#include <apr.h>
#include <apr_pools.h>
#include <apr_errno.h>
#include <processor.h>

static apr_pool_t* pool_;

class ProcessorTest : public ::testing::Test {

protected:

    static void TearDownTestSuite() {
        proc_complete();
        apr_pool_destroy(pool_);
        apr_terminate();
    }

    static void SetUpTestSuite() {
        auto argc = 1;

        constexpr char* const argv[] = { "1" };

        auto status = apr_app_initialize(&argc, (const char *const **)&argv, nullptr);

        if(status != APR_SUCCESS) {
            throw status;
        }
        apr_pool_create(&pool_, NULL);
        proc_init(pool_);
    }
};
