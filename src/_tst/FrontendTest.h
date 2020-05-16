/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains frontend test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#pragma once

#include "gtest.h"
#include <tchar.h>
#include <apr.h>
#include <apr_pools.h>
#include <apr_errno.h>
#include <frontend.h>

static apr_pool_t* pool_;

class FrontendTest : public ::testing::Test {
private:
    std::streambuf* cout_stream_buffer_;

protected:
    std::ostringstream oss_;
    const char* parameter_;

    virtual void SetUp() override;
    virtual void TearDown() override;

    static bool Compile(const char* q);

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
        fend_init(pool_);
    }
};
