/*!
 * \brief   The file contains encoding test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2020-04-12
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#pragma once

#include "gtest.h"
#ifdef _MSC_VER
#include <tchar.h>
#endif
#include <apr_pools.h>

static apr_pool_t* pool_;

class EncodingTest : public ::testing::Test {
public:
protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
    apr_pool_t* testPool_;

    const char* utf8_ = "\xd1\x82\xd0\xb5\xd1\x81\xd1\x82"; // тест
    const char* ansi_ = "\xf2\xe5\xf1\xf2";                 // тест
    const wchar_t* unicode_ = L"\x0442\x0435\x0441\x0442";  // тест

    static void TearDownTestSuite() {
        apr_pool_destroy(pool_);
        apr_terminate();
    }

    static void SetUpTestSuite() {
        auto argc = 1;

        const char* const argv[] = { "1" };

        auto status = apr_app_initialize(&argc, (const char* const**)&argv, nullptr);

        if(status != APR_SUCCESS) {
            throw status;
        }
        apr_pool_create(&pool_, NULL);
    }
};
