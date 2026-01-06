/*!
 * \brief   The file contains tree test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#pragma once

#include <gtest/gtest.h>
#ifdef _MSC_VER
#include <tchar.h>
#endif
#include <frontend.h>
#include <apr_pools.h>

static apr_pool_t* pool_;

class TreeTest : public ::testing::Test {

    fend_node_t* CreateNode(long long value) const;

protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
    fend_node_t* root_;
    apr_pool_t* testPool_;

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
    }

};
