// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains tree test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#pragma once

#include "gtest.h"
#include <tchar.h>

#ifdef __cplusplus


extern "C" {
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


        static void TearDownTestCase() {
            apr_pool_destroy(pool_);
            apr_terminate();
        }

        static void SetUpTestCase() {
            auto argc = 1;

            const char* const argv[] = {"1"};

            auto status = apr_app_initialize(&argc, (const char *const **)&argv, nullptr);

            if(status != APR_SUCCESS) {
                throw status;
            }
            apr_pool_create(&pool_, NULL);
        }

    };

#ifdef __cplusplus
}
#endif
