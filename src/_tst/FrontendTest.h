/*!
 * \brief   The file contains HLINQ test class interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2015
 */

#pragma once

#include "gtest.h"
#include <tchar.h>

#ifdef __cplusplus

extern "C" {
#endif
    #include "apr.h"
    #include "apr_pools.h"
    #include <frontend.h>

    static apr_pool_t* pool_;

class FrontendTest : public ::testing::Test {
    private:
        std::streambuf* cout_stream_buffer_;

    protected:
        std::ostringstream oss_;
        const char* parameter_;

        virtual void SetUp();
        virtual void TearDown();
        
        void Run(const char* q, BOOL dontRunActions = TRUE);
        void ValidateNoError();
        void ValidateError();

        static void TearDownTestCase()
        {
            apr_pool_destroy(pool_);
            apr_terminate();
        }

        static void SetUpTestCase()
        {
            int argc = 1;

            const char* const argv[] = { "1" };

            apr_status_t status = apr_app_initialize(&argc, (const char *const **)&argv, NULL);

            if (status != APR_SUCCESS) {
                throw status;
            }
            apr_pool_create(&pool_, NULL);
            FrontendInit(pool_);
        }
};

#ifdef __cplusplus
}
#endif