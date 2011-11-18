/*!
 * \brief   The file contains HLINQ test class interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#pragma once

#include "gtest.h"
#include <stdio.h>
#include <tchar.h>
#include "apr.h"
#include "apr_pools.h"
#include "encoding.h"
#include "HLINQLexer.h"
#include "HLINQParser.h"

#ifdef __cplusplus
extern "C" {
#endif

class HLINQTest : public ::testing::Test {
    private:


    protected:
        apr_pool_t* pool_;
        pANTLR3_INPUT_STREAM input_;

        pHLINQLexer lxr_;
        pANTLR3_COMMON_TOKEN_STREAM tstream_;
        pHLINQParser psr_;
        pANTLR3_COMMON_TREE_NODE_STREAM nodes_;

        virtual void SetUp();
        virtual void TearDown();
        
        void Run(const char* q);
        void ValidateNoError();

        static void TearDownTestCase()
        {
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
        }
};

#ifdef __cplusplus
}
#endif