/*!
 * \brief   The file contains abstract buffered test class interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-08
            \endverbatim
 * Copyright: (c) Alexander Egorov 2010
 */

#pragma once

#include "gtest.h"
#include <stdio.h>
#include <tchar.h>

class BufferedTest : public ::testing::Test {
    private:
        std::auto_ptr<char> buffer_;
    protected:
        virtual void SetUp();
        virtual size_t GetBufferSize() const = 0;
    public:
        char* GetBuffer() const;
};
