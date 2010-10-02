/*!
 * \brief   The file contains test of SizeToString function interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2010
 */

#pragma once

#include "gtest.h"
#include <stdio.h>
#include <tchar.h>
#include <windows.h>

const size_t kBufferSize = 128;

class TSizeToString : public ::testing::Test {
    private:
        std::auto_ptr<char> buffer_;
    protected:
        virtual void SetUp();
    public:
        char* GetBuffer() const;
};
