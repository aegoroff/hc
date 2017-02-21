// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains abstract buffered test class interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-08
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#pragma once

#include "gtest.h"

class BufferedTest : public ::testing::Test {
    std::auto_ptr<char> buffer_;
    protected:
    virtual void SetUp() override;
    virtual size_t GetBufferSize() const = 0;
    public:
    char* GetBuffer() const;
};
