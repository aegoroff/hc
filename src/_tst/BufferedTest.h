/*!
 * \brief   The file contains abstract buffered test class interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-08
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#pragma once

#include <memory>
#include "gtest.h"

class BufferedTest : public ::testing::Test {
    std::unique_ptr<char[]> buffer_;
protected:
    virtual void SetUp() override;
    virtual size_t GetBufferSize() const = 0;
public:
    char* GetBuffer() const;
};
