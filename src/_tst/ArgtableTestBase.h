/*!
 * \brief   The file contains argtable test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-04-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#pragma once
#include "gtest.h"

class ArgtableTestBase : public testing::Test {
public:
protected:
    void** argtable;
    virtual size_t GetOptionsCount() = 0;
    void TearDown() override;
};
