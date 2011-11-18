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

class HLINQTest : public ::testing::Test {
    protected:
        virtual void SetUp();
};
