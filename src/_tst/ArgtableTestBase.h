/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
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
