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
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#pragma once
#include "gtest.h"

class ArgtableFileTest : public testing::Test {
public:
    void** argtable;
    struct arg_file* a;
    size_t n;
    void SetUp() override;
    void TearDown() override;
};
