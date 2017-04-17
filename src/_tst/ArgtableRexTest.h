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
            Creation date: 2017-04-17
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#pragma once
#include "ArgtableTestBase.h"

class ArgtableRexTest : public ArgtableTestBase {
public:
protected:
    struct arg_rex* a;
    struct arg_rex* b;
    struct arg_rex* c;
    struct arg_rex* d;
    void SetUp() override;
    size_t GetOptionsCount() override;
};
