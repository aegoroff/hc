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
            Creation date: 2017-04-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#pragma once
#include "ArgtableTestBase.h"

class ArgtableDateTest : public ArgtableTestBase {
public:
protected:
    struct arg_date* a;
    struct arg_date* b;
    struct arg_date* c;
    void SetUp() override;
    size_t GetOptionsCount() override;
};
