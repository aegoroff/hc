/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains argtable test implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-04-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "ArgtableFileTest.h"
#include "argtable3.h"

void ArgtableFileTest::SetUp() {
    a = arg_file1(nullptr, nullptr, "<file>", "filename to test");
    auto end = arg_end(20);
    n = 1;
    argtable = static_cast<void**>(malloc(n * sizeof(arg_dbl *) + sizeof(struct arg_end *)));
    argtable[0] = a;
    argtable[1] = end;
}

void ArgtableFileTest::TearDown() {
    arg_freetable(argtable, n + 1);
}
