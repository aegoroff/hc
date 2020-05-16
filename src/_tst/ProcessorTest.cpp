/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains processor test class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#include "ProcessorTest.h"
#include "processor.h"

TEST_F(ProcessorTest, MatchSuccess) {
    ASSERT_TRUE(proc_match_re("[0-9]+", "123"));
}

TEST_F(ProcessorTest, MatchFailure) {
    ASSERT_FALSE(proc_match_re("[0-9]+", "num"));
}
