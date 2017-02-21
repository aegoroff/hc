// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-08
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <memory>
#include "BufferedTest.h"

using namespace std;

void BufferedTest::SetUp() {
    buffer_ = auto_ptr<char>(new char[GetBufferSize()]);
    memset(buffer_.get(), 0, GetBufferSize());
}

char* BufferedTest::GetBuffer() const {
    return buffer_.get();
}
