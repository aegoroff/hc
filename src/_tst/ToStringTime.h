/*!
 * \brief   The file contains test of ToStringTime function interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#pragma once

#include <stdio.h>
#include <tchar.h>
#include <windows.h>
#include "BufferedTest.h"

class ToStringTime : public BufferedTest {
    protected:
        size_t GetBufferSize() const;
};
