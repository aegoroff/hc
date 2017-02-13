// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains string builtin interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#ifndef HLINQ_STR_H_
#define HLINQ_STR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "builtin.h"

/**
 * \brief String builtin parameters structure
 */
typedef struct string_builtin_ctx_t {
    builtin_ctx_t* builtin_ctx_;
    const char* string_;
} string_builtin_ctx_t;


/**
 * \brief Start running string builtin
 * \param ctx string builtin context
 */
void str_run(string_builtin_ctx_t* ctx);


#ifdef __cplusplus
}
#endif

#endif // HLINQ_STR_H_
