﻿/*!
 * \brief   The file contains string builtin interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2025
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
    BOOL is_base64_;
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
