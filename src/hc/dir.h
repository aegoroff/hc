/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains directory builtin interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-11
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#ifndef HLINQ_DIR_H_
#define HLINQ_DIR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "builtin.h"

/**
 * \brief Directory builtin parameters structure
 */
typedef struct dir_builtin_ctx_t {
    builtin_ctx_t* builtin_ctx_;
    const char* dir_path_;
    apr_off_t limit_;
    apr_off_t offset_;
    const char* hash_;
    BOOL show_time_;
    const char* save_result_path_;
    BOOL result_in_sfv_;
    BOOL is_verify_;
    const char* include_pattern_;
    const char* exclude_pattern_;
    BOOL recursively_;
    BOOL no_error_on_find_;
    const char* search_hash_;
    BOOL is_base64_;
} dir_builtin_ctx_t;

/**
 * \brief Start running directory builtin
 * \param ctx directory builtin context
 */
void dir_run(dir_builtin_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // HLINQ_DIR_H_
