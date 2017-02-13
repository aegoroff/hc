// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains file builtin interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-11
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#ifndef HLINQ_FILE_H_
#define HLINQ_FILE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "builtin.h"

/**
 * \brief File builtin parameters structure
 */
typedef struct file_builtin_ctx_t {
    builtin_ctx_t* builtin_ctx_;
    const char* file_path_;
    apr_off_t limit_;
    apr_off_t offset_;
    const char* hash_;
    BOOL show_time_;
    const char* save_result_path_;
    BOOL result_in_sfv_;
    BOOL is_verify_;
} file_builtin_ctx_t;


/**
 * \brief Start running file builtin
 * \param ctx file builtin context
 */
void file_run(file_builtin_ctx_t* ctx);


#ifdef __cplusplus
}
#endif

#endif // HLINQ_FILE_H_

