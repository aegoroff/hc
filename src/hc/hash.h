/*!
 * \brief   The file contains hash builtin interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#ifndef HLINQ_HASH_H_
#define HLINQ_HASH_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "builtin.h"

/**
 * \brief Hash builtin parameters structure
 */
typedef struct hash_builtin_ctx_t {
    builtin_ctx_t* builtin_ctx_;
    const char* hash_;
    int min_;
    int max_;
    const char* dictionary_;
    int threads_;
    BOOL performance_;
    BOOL no_probe_;
    BOOL is_base64_;
} hash_builtin_ctx_t;

/**
 * \brief Start running hash builtin
 * \param ctx hash builtin context
 */
void hash_run(hash_builtin_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // HLINQ_HASH_H_
