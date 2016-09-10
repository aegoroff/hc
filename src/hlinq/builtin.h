/*!
 * \brief   The file contains commit builtins interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#ifndef HLINQ_BUILTIN_H_
#define HLINQ_BUILTIN_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "output.h"
#include "../linq2hash/hashes.h"

/**
 * \brief Builtin common parameters structure. These parameters applied to all builtins
 */
typedef struct builtin_ctx_t {
    int is_print_low_case_;
    const char* hash_algorithm_;
    void (*pfn_output_)(out_context_t* ctx);
} builtin_ctx_t;

/**
 * \brief Initializes new builtin. Allocates memory, hash algorithm table and validates hash existence
 * \param ctx builtin common context
 * \param root root memory pool
 * \return TRUE IF initialization successful and hash supported. FALSE otherwise.
 */
BOOL builtin_init(builtin_ctx_t* ctx, apr_pool_t* root);

/**
 * \brief frees all builtin resources
 */
void builtin_close();

/**
 * \brief gets builtin memory pool
 * \return memory pool
 */
apr_pool_t* builtin_get_pool(void);


/**
 * \brief Gets builtin hash definition
 * \return Hash definition structure
 */
hash_definition_t* builtin_get_hash_definition(void);

#ifdef __cplusplus
}
#endif

#endif // HLINQ_BUILTIN_H_
