/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains commit builtins interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#ifndef HLINQ_BUILTIN_H_
#define HLINQ_BUILTIN_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "output.h"
#include "../l2h/hashes.h"

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
void builtin_close(void);

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

/**
 * \brief 
 * \param ctx Runs builtin
 * \param concrete_ctx specific builtin context
 * \param pfn_action specific builtin action
 * \param root memory pool
 */
void builtin_run(builtin_ctx_t* ctx, void* concrete_ctx, void (*pfn_action)(void* concrete_builtin_ctx), apr_pool_t* root);

/**
 * \brief Creates binary hash from string passed. Algorithm taken from context i.e. builtin_init must be called before using this function 
 * \param string input string
 * \return binary hash
 */
apr_byte_t* builtin_hash_from_string(const char* string);

/**
 * \brief Output both into file and console function implementation
 * \param file result file
 * \param ctx output context
 */
void builtin_output_both_file_and_console(FILE* file, out_context_t* ctx);

/**
* \brief Checks whether --sfv can be used
* \param result_in_sfv --sfv option value
*/
BOOL builtin_allow_sfv_option(BOOL result_in_sfv);

#ifdef __cplusplus
}
#endif

#endif // HLINQ_BUILTIN_H_
