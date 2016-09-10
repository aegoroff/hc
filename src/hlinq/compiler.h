/*!
 * \brief   The file contains HLINQ compiler API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#ifndef COMPILER_HCALC_H_
#define COMPILER_HCALC_H_

#include <apr.h>
#include <apr_pools.h>

#include "../srclib/lib.h"
#include "../srclib/traverse.h"
#include "../linq2hash/hashes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct program_options_t {
    BOOL PrintCalcTime;
    BOOL PrintLowCase;
    BOOL PrintSfv;
    BOOL PrintVerify;
    BOOL OnlyValidate;
    const char* FileToSave;
    BOOL NoProbe;
    BOOL NoErrorOnFind;
    uint32_t NumOfThreads;
} program_options_t;

typedef enum ctx_type_t {
    CtxTypeUndefined = -1,
    CtxTypeFile,
    CtxTypeString,
    CtxTypeDir,
    CtxTypeHash
} ctx_type_t;

typedef struct file_ctx_t {
    apr_finfo_t* Info;
    const char* Dir;
    void (* PfnOutput)(out_context_t* ctx);
} file_ctx_t;

typedef struct statement_ctx_t {
    const char* Id;
    const char* Source;
    ctx_type_t Type;
    hash_definition_t* HashAlgorithm;
} statement_ctx_t;

typedef struct string_statement_ctx_t {
    BOOL BruteForce;
    int Min;
    int Max;
    const char* Dictionary;
} string_statement_ctx_t;

typedef struct dir_statement_ctx_t {
    const char* hash_to_search_;
    BOOL find_files_;
    BOOL recursively_;
    apr_off_t limit_;
    apr_off_t offset_;
    const char* exclude_pattern_;
    const char* include_pattern_;
} dir_statement_ctx_t;

void cpl_init_program(program_options_t* po, const char* file_param, apr_pool_t* root);
void cpl_open_statement();
void cpl_close_statement(void);
void cpl_define_query_type(ctx_type_t type);
void cpl_register_variable(const char* var, const char* value);

void cpl_set_source(const char* str, void* token);

void cpl_set_hash_algorithm_into_context(const char* str);
void cpl_set_recursively();

dir_statement_ctx_t* cpl_get_dir_context();

#ifdef __cplusplus
}
#endif

#endif // COMPILER_HCALC_H_
