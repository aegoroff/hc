/*!
 * \brief   The file contains l2h processor interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2019-08-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#ifndef LINQ2HASH_PROC_H_
#define LINQ2HASH_PROC_H_
#include <apr_tables.h>
#include "frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef enum instr_type_t {
        instr_type_file_decl,
        instr_type_dir_decl,
        instr_type_string_decl,
        instr_type_hash_decl,
        instr_type_string_def,
        instr_type_prop_call,
        instr_type_hash_definition
    } instr_type_t;

    typedef struct source_t {
        instr_type_t type;
        char* name;
        char* value;
    } source_t;

    void proc_init(apr_pool_t* pool);
    void proc_complete();
    BOOL proc_match_re(const char* pattern, const char* subject);
    void proc_run(apr_array_header_t* instructions);
    const char* proc_get_cond_op_name(cond_op_t op);
    const char* proc_get_type_name(type_def_t type);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_PROC_H_
