/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains l2h processor implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2019-08-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#define PCRE2_CODE_UNIT_WIDTH 8

#include <pcre2.h>
#include <apr_strings.h>
#include <lib.h>
#include "backend.h"
#include "processor.h"

 /*
     proc_ - public members
     prproc_ - private members
 */

static void prproc_print_op(triple_t* triple, int i);
static const char* prproc_to_string(opcode_t code, op_value_t* value, int position);

pcre2_general_context* pcre_context = NULL;

static apr_pool_t* proc_pool = NULL;

static char* proc_opcode_names[] = {
    "opcode_from     ",
    "opcode_def      ",
    "opcode_let      ",
    "opcode_select   ",
    "opcode_call     ",
    "opcode_property ",
    "opcode_type     ",
    "opcode_usage    ",
    "opcode_integer  ",
    "opcode_string   ",
    "opcode_and_rel  ",
    "opcode_or_rel   ",
    "opcode_not_rel  ",
    "opcode_relation ",
    "opcode_continue ",
    "opcode_into     "
};

static char* proc_cond_op_names[] = {
    "==",
    "!=",
    "~",
    "!~",
    ">",
    "<",
    ">=",
    "<=",
    "or",
    "and",
    "not"
};

static const char* proc_type_names[] = {
    "dynamic",
    "file",
    "dir",
    "string",
    "user"
};

/**
 * \brief PCRE requred function. Allocates memory from apache pool
 * \param size the number of bytes to allocate
 * \param memory_data
 * \return
 */
void* pcre_alloc(size_t size, void* memory_data) {
    return apr_palloc(proc_pool, size);
}

/**
 * \brief Frees memory allocated. Requied by PCRE engine. Does nothing because memory released by destroying apache pool
 * \param p1
 * \param p2
 */
void pcre_free(void* p1, void* p2) {

}

void proc_init(apr_pool_t* pool) {
    apr_pool_create(&proc_pool, pool);
    pcre_context = pcre2_general_context_create(&pcre_alloc, &pcre_free, NULL);
}

void proc_complete() {
    pcre2_general_context_free(pcre_context);
    apr_pool_destroy(proc_pool);
}

BOOL proc_match_re(const char* pattern, const char* subject) {
    int errornumber = 0;
    size_t erroroffset = 0;

    pcre2_code* re = pcre2_compile(
        (unsigned char*)pattern, /* the pattern */
        PCRE2_ZERO_TERMINATED,   /* indicates pattern is zero-terminated */
        0,                       /* default options */
        &errornumber,            /* for error number */
        &erroroffset,            /* for error offset */
        NULL);                   /* use default compile context */

    if (re == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errornumber, buffer, sizeof(buffer));
        lib_printf("PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, buffer);
        return FALSE;
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, NULL);

    int flags = PCRE2_NOTEMPTY;
    if (!strchr(subject, '^')) {
        flags |= PCRE2_NOTBOL;
    }
    if (!strchr(subject, '$')) {
        flags |= PCRE2_NOTEOL;
    }

    int rc = pcre2_match(
        re,                      /* the compiled pattern */
        (unsigned char*)subject, /* the subject string */
        strlen(subject),         /* the length of the subject */
        0,                       /* start at offset 0 in the subject */
        flags,
        match_data, /* block for storing the result */
        NULL);      /* use default match context */
    return rc >= 0;
}

void proc_run(apr_array_header_t* instructions)
{
    int i;
    for (i = 0; i < instructions->nelts; i++) {
        triple_t* triple = ((triple_t * *)instructions->elts)[i];
        prproc_print_op(triple, i);
    }
}

const char* proc_get_cond_op_name(cond_op_t op)
{
    return proc_cond_op_names[op];
}

const char* proc_get_type_name(type_def_t type)
{
    return proc_type_names[type];
}

void prproc_print_op(triple_t* triple, int i) {
    char* type;
    if (triple->op2 != NULL) {
        type = apr_psprintf(proc_pool, "%2d: %s %s, %s", i, proc_opcode_names[triple->code],
            prproc_to_string(triple->code, triple->op1, 0),
            prproc_to_string(triple->code, triple->op2, 1));
    }
    else {
        type = apr_psprintf(proc_pool, "%2d: %s %s", i, proc_opcode_names[triple->code],
            prproc_to_string(triple->code, triple->op1, 0));
    }
    lib_printf("%s\n", type);
}

const char* prproc_to_string(opcode_t code, op_value_t* value, int position) {
    switch (code) {
    case opcode_integer:
    case opcode_from:
        return apr_psprintf(proc_pool, "%d", value->number);
    case opcode_string:
    case opcode_property:
    case opcode_call:
        return value->string;
    case opcode_usage:
        if (position) {
            return value->string;
        }
        return "";
    case opcode_into:
        if (position) {
            return value->string;
        }
        else if (value != NULL) { // SELECT INTO case handling
            return apr_psprintf(proc_pool, "%d", value->number);
        }
        return "";
    case opcode_relation:
        return proc_cond_op_names[value->relation_op];
    case opcode_def:
        // 0
        if (value->type >= type_def_dynamic && value->type <= type_def_user) {
            return proc_type_names[value->type];
        }
        return value->string;
    default:
        return "";
    }
}
