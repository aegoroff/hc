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
#include <apr_hash.h>
#include <lib.h>
#include "output.h"
#include "backend.h"
#include "processor.h"
#include "hashes.h"

 /*
     proc_ - public members
     prproc_ - private members
 */

#define STACK_INIT_SZ 32

static void prproc_print_op(triple_t* triple, int i);
static const char* prproc_to_string(opcode_t code, op_value_t* value, int position);

// Processors
void prproc_on_def(triple_t* triple);
void prproc_on_string(triple_t* triple);
void prproc_on_from(triple_t* triple);
void prproc_on_property(triple_t* triple);
void prproc_on_select(triple_t* triple);

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

static void (*proc_processors[])(triple_t*) = {
    &prproc_on_from, // opcode_from
    &prproc_on_def, // opcode_def
    NULL, // opcode_let
    &prproc_on_select, // opcode_select
    NULL, // opcode_call
    &prproc_on_property, // opcode_property
    NULL, // opcode_type
    NULL, // opcode_usage
    NULL, // opcode_integer
    &prproc_on_string, // opcode_string
    NULL, // opcode_and_rel
    NULL, // opcode_or_rel
    NULL, // opcode_not_rel
    NULL, // opcode_relation
    NULL, // opcode_query_continuation
    NULL // opcode_into
};

static apr_array_header_t* proc_instructions;

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
    proc_instructions = apr_array_make(proc_pool, STACK_INIT_SZ, sizeof(source_t*));
    pcre_context = pcre2_general_context_create(&pcre_alloc, &pcre_free, NULL);
    hsh_initialize_hashes(proc_pool);
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

        void (*proc_processor)(triple_t*) = proc_processors[triple->code];

        if (proc_processor != NULL) {
            proc_processor(triple);
        }
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

void prproc_on_def(triple_t* triple)
{
    hash_definition_t* hash = NULL;
    source_t* instruction = NULL;

    switch (triple->op1->type) {
    case type_def_string:
        instruction = (source_t*)apr_pcalloc(proc_pool, sizeof(source_t));
        instruction->type = instr_type_string_decl;
        instruction->name = triple->op2->string;
        * (triple_t * *)apr_array_push(proc_instructions) = instruction;
        break;
    case type_def_file:
    case type_def_dir:
    case type_def_dynamic:
        break;
    default:
        hash = hsh_get_hash(triple->op1->string);
        break;
    }
}

void prproc_on_string(triple_t* triple)
{
    source_t* instruction = NULL;

    instruction = (source_t*)apr_pcalloc(proc_pool, sizeof(source_t));
    instruction->type = instr_type_string_def;
    instruction->value = triple->op1->string;
    *(source_t * *)apr_array_push(proc_instructions) = instruction;
}

void prproc_on_from(triple_t* triple)
{
    source_t* to = ((source_t * *)proc_instructions->elts)[triple->op1->number];
    source_t* from = ((source_t * *)proc_instructions->elts)[triple->op2->number];
    to->value = from->value;
}

void prproc_on_property(triple_t* triple)
{
    source_t* instruction = NULL;

    instruction = (source_t*)apr_pcalloc(proc_pool, sizeof(source_t));
    instruction->type = instr_type_hash_prop;
    instruction->name = triple->op1->string;
    instruction->value = triple->op2->string;

    *(source_t * *)apr_array_push(proc_instructions) = instruction;
}

void prproc_on_select(triple_t* triple)
{
    int i;
    apr_hash_t* properties = NULL;
    properties = apr_hash_make(proc_pool);

    for (i = proc_instructions->nelts - 1; i >= 0; i--) {
        source_t* instr = ((source_t **)proc_instructions->elts)[i];
        if (instr->type == instr_type_hash_prop) {
            hash_definition_t* hash = hsh_get_hash(instr->value);

            if (hash != NULL) {
                apr_hash_set(properties, instr->name, APR_HASH_KEY_STRING, hash);
            }
        }

        if (instr->type == instr_type_string_decl) {
            hash_definition_t* hash_to_calculate = (hash_definition_t*)apr_hash_get(properties, instr->name, APR_HASH_KEY_STRING);
            if (hash_to_calculate != NULL) {

                apr_byte_t* digest = apr_pcalloc(proc_pool, sizeof(apr_byte_t) * hash_to_calculate->hash_length_);
                const apr_size_t sz = hash_to_calculate->hash_length_;
                out_context_t ctx = { 0 };

                // some hashes like NTLM required unicode string so convert multi byte string to unicode one
                if (hash_to_calculate->use_wide_string_) {
                    wchar_t* str = enc_from_ansi_to_unicode(instr->value, proc_pool);
                    hash_to_calculate->pfn_digest_(digest, str, wcslen(str) * sizeof(wchar_t));
                }
                else {
                    hash_to_calculate->pfn_digest_(digest, instr->value, strlen(instr->value));
                }

                ctx.string_to_print_ = out_hash_to_string(digest, 1, sz, proc_pool);

                out_output_to_console(&ctx);
            }
        }
    }
}
