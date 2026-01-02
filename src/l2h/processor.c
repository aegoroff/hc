/*!
 * \brief   The file contains l2h processor implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2019-08-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#define PCRE2_CODE_UNIT_WIDTH 8

#include "intl.h"
#include <apr_file_info.h>
#include <apr_file_io.h>
#include <apr_hash.h>
#include <apr_strings.h>
#include <lib.h>
#include <pcre2.h>
#ifdef _MSC_VER
#include <basetsd.h>
#else
#include <limits.h>
#define MAXLONG64 LLONG_MAX
#endif
#include "../hc/builtin.h"
#include "../hc/dir.h"
#include "../hc/file.h"
#include "../hc/hash.h"
#include "../hc/str.h"
#include "backend.h"
#include "hashes.h"
#include "output.h"
#include "processor.h"

/*
    proc_ - public members
    prproc_ - private members
*/

#define STACK_INIT_SZ 32

static void prproc_print_op(triple_t *triple, int i);
static const char *prproc_to_string(opcode_t code, op_value_t *value, int position);
static void prproc_calculate_string(const char *hash, const char *string);
static void prproc_calculate_file(const char *hash, const char *string);
static void prproc_calculate_dir(const char *hash, const char *path);
static void prproc_calculate_hash(const char *hash, const char *digest);
static uint32_t prproc_get_threads_count();

// Processors
static void prproc_on_def(triple_t *triple);
static void prproc_on_string(triple_t *triple);
static void prproc_on_from(triple_t *triple);
static void prproc_on_property(triple_t *triple);
static void prproc_on_select(triple_t *triple);

pcre2_general_context *pcre_context = NULL;

static apr_pool_t *proc_pool = NULL;

static char *proc_opcode_names[] = {"opcode_from     ", "opcode_def      ", "opcode_let      ", "opcode_select   ",
                                    "opcode_call     ", "opcode_property ", "opcode_type     ", "opcode_usage    ",
                                    "opcode_integer  ", "opcode_string   ", "opcode_and_rel  ", "opcode_or_rel   ",
                                    "opcode_not_rel  ", "opcode_relation ", "opcode_continue ", "opcode_into     "};

static char *proc_cond_op_names[] = {"==", "!=", "~", "!~", ">", "<", ">=", "<=", "or", "and", "not"};

static const char *proc_type_names[] = {"hash", "file", "dir", "string"};

static const char *hash_value_to_restore = "digest";

static void (*proc_processors[])(triple_t *) = {
    &prproc_on_from,     // opcode_from
    &prproc_on_def,      // opcode_def
    NULL,                // opcode_let
    &prproc_on_select,   // opcode_select
    NULL,                // opcode_call
    &prproc_on_property, // opcode_property
    NULL,                // opcode_type
    NULL,                // opcode_usage
    NULL,                // opcode_integer
    &prproc_on_string,   // opcode_string
    NULL,                // opcode_and_rel
    NULL,                // opcode_or_rel
    NULL,                // opcode_not_rel
    NULL,                // opcode_relation
    NULL,                // opcode_query_continuation
    NULL                 // opcode_into
};

static apr_array_header_t *proc_instructions;

/**
 * \brief PCRE required function. Allocates memory from apache pool
 * \param size the number of bytes to allocate
 * \param memory_data unused
 * \return allocated memory
 */
void *pcre_alloc(size_t size, void *memory_data) { return apr_palloc(proc_pool, size); }

/**
 * \brief Frees memory allocated. Required by PCRE engine. Does nothing because memory released by destroying apache
 * pool \param p1 unused \param p2 unused
 */
void pcre_free(void *p1, void *p2) {}

void proc_init(apr_pool_t *pool) {
    apr_pool_create(&proc_pool, pool);
    proc_instructions = apr_array_make(proc_pool, STACK_INIT_SZ, sizeof(source_t *));
    pcre_context = pcre2_general_context_create(&pcre_alloc, &pcre_free, NULL);
    hsh_initialize_hashes(proc_pool);
}

void proc_complete(void) {
    pcre2_general_context_free(pcre_context);
    apr_pool_destroy(proc_pool);
}

BOOL proc_match_re(const char *pattern, const char *subject) {
    int errornumber = 0;
    size_t erroroffset = 0;

    pcre2_compile_context *compile_ctx = pcre2_compile_context_create(pcre_context);

    pcre2_code *re = pcre2_compile((unsigned char *)pattern, /* the pattern */
                                   PCRE2_ZERO_TERMINATED,    /* indicates pattern is zero-terminated */
                                   0,                        /* default options */
                                   &errornumber,             /* for error number */
                                   &erroroffset,             /* for error offset */
                                   compile_ctx);             /* use default compile context */

    if (re == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errornumber, buffer, sizeof(buffer));
        lib_printf("PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, buffer);
        return FALSE;
    }
    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(re, NULL);

    int flags = PCRE2_NOTEMPTY;
    if (!strchr(subject, '^')) {
        flags |= PCRE2_NOTBOL;
    }
    if (!strchr(subject, '$')) {
        flags |= PCRE2_NOTEOL;
    }

    pcre2_match_context *match_ctx = pcre2_match_context_create(pcre_context);
    int rc = pcre2_match(re,                       /* the compiled pattern */
                         (unsigned char *)subject, /* the subject string */
                         strlen(subject),          /* the length of the subject */
                         0,                        /* start at offset 0 in the subject */
                         flags,                    /* flags */
                         match_data,               /* block for storing the result */
                         match_ctx);               /* use default match context */
    return rc >= 0;
}

void proc_run(apr_array_header_t *instructions) {
    int i;
    for (i = 0; i < instructions->nelts; i++) {
        triple_t *triple = ((triple_t **)instructions->elts)[i];
#ifdef DEBUG
        prproc_print_op(triple, i);
#endif

        void (*proc_processor)(triple_t *) = proc_processors[triple->code];

        if (proc_processor != NULL) {
            proc_processor(triple);
        }
    }
}

const char *proc_get_cond_op_name(cond_op_t op) { return proc_cond_op_names[op]; }

const char *proc_get_type_name(type_def_t type) {
    size_t n = sizeof(proc_type_names) / sizeof(proc_type_names[0]);
    if (type >= n) {
        return NULL;
    }

    return proc_type_names[type];
}

void prproc_print_op(triple_t *triple, int i) {
    char *type;
    if (triple->op2 != NULL) {
        type = apr_psprintf(proc_pool, "%2d: %s %s, %s", i, proc_opcode_names[triple->code],
                            prproc_to_string(triple->code, triple->op1, 0),
                            prproc_to_string(triple->code, triple->op2, 1));
    } else {
        type = apr_psprintf(proc_pool, "%2d: %s %s", i, proc_opcode_names[triple->code],
                            prproc_to_string(triple->code, triple->op1, 0));
    }
    lib_printf("%s\n", type);
}

const char *prproc_to_string(opcode_t code, op_value_t *value, int position) {
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
        } else if (value != NULL) {
            // SELECT INTO case handling
            return apr_psprintf(proc_pool, "%d", value->number);
        }
        return "";
    case opcode_relation:
        return proc_cond_op_names[value->relation_op];
    case opcode_def:
        if (position) {
            return value->string;
        }
        const char *name = proc_get_type_name(value->type);
        if (name == NULL) {
            return value->string;
        }
        return name;
    default:
        return "";
    }
}

void prproc_on_def(triple_t *triple) {
    source_t *instruction = NULL;

    switch (triple->op1->type) {
    case type_def_string:
        instruction = (source_t *)apr_pcalloc(proc_pool, sizeof(source_t));
        instruction->type = instr_type_string_decl;
        instruction->name = triple->op2->string;
        *(source_t **)apr_array_push(proc_instructions) = instruction;
        break;
    case type_def_file:
        instruction = (source_t *)apr_pcalloc(proc_pool, sizeof(source_t));
        instruction->type = instr_type_file_decl;
        instruction->name = triple->op2->string;
        *(source_t **)apr_array_push(proc_instructions) = instruction;
        break;
    case type_def_dir:
        instruction = (source_t *)apr_pcalloc(proc_pool, sizeof(source_t));
        instruction->type = instr_type_dir_decl;
        instruction->name = triple->op2->string;
        *(source_t **)apr_array_push(proc_instructions) = instruction;
        break;
    case type_def_custom:
        instruction = (source_t *)apr_pcalloc(proc_pool, sizeof(source_t));
        instruction->type = instr_type_hash_decl;
        instruction->name = triple->op2->string;
        *(source_t **)apr_array_push(proc_instructions) = instruction;
        break;
    default:
        break;
    }
}

void prproc_on_string(triple_t *triple) {
    source_t *instruction = NULL;

    instruction = (source_t *)apr_pcalloc(proc_pool, sizeof(source_t));
    instruction->type = instr_type_string_def;
    instruction->value = triple->op1->string;
    *(source_t **)apr_array_push(proc_instructions) = instruction;
}

void prproc_on_from(triple_t *triple) {
    source_t *to = ((source_t **)proc_instructions->elts)[triple->op1->number];
    source_t *from = ((source_t **)proc_instructions->elts)[triple->op2->number];

    if (from != NULL) {
        if (to->type != instr_type_hash_decl) {
            to->value = from->value;
            // remove definition from stack so as not to duplicate
            *(source_t **)apr_array_pop(proc_instructions);
        } else {
            // remove definition from stack so as not to duplicate
            *(source_t **)apr_array_pop(proc_instructions);

            source_t *instruction = NULL;
            instruction = (source_t *)apr_pcalloc(proc_pool, sizeof(source_t));
            instruction->type = instr_type_hash_definition;
            instruction->name = to->value;
            instruction->value = from->value;
            *(source_t **)apr_array_push(proc_instructions) = instruction;
        }
    }
}

void prproc_on_property(triple_t *triple) {
    source_t *instruction = NULL;

    instruction = (source_t *)apr_pcalloc(proc_pool, sizeof(source_t));
    instruction->type = instr_type_prop_call;
    instruction->name = triple->op1->string;
    instruction->value = triple->op2->string;

    *(source_t **)apr_array_push(proc_instructions) = instruction;
}

void prproc_on_select(triple_t *triple) {
    int i;
    apr_hash_t *properties = NULL;
    properties = apr_hash_make(proc_pool);

    for (i = proc_instructions->nelts - 1; i >= 0; i--) {
        source_t *instr = ((source_t **)proc_instructions->elts)[i];
        if (instr->type == instr_type_prop_call) {
            if (hsh_get_hash(instr->value) != NULL) {
                apr_hash_set(properties, instr->name, APR_HASH_KEY_STRING, instr->value);
            }
        }

        if (instr->type == instr_type_hash_definition) {
            apr_hash_set(properties, hash_value_to_restore, APR_HASH_KEY_STRING, instr->value);
        }

        if (instr->type == instr_type_string_def) {
            // Dynamic type case
            apr_hash_index_t *hi = NULL;
            const char *k;
            char *hash_to_calculate;

            hi = apr_hash_first(NULL, properties);

            apr_hash_this(hi, (const void **)&k, NULL, (void **)&hash_to_calculate);

            if (hash_to_calculate != NULL) {
                apr_finfo_t finfo;

                // Only string, file or dir
                const apr_status_t rv = apr_stat(&finfo, instr->value, APR_FINFO_NORM, proc_pool);

                if (rv == APR_SUCCESS) {
                    if (finfo.filetype == APR_DIR) {
                        // Dir case
                        prproc_calculate_dir(hash_to_calculate, instr->value);
                    }
                    if (finfo.filetype == APR_REG) {
                        // file case
                        prproc_calculate_file(hash_to_calculate, instr->value);
                    }
                } else {
                    // String case
                    prproc_calculate_string(hash_to_calculate, instr->value);
                }
            }
        }

        if (instr->type == instr_type_string_decl) {
            char *hash_to_calculate = (char *)apr_hash_get(properties, instr->name, APR_HASH_KEY_STRING);
            if (hash_to_calculate != NULL) {
                prproc_calculate_string(hash_to_calculate, instr->value);
            }
        }

        if (instr->type == instr_type_hash_decl) {
            char *hash_to_calculate = (char *)apr_hash_get(properties, instr->name, APR_HASH_KEY_STRING);
            char *digest = (char *)apr_hash_get(properties, hash_value_to_restore, APR_HASH_KEY_STRING);
            if (hash_to_calculate != NULL && digest != NULL) {
                prproc_calculate_hash(hash_to_calculate, digest);
            }
        }

        if (instr->type == instr_type_file_decl) {
            char *hash_to_calculate = (char *)apr_hash_get(properties, instr->name, APR_HASH_KEY_STRING);
            if (hash_to_calculate != NULL) {
                prproc_calculate_file(hash_to_calculate, instr->value);
            }
        }

        if (instr->type == instr_type_dir_decl) {
            char *hash_to_calculate = (char *)apr_hash_get(properties, instr->name, APR_HASH_KEY_STRING);
            if (hash_to_calculate != NULL) {
                prproc_calculate_dir(hash_to_calculate, instr->value);
            }
        }
    }
}

void prproc_calculate_string(const char *hash, const char *string) {
    builtin_ctx_t *builtin_ctx = apr_pcalloc(proc_pool, sizeof(builtin_ctx_t));
    string_builtin_ctx_t *str_ctx = apr_pcalloc(proc_pool, sizeof(string_builtin_ctx_t));

    builtin_ctx->is_print_low_case_ = 1;
    builtin_ctx->hash_algorithm_ = hash;
    builtin_ctx->pfn_output_ = out_output_to_console;

    str_ctx->builtin_ctx_ = builtin_ctx;
    str_ctx->string_ = string;

    builtin_run(builtin_ctx, str_ctx, (void (*)(void *))str_run, proc_pool);
}

void prproc_calculate_hash(const char *hash, const char *digest) {
    builtin_ctx_t *builtin_ctx = apr_pcalloc(proc_pool, sizeof(builtin_ctx_t));
    hash_builtin_ctx_t *hash_ctx = apr_pcalloc(proc_pool, sizeof(hash_builtin_ctx_t));

    builtin_ctx->is_print_low_case_ = 1;
    builtin_ctx->hash_algorithm_ = hash;
    builtin_ctx->pfn_output_ = out_output_to_console;

    hash_ctx->builtin_ctx_ = builtin_ctx;
    hash_ctx->hash_ = digest;
    hash_ctx->is_base64_ = FALSE;
    hash_ctx->no_probe_ = FALSE;
    hash_ctx->performance_ = FALSE;
    hash_ctx->threads_ = prproc_get_threads_count();

    builtin_run(builtin_ctx, hash_ctx, (void (*)(void *))hash_run, proc_pool);
}

uint32_t prproc_get_threads_count() {
    uint32_t processors = lib_get_processor_count();
    uint32_t num_of_threads = processors == 1 ? 1 : MIN(processors, processors / 2);

    if (num_of_threads < 1 || num_of_threads > processors) {
        const uint32_t def = processors == 1 ? processors : processors / 2;
        lib_printf(_("Threads number must be between 1 and %u but it was set to %lu. Reset to default %u"), processors,
                   num_of_threads, def);
        lib_new_line();
        num_of_threads = def;
    }
    return num_of_threads;
}

void prproc_calculate_file(const char *hash, const char *path) {
    builtin_ctx_t *builtin_ctx = apr_pcalloc(proc_pool, sizeof(builtin_ctx_t));
    file_builtin_ctx_t *file_ctx = apr_palloc(proc_pool, sizeof(file_builtin_ctx_t));

    file_ctx->builtin_ctx_ = builtin_ctx;
    file_ctx->file_path_ = path;
    file_ctx->limit_ = MAXLONG64;
    file_ctx->offset_ = 0;
    file_ctx->hash_ = NULL;
    file_ctx->result_in_sfv_ = FALSE;
    file_ctx->is_base64_ = FALSE;
    file_ctx->is_verify_ = FALSE;
    file_ctx->save_result_path_ = NULL;

    builtin_ctx->is_print_low_case_ = 1;
    builtin_ctx->hash_algorithm_ = hash;
    builtin_ctx->pfn_output_ = out_output_to_console;

    builtin_run(builtin_ctx, file_ctx, (void (*)(void *))file_run, proc_pool);
}

void prproc_calculate_dir(const char *hash, const char *path) {
    builtin_ctx_t *builtin_ctx = apr_pcalloc(proc_pool, sizeof(builtin_ctx_t));
    dir_builtin_ctx_t *dir_ctx = apr_palloc(proc_pool, sizeof(dir_builtin_ctx_t));

    dir_ctx->builtin_ctx_ = builtin_ctx;
    dir_ctx->dir_path_ = path;
    dir_ctx->limit_ = MAXLONG64;
    dir_ctx->offset_ = 0;
    dir_ctx->show_time_ = FALSE;
    dir_ctx->is_verify_ = FALSE;
    dir_ctx->result_in_sfv_ = FALSE;
    dir_ctx->no_error_on_find_ = FALSE;
    dir_ctx->recursively_ = FALSE;
    dir_ctx->include_pattern_ = NULL;
    dir_ctx->exclude_pattern_ = NULL;
    dir_ctx->hash_ = NULL;
    dir_ctx->search_hash_ = NULL;
    dir_ctx->save_result_path_ = NULL;
    dir_ctx->is_base64_ = FALSE;

    builtin_ctx->is_print_low_case_ = 1;
    builtin_ctx->hash_algorithm_ = hash;
    builtin_ctx->pfn_output_ = out_output_to_console;

    builtin_run(builtin_ctx, dir_ctx, (void (*)(void *))dir_run, proc_pool);
}
