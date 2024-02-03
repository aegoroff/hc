/*!
 * \brief   The file contains common builtin implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#include "builtin.h"
#include "hashes.h"
#include "encoding.h"
#include "intl.h"

static apr_pool_t* builtin_pool = NULL;
static hash_definition_t* builtin_hash = NULL;

BOOL builtin_init(builtin_ctx_t* ctx, apr_pool_t* root) {
    apr_pool_create(&builtin_pool, root);
    hsh_initialize_hashes(builtin_pool);
    builtin_hash = hsh_get_hash(ctx->hash_algorithm_);
    if(builtin_hash == NULL) {
        lib_printf(_("Unknown hash: %s"), ctx->hash_algorithm_);
        lib_new_line();
        builtin_close();
        return FALSE;
    }
    return TRUE;
}

void builtin_close(void) {
    apr_pool_destroy(builtin_pool);
    builtin_hash = NULL;
}

apr_pool_t* builtin_get_pool(void) {
    return builtin_pool;
}

hash_definition_t* builtin_get_hash_definition(void) {
    return builtin_hash;
}

void builtin_run(builtin_ctx_t* ctx, void* concrete_ctx, void (*pfn_action)(void* concrete_builtin_ctx),
                 apr_pool_t* root) {
    if(!builtin_init(ctx, root)) {
        return;
    }

    pfn_action(concrete_ctx);

    builtin_close();
}

apr_byte_t* builtin_hash_from_string(const char* string) {

    apr_byte_t* digest = apr_pcalloc(builtin_pool, sizeof(apr_byte_t) * builtin_hash->hash_length_);

    // some hashes like NTLM required unicode string so convert multi byte string to unicode one
    if(builtin_hash->use_wide_string_) {
        wchar_t* str = enc_from_ansi_to_unicode(string, builtin_pool);
        builtin_hash->pfn_digest_(digest, str, wcslen(str) * sizeof(wchar_t));
    } else {
        builtin_hash->pfn_digest_(digest, string, strlen(string));
    }
    return digest;
}

void builtin_output_both_file_and_console(FILE* file, out_context_t* ctx) {
    out_output_to_console(ctx);

    lib_fprintf(file, "%s", ctx->string_to_print_); //-V111
    if(ctx->is_print_separator_) {
        lib_fprintf(file, FILE_INFO_COLUMN_SEPARATOR);
    }
    if(ctx->is_finish_line_) {
        lib_fprintf(file, NEW_LINE);
    }
}

BOOL builtin_allow_sfv_option(BOOL result_in_sfv) {
    const char* hash = builtin_get_hash_definition()->name_;
    if(result_in_sfv && (0 != strcmp(hash, "crc32") && 0 != strcmp(hash, "crc32c"))) {
        lib_printf(_("\n --sfv option doesn't support %s algorithm. Only crc32 or crc32c supported"), hash);
        return FALSE;
    }
    return TRUE;
}

apr_size_t fhash_get_digest_size(void) {
    return builtin_hash->hash_length_;
}

int fhash_compare_digests(apr_byte_t* digest1, apr_byte_t* digest2) {
    return memcmp(digest1, digest2, builtin_hash->hash_length_) == 0;
}

void fhash_to_digest(const char* hash, apr_byte_t* digest) {
    lib_hex_str_2_byte_array(hash, digest, builtin_hash->hash_length_);
}

void fhash_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    builtin_hash->pfn_digest_(digest, input, input_len);
}

void fhash_init_hash_context(void* context) {
    builtin_hash->pfn_init_(context);
}

void fhash_final_hash(void* context, apr_byte_t* digest) {
    builtin_hash->pfn_final_(context, digest);
}

void fhash_update_hash(void* context, const void* input, const apr_size_t input_len) {
    builtin_hash->pfn_update_(context, input, input_len);
}

void* fhash_allocate_context(apr_pool_t* p) {
    return apr_pcalloc(p, builtin_hash->context_size_);
}
