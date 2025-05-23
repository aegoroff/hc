/*!
 * \brief   The file contains file hash implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2025
 */

#include "apr_strings.h"
#include "apr_hash.h"
#include "filehash.h"
#include "lib.h"
#include "encoding.h"
#include "intl.h"

#define FILE_BIG_BUFFER_SIZE (1 * BINARY_THOUSAND * BINARY_THOUSAND)  // 1 megabyte

#define VERIFY_FORMAT "%s    %s"
#define APP_ERROR_OR_SEARCH_MODE "%s | %s"
#define APP_SHORT_FORMAT "%s | %s | %s"
#define APP_FULL_FORMAT "%s | %s | %s | %s"
#define KEY_FILE "file"
#define KEY_TIME "time"
#define KEY_SIZE "size"
#define KEY_HASH "hash"
#define KEY_ERR_OPEN "open_file"
#define KEY_ERR_INFO "info_file"
#define KEY_ERR_OFFSET "offset_file"
#define KEY_ERR_CLOSE "close_file"
#define KEY_ERR_HASH "hash_file"
#define VALID _("File is valid")
#define INVALID _("File is invalid")

void fhash_calculate_file(const char* full_path_to_file, data_ctx_t* ctx, apr_pool_t* pool) {
    apr_file_t* file_handle = NULL;
    apr_finfo_t info = { 0 };
    apr_status_t status = APR_SUCCESS;
    int result = TRUE;
    int do_not_output_results = FALSE;

    char* file_ansi = NULL;
    int is_zero_search_hash = FALSE;

    apr_byte_t* digest = NULL;
    apr_byte_t* digest_to_compare = NULL;

    apr_pool_t* file_pool = NULL;
    out_context_t output = { 0 };
    apr_hash_t* results_container = NULL;
    BOOL error = FALSE;
    const char* validation_message = NULL;
    lib_time_t time = { 0 };

    const int is_print_sfv = ctx->is_print_sfv_;
    const int is_print_verify = ctx->is_print_verify_;
    const int is_validate_file_by_hash = ctx->is_validate_file_by_hash_;
    const char* hash_to_search = ctx->hash_to_search_;

    apr_pool_create(&file_pool, pool);
    results_container = apr_hash_make(file_pool);
    digest = (apr_byte_t*)apr_pcalloc(file_pool, sizeof(apr_byte_t) * fhash_get_digest_size());
    if(hash_to_search) {
        digest_to_compare = (apr_byte_t*)apr_pcalloc(file_pool, sizeof(apr_byte_t) * fhash_get_digest_size());
    }

    status = apr_file_open(&file_handle, full_path_to_file, APR_READ | APR_BINARY, APR_FPROT_WREAD, file_pool);
    file_ansi = enc_from_utf8_to_ansi(full_path_to_file, file_pool);

    // File name or path depends on mode
    if(is_print_sfv) {
        apr_hash_set(results_container, KEY_FILE, APR_HASH_KEY_STRING,
                     file_ansi == NULL ? lib_get_file_name(full_path_to_file) : lib_get_file_name(file_ansi));
    } else {
        apr_hash_set(results_container, KEY_FILE, APR_HASH_KEY_STRING,
                     file_ansi == NULL ? full_path_to_file : file_ansi);
    }

    if(status != APR_SUCCESS) {
        if(!is_print_sfv && !is_print_verify) {
            apr_hash_set(results_container, KEY_ERR_OPEN, APR_HASH_KEY_STRING,
                         out_create_error_message(status, file_pool));
        }
        goto outputResults;
    }

    status = apr_file_info_get(&info, APR_FINFO_MIN, file_handle);

    if(status != APR_SUCCESS) {
        apr_hash_set(results_container, KEY_ERR_INFO, APR_HASH_KEY_STRING, out_create_error_message(status, file_pool));
        result = FALSE;
        goto cleanup;
    }
    apr_hash_set(results_container, KEY_SIZE, APR_HASH_KEY_STRING, out_copy_size_to_string(info.size, file_pool));

    lib_start_timer();
    if(hash_to_search) {
        fhash_to_digest(hash_to_search, digest_to_compare);
        fhash_calculate_digest(digest, "", 0);
        if(fhash_compare_digests(digest, digest_to_compare)) {
            // Empty file optimization
            is_zero_search_hash = TRUE;
        }
    }

    if(ctx->offset_ >= info.size && info.size > 0) {
        apr_hash_set(results_container, KEY_ERR_OFFSET, APR_HASH_KEY_STRING, _("Offset is greater then file size"));
    } else {
        const char* error_message = fhash_calculate_hash(file_handle, info.size, digest, ctx->limit_, ctx->offset_,
                                                         file_pool);
        if(error_message != NULL) {
            apr_hash_set(results_container, KEY_ERR_HASH, APR_HASH_KEY_STRING, error_message);
        } else {
            apr_hash_set(
                         results_container,
                         KEY_HASH,
                         APR_HASH_KEY_STRING,
                         ctx->is_base64_
                             ? out_hash_to_base64_string(digest, fhash_get_digest_size(), file_pool)
                             : out_hash_to_string(digest, ctx->is_print_low_case_, fhash_get_digest_size(), file_pool));
        }
    }
    lib_stop_timer();

    time = lib_read_elapsed_time();
    apr_hash_set(results_container, KEY_TIME, APR_HASH_KEY_STRING, out_copy_time_to_string(&time, file_pool));

    if(hash_to_search) {
        result = !is_zero_search_hash && fhash_compare_digests(digest, digest_to_compare) || is_zero_search_hash && info
                .size == 0;
    }
    if(!is_validate_file_by_hash) {
        do_not_output_results = !result;
    }
cleanup:
    status = apr_file_close(file_handle);
    if(status != APR_SUCCESS) {
        apr_hash_set(results_container, KEY_ERR_CLOSE, APR_HASH_KEY_STRING,
                     out_create_error_message(status, file_pool));
    }
    // Output results
outputResults:

    error = apr_hash_get(results_container, KEY_ERR_OPEN, APR_HASH_KEY_STRING) != NULL ||
            apr_hash_get(results_container, KEY_ERR_CLOSE, APR_HASH_KEY_STRING) != NULL ||
            apr_hash_get(results_container, KEY_ERR_OFFSET, APR_HASH_KEY_STRING) != NULL ||
            apr_hash_get(results_container, KEY_ERR_INFO, APR_HASH_KEY_STRING) != NULL;

    if(hash_to_search) {
        if(result) {
            validation_message = VALID;
        } else {
            validation_message = INVALID;
        }
    }
    if(do_not_output_results) {
        goto end;
    }

    // Output message
    if(is_print_sfv) {
        if(apr_hash_get(results_container, KEY_HASH, APR_HASH_KEY_STRING) != NULL) {
            output.string_to_print_ = apr_psprintf(file_pool, VERIFY_FORMAT,
                                                   apr_hash_get(results_container, KEY_FILE, APR_HASH_KEY_STRING),
                                                   apr_hash_get(results_container, KEY_HASH, APR_HASH_KEY_STRING));
        }
    } else if(is_print_verify) {
        if(apr_hash_get(results_container, KEY_HASH, APR_HASH_KEY_STRING) != NULL) {
            output.string_to_print_ = apr_psprintf(file_pool, VERIFY_FORMAT,
                                                   apr_hash_get(results_container, KEY_HASH, APR_HASH_KEY_STRING),
                                                   apr_hash_get(results_container, KEY_FILE, APR_HASH_KEY_STRING));
        }
    } else if(error) {
        char* error_open = apr_hash_get(results_container, KEY_ERR_OPEN, APR_HASH_KEY_STRING);
        char* error_close = apr_hash_get(results_container, KEY_ERR_CLOSE, APR_HASH_KEY_STRING);
        char* error_offset = apr_hash_get(results_container, KEY_ERR_OFFSET, APR_HASH_KEY_STRING);
        char* error_info = apr_hash_get(results_container, KEY_ERR_INFO, APR_HASH_KEY_STRING);
        char* error_hash = apr_hash_get(results_container, KEY_ERR_HASH, APR_HASH_KEY_STRING);

        char* error_message = apr_pstrcat(file_pool,
                                          error_open == NULL ? "" : error_open,
                                          error_close == NULL ? "" : error_close,
                                          error_offset == NULL ? "" : error_offset,
                                          error_info == NULL ? "" : error_info,
                                          error_hash == NULL ? "" : error_hash,
                                          NULL
                                         );
        output.string_to_print_ = apr_psprintf(file_pool, APP_ERROR_OR_SEARCH_MODE,
                                               apr_hash_get(results_container, KEY_FILE, APR_HASH_KEY_STRING),
                                               error_message);
    } else if(hash_to_search && !is_validate_file_by_hash) {
        // Search file mode
        output.string_to_print_ = apr_psprintf(
                                               file_pool,
                                               APP_ERROR_OR_SEARCH_MODE,
                                               apr_hash_get(results_container, KEY_FILE, APR_HASH_KEY_STRING),
                                               apr_hash_get(results_container, KEY_SIZE, APR_HASH_KEY_STRING)
                                              );
    } else if(ctx->is_print_calc_time_) {
        // Normal output with calc time
        output.string_to_print_ = apr_psprintf(
                                               file_pool,
                                               APP_FULL_FORMAT,
                                               apr_hash_get(results_container, KEY_FILE, APR_HASH_KEY_STRING),
                                               apr_hash_get(results_container, KEY_SIZE, APR_HASH_KEY_STRING),
                                               apr_hash_get(results_container, KEY_TIME, APR_HASH_KEY_STRING),
                                               validation_message == NULL
                                                   ? apr_hash_get(results_container, KEY_HASH, APR_HASH_KEY_STRING)
                                                   : validation_message
                                              );
    } else {
        // Normal output without calc time
        output.string_to_print_ = apr_psprintf(
                                               file_pool,
                                               APP_SHORT_FORMAT,
                                               apr_hash_get(results_container, KEY_FILE, APR_HASH_KEY_STRING),
                                               apr_hash_get(results_container, KEY_SIZE, APR_HASH_KEY_STRING),
                                               validation_message == NULL
                                                   ? apr_hash_get(results_container, KEY_HASH, APR_HASH_KEY_STRING)
                                                   : validation_message
                                              );
    }

    // Write output
    if(output.string_to_print_ != NULL) {
        output.is_finish_line_ = TRUE;
        ctx->pfn_output_(&output);
    }
end:
    apr_pool_destroy(file_pool);
}

const char* fhash_calculate_hash(apr_file_t* file_handle,
                                 apr_off_t file_size,
                                 apr_byte_t* digest,
                                 apr_off_t limit,
                                 apr_off_t offset,
                                 apr_pool_t* pool) {
    apr_status_t status;
    apr_off_t page_size;
    apr_off_t file_part_size = MIN(limit, file_size);
    apr_off_t start_offset = offset;
    void* context = fhash_allocate_context(pool);

    fhash_init_hash_context(context);

    if(file_part_size > FILE_BIG_BUFFER_SIZE) {
        page_size = FILE_BIG_BUFFER_SIZE;
    } else if(file_part_size == 0) {
        fhash_calculate_digest(digest, "", 0);
        return NULL;
    } else {
        page_size = file_part_size;
    }

    if(offset >= file_size) {
        return NULL;
    }

    apr_size_t bytes_read;
    apr_size_t size = (apr_size_t)MIN(page_size, (file_part_size + start_offset) - offset);
    if(size + offset > file_size) {
        size = file_size - offset;
    }
    if (size > limit) {
        size = limit;
    }
    
    apr_byte_t* buffer = (apr_byte_t*)apr_pcalloc(pool, sizeof(apr_byte_t) * size);
    apr_size_t total_read = 0;

    if (offset > 0) {
        status = apr_file_seek(file_handle, APR_SET, &offset);
        if (status != APR_SUCCESS) {
            return out_create_error_message(status, pool);
        }
    }

    do {
        bytes_read = MIN(size,(limit - total_read));
        status = apr_file_read(file_handle, buffer, &bytes_read);

        if(status != APR_SUCCESS && status != APR_EOF) {
            return out_create_error_message(status, pool);
        }

        if (bytes_read > 0) {
            fhash_update_hash(context, buffer, bytes_read);
        }
        total_read += bytes_read;
    } while (status == APR_SUCCESS && total_read < limit);


    fhash_final_hash(context, digest);
    return NULL;
}
