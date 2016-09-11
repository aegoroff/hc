/*!
 * \brief   The file contains file hash implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2015
 */

#include "apr_mmap.h"
#include "apr_strings.h"
#include "apr_hash.h"
#include "filehash.h"
#include "lib.h"
#include "encoding.h"

#define FILE_BIG_BUFFER_SIZE 1 * BINARY_THOUSAND * BINARY_THOUSAND  // 1 megabyte
#define ARRAY_INIT_SZ 4

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
#define FILE_IS "File is "
#define VALID FILE_IS "valid"
#define INVALID FILE_IS "invalid"

 void fhash_calculate_file(const char* fullPathToFile, data_ctx_t* ctx, apr_pool_t* pool) {
    apr_file_t* fileHandle = NULL;
    apr_finfo_t info = {0};
    apr_status_t status = APR_SUCCESS;
    int result = TRUE;
    int doNotOutputResults = FALSE;

    char* fileAnsi = NULL;
    int isZeroSearchHash = FALSE;

    apr_byte_t* digest = NULL;
    apr_byte_t* digestToCompare = NULL;

    apr_pool_t* filePool = NULL;
    out_context_t output = {0};
    apr_hash_t* message = NULL;
    BOOL error = FALSE;
    const char* validationMessage = NULL;
    lib_time_t time = { 0 };

    int isPrintSfv = ctx->IsPrintSfv;
    int isPrintVerify = ctx->IsPrintVerify;
    int isValidateFileByHash = ctx->IsValidateFileByHash;
    const char* hashToSearch = ctx->HashToSearch;

    apr_pool_create(&filePool, pool);
    message = apr_hash_make(filePool);
    digest = (apr_byte_t*)apr_pcalloc(filePool, sizeof(apr_byte_t) * fhash_get_digest_size());
    if(hashToSearch) {
        digestToCompare = (apr_byte_t*)apr_pcalloc(filePool, sizeof(apr_byte_t) * fhash_get_digest_size());
    }

    status = apr_file_open(&fileHandle, fullPathToFile, APR_READ | APR_BINARY, APR_FPROT_WREAD, filePool);
    fileAnsi = enc_from_utf8_to_ansi(fullPathToFile, filePool);

    // File name or path depends on mode
    if(isPrintSfv) {
        apr_hash_set(message, KEY_FILE, APR_HASH_KEY_STRING, fileAnsi == NULL ? lib_get_file_name(fullPathToFile) : lib_get_file_name(fileAnsi));
    }
    else {
        apr_hash_set(message, KEY_FILE, APR_HASH_KEY_STRING, fileAnsi == NULL ? fullPathToFile : fileAnsi);
    }

    if(status != APR_SUCCESS) {
        if(!isPrintSfv && !isPrintVerify) {
            apr_hash_set(message, KEY_ERR_OPEN, APR_HASH_KEY_STRING, out_create_error_message(status, filePool));
        }
        goto outputResults;
    }

    status = apr_file_info_get(&info, APR_FINFO_MIN | APR_FINFO_NAME, fileHandle);

    if(status != APR_SUCCESS) {
        apr_hash_set(message, KEY_ERR_INFO, APR_HASH_KEY_STRING, out_create_error_message(status, filePool));
        result = FALSE;
        goto cleanup;
    }
    apr_hash_set(message, KEY_SIZE, APR_HASH_KEY_STRING, out_copy_size_to_string(info.size, filePool));

    lib_start_timer();
    if(hashToSearch) {
        fhash_to_digest(hashToSearch, digestToCompare);
        fhash_calculate_digest(digest, "", 0);
        if(fhash_compare_digests(digest, digestToCompare)) { // Empty file optimization
            isZeroSearchHash = TRUE;
        }
    }

    if(ctx->Offset >= info.size && info.size > 0) {
        apr_hash_set(message, KEY_ERR_OFFSET, APR_HASH_KEY_STRING, "Offset is greater then file size");
    }
    else {
        const char* msg = fhash_calculate_hash(fileHandle, info.size, digest, ctx->Limit, ctx->Offset, filePool);
        if(msg != NULL) {
            apr_hash_set(message, KEY_ERR_HASH, APR_HASH_KEY_STRING, msg);
        }
        else {
            apr_hash_set(message, KEY_HASH, APR_HASH_KEY_STRING, out_hash_to_string(digest, ctx->IsPrintLowCase, fhash_get_digest_size(), filePool));
        }
    }
    lib_stop_timer();

    time = lib_read_elapsed_time();
    apr_hash_set(message, KEY_TIME, APR_HASH_KEY_STRING, out_copy_time_to_string(time, filePool));

    if(hashToSearch) {
        result = !isZeroSearchHash && fhash_compare_digests(digest, digestToCompare) || isZeroSearchHash && info.size == 0;
    }
    if(!isValidateFileByHash) {
        doNotOutputResults = !result;
    }
cleanup:
    status = apr_file_close(fileHandle);
    if(status != APR_SUCCESS) {
        apr_hash_set(message, KEY_ERR_CLOSE, APR_HASH_KEY_STRING, out_create_error_message(status, filePool));
    }
    // Output results
outputResults:

    error = apr_hash_get(message, KEY_ERR_OPEN, APR_HASH_KEY_STRING) != NULL ||
            apr_hash_get(message, KEY_ERR_CLOSE, APR_HASH_KEY_STRING) != NULL ||
            apr_hash_get(message, KEY_ERR_OFFSET, APR_HASH_KEY_STRING) != NULL ||
            apr_hash_get(message, KEY_ERR_INFO, APR_HASH_KEY_STRING) != NULL;

    if(hashToSearch) {
        if(result) {
            validationMessage = VALID;
        }
        else {
            validationMessage = INVALID;
        }
    }
    if(doNotOutputResults) {
        goto end;
    }

    // Output message
    if(isPrintSfv) {
        if(apr_hash_get(message, KEY_HASH, APR_HASH_KEY_STRING) != NULL) {
            output.string_to_print_ = apr_psprintf(filePool, VERIFY_FORMAT, apr_hash_get(message, KEY_FILE, APR_HASH_KEY_STRING), apr_hash_get(message, KEY_HASH, APR_HASH_KEY_STRING));
        }
    }
    else if(isPrintVerify) {
        if(apr_hash_get(message, KEY_HASH, APR_HASH_KEY_STRING) != NULL) {
            output.string_to_print_ = apr_psprintf(filePool, VERIFY_FORMAT, apr_hash_get(message, KEY_HASH, APR_HASH_KEY_STRING), apr_hash_get(message, KEY_FILE, APR_HASH_KEY_STRING));
        }
    }
    else if(error) {
        char* errorMessage;
        char* errorOpen = apr_hash_get(message, KEY_ERR_OPEN, APR_HASH_KEY_STRING);
        char* errorClose = apr_hash_get(message, KEY_ERR_CLOSE, APR_HASH_KEY_STRING);
        char* errorOffset = apr_hash_get(message, KEY_ERR_OFFSET, APR_HASH_KEY_STRING);
        char* errorInfo = apr_hash_get(message, KEY_ERR_INFO, APR_HASH_KEY_STRING);
        char* errorHash = apr_hash_get(message, KEY_ERR_HASH, APR_HASH_KEY_STRING);

        errorMessage = apr_pstrcat(filePool,
                                   errorOpen == NULL ? "" : errorOpen,
                                   errorClose == NULL ? "" : errorClose,
                                   errorOffset == NULL ? "" : errorOffset,
                                   errorInfo == NULL ? "" : errorInfo,
                                   errorHash == NULL ? "" : errorHash,
                                   NULL
        );
        output.string_to_print_ = apr_psprintf(filePool, APP_ERROR_OR_SEARCH_MODE, apr_hash_get(message, KEY_FILE, APR_HASH_KEY_STRING), errorMessage);
    }
    else if(hashToSearch && !isValidateFileByHash) { // Search file mode
        output.string_to_print_ = apr_psprintf(
            filePool,
            APP_ERROR_OR_SEARCH_MODE,
            apr_hash_get(message, KEY_FILE, APR_HASH_KEY_STRING),
            apr_hash_get(message, KEY_SIZE, APR_HASH_KEY_STRING)
        );
    }
    else if(ctx->IsPrintCalcTime) { // Normal output with calc time
        output.string_to_print_ = apr_psprintf(
            filePool,
            APP_FULL_FORMAT,
            apr_hash_get(message, KEY_FILE, APR_HASH_KEY_STRING),
            apr_hash_get(message, KEY_SIZE, APR_HASH_KEY_STRING),
            apr_hash_get(message, KEY_TIME, APR_HASH_KEY_STRING),
            validationMessage == NULL ? apr_hash_get(message, KEY_HASH, APR_HASH_KEY_STRING) : validationMessage
        );
    }
    else { // Normal output without calc time
        output.string_to_print_ = apr_psprintf(
            filePool,
            APP_SHORT_FORMAT,
            apr_hash_get(message, KEY_FILE, APR_HASH_KEY_STRING),
            apr_hash_get(message, KEY_SIZE, APR_HASH_KEY_STRING),
            validationMessage == NULL ? apr_hash_get(message, KEY_HASH, APR_HASH_KEY_STRING) : validationMessage
        );
    }

    // Write output
    if(output.string_to_print_ != NULL) {
        output.is_finish_line_ = TRUE;
        ctx->PfnOutput(&output);
    }
end:
    apr_pool_destroy(filePool);
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
    apr_mmap_t* mmap = NULL;
    void* context = fhash_allocate_context(pool);
    const char* result = NULL;

    fhash_init_hash_context(context);

    if(file_part_size > FILE_BIG_BUFFER_SIZE) {
        page_size = FILE_BIG_BUFFER_SIZE ;
    }
    else if(file_part_size == 0) {
        fhash_calculate_digest(digest, "", 0);
        goto cleanup;
    }
    else {
        page_size = file_part_size;
    }

    if(offset >= file_size) {
        goto cleanup;
    }

    do {
        apr_size_t size = (apr_size_t)MIN(page_size, (file_part_size + start_offset) - offset);

        if(size + offset > file_size) {
            size = file_size - offset;
        }

        status =
                apr_mmap_create(&mmap, file_handle, offset, size, APR_MMAP_READ, pool);
        if(status != APR_SUCCESS) {
            result = out_create_error_message(status, pool);
            mmap = NULL;
            goto cleanup;
        }
        fhash_update_hash(context, mmap->mm, mmap->size);
        offset += mmap->size;
        status = apr_mmap_delete(mmap);
        if(status != APR_SUCCESS) {
            result = out_create_error_message(status, pool);
            mmap = NULL;
            goto cleanup;
        }
        mmap = NULL;
    }
    while(offset < file_part_size + start_offset && offset < file_size);
    fhash_final_hash(context, digest);
cleanup:
    if(mmap != NULL) {
        status = apr_mmap_delete(mmap);
        if(status != APR_SUCCESS) {
            result = out_create_error_message(status, pool);
        }
    }
    return result;
}
