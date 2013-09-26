/*!
 * \brief   The file contains file hash implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include <string.h>
#include "apr_mmap.h"
#include "filehash.h"
#include "lib.h"
#include "encoding.h"
#include "implementation.h"

#define FILE_BIG_BUFFER_SIZE 1 * BINARY_THOUSAND * BINARY_THOUSAND  // 1 megabyte

void CheckHash(apr_byte_t* digest, const char* checkSum, DataContext* ctx)
{
    OutputContext output = { 0 };
    output.StringToPrint = "File is ";
    ctx->PfnOutput(&output);
    output.StringToPrint = CompareHash(digest, checkSum) ? "valid" : "invalid";
    ctx->PfnOutput(&output);
}

apr_status_t CalculateFile(const char* fullPathToFile, DataContext* ctx, apr_pool_t* pool)
{
    apr_byte_t digest[DIGESTSIZE];
    size_t len = 0;
    apr_status_t status = APR_SUCCESS;

    if (!CalculateFileHash(fullPathToFile, digest, ctx->IsPrintCalcTime, ctx->IsPrintSfv,
                           ctx->HashToSearch, ctx->Limit, ctx->Offset, ctx->PfnOutput, pool)) {
        return status;
    }

    OutputDigest(digest, ctx, GetDigestSize(), pool);

    if (!(ctx->FileToSave)) {
        return status;
    }
    apr_file_printf(ctx->FileToSave, HashToString(digest, ctx->IsPrintLowCase, GetDigestSize(), pool));

    len = strlen(fullPathToFile);

    while (len > 0 && *(fullPathToFile + (len - 1)) != PATH_ELT_SEPARATOR) {
        --len;
    }

    apr_file_printf(ctx->FileToSave,
                    HASH_FILE_COLUMN_SEPARATOR "%s" NEW_LINE,
                    fullPathToFile + len);
    return status;
}

const char* GetFileName(const char *path)
{
    const char* filename = strrchr(path, '\\');

    if (filename == NULL) {
        filename = path;
    } else {
        filename++;
    }
    return filename;
}

int CalculateFileHash(const char* filePath,
                      apr_byte_t* digest,
                      int         isPrintCalcTime,
                      int         isPrintSfv,
                      const char* hashToSearch,
                      apr_off_t   limit,
                      apr_off_t   offset,
                      void        (* PfnOutput)(OutputContext* ctx),
                      apr_pool_t* pool)
{
    apr_file_t* fileHandle = NULL;
    apr_finfo_t info = { 0 };
    apr_status_t status = APR_SUCCESS;
    int result = TRUE;
    int r = TRUE;
    char* fileAnsi = NULL;
    char* root = NULL;
    int isZeroSearchHash = FALSE;
    apr_byte_t digestToCompare[DIGESTSIZE];
    OutputContext output = { 0 };

    fileAnsi = FromUtf8ToAnsi(filePath, pool);
    if (!hashToSearch) {
        if (isPrintSfv) {
            output.StringToPrint = fileAnsi == NULL ? GetFileName(filePath) : GetFileName(fileAnsi);
        } else {
            output.StringToPrint = fileAnsi == NULL ? filePath : fileAnsi;
        }
        output.IsPrintSeparator = isPrintSfv ? FALSE : TRUE;
        PfnOutput(&output);
    }
    
    status = apr_file_open(&fileHandle, filePath, APR_READ | APR_BINARY, APR_FPROT_WREAD, pool);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
        return FALSE;
    }

    status = apr_file_info_get(&info, APR_FINFO_MIN | APR_FINFO_NAME, fileHandle);

    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
        result = FALSE;
        goto cleanup;
    }

    if (isPrintSfv) {
        output.StringToPrint = "    ";
        output.IsPrintSeparator = FALSE;
        PfnOutput(&output);
    }

    if (!hashToSearch && !isPrintSfv) {
        output.IsPrintSeparator = TRUE;
        output.IsFinishLine = FALSE;
        output.StringToPrint = CopySizeToString(info.size, pool);
        PfnOutput(&output);
    }

    StartTimer();
    if (hashToSearch) {
        ToDigest(hashToSearch, digestToCompare);
        CalculateDigest(digest, "", 0);
        if (CompareDigests(digest, digestToCompare)) { // Empty file optimization
            isZeroSearchHash = TRUE;
            goto endtiming;
        }
    }

    if (offset >= info.size && info.size > 0) {
        output.IsFinishLine = TRUE;
        output.IsPrintSeparator = FALSE;
        output.StringToPrint = "Offset is greater then file size";
        PfnOutput(&output);
        result = FALSE;
        goto endtiming;
    }
    CalculateHash(fileHandle, info.size, digest, limit, offset, PfnOutput, pool);
endtiming:
    StopTimer();

    if (!hashToSearch) {
        goto printtime;
    }

    result = FALSE;
    r = (!isZeroSearchHash && CompareDigests(digest, digestToCompare)) || (isZeroSearchHash && (info.size == 0));
    if (ComparisonFailure(r)) {
        goto printtime;
    }

    output.IsFinishLine = FALSE;
    output.IsPrintSeparator = TRUE;

    // file name
    output.StringToPrint = fileAnsi == NULL ? filePath : fileAnsi;
    PfnOutput(&output);

    // file size
    output.StringToPrint = CopySizeToString(info.size, pool);

    if (isPrintCalcTime && !isPrintSfv) {
        output.IsPrintSeparator = TRUE;
        PfnOutput(&output); // file size output before time

        // time
        output.StringToPrint = CopyTimeToString(ReadElapsedTime(), pool);
    }
    output.IsFinishLine = TRUE;
    output.IsPrintSeparator = FALSE;
    PfnOutput(&output); // file size or time output

printtime:
    if (isPrintCalcTime && !hashToSearch && !isPrintSfv) {
        // time
        output.StringToPrint = CopyTimeToString(ReadElapsedTime(), pool);
        output.IsFinishLine = FALSE;
        output.IsPrintSeparator = TRUE;
        PfnOutput(&output);
    }
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
    }
cleanup:
    status = apr_file_close(fileHandle);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
    }
    return result;
}

void CalculateHash(apr_file_t* fileHandle,
                   apr_off_t fileSize,
                   apr_byte_t* digest,
                   apr_off_t   limit,
                   apr_off_t   offset,
                   void        (* PfnOutput)(OutputContext* ctx),
                   apr_pool_t* pool)
{
    apr_status_t status = APR_SUCCESS;
    apr_off_t pageSize = 0;
    apr_off_t filePartSize = 0;
    apr_off_t startOffset = offset;
    apr_mmap_t* mmap = NULL;
    void* context = NULL;

    context = AllocateContext(pool);
    InitContext(context);

    filePartSize = MIN(limit, fileSize);

    if (filePartSize > FILE_BIG_BUFFER_SIZE) {
        pageSize = FILE_BIG_BUFFER_SIZE;
    } else if (filePartSize == 0) {
        CalculateDigest(digest, "", 0);
        goto cleanup;
    } else {
        pageSize = filePartSize;
    }

    if (offset >= fileSize) {
        goto cleanup;
    }

    do {
        apr_size_t size = (apr_size_t)MIN(pageSize, (filePartSize + startOffset) - offset);

        if (size + offset > fileSize) {
            size = fileSize - offset;
        }

        status =
            apr_mmap_create(&mmap, fileHandle, offset, size, APR_MMAP_READ, pool);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, PfnOutput, pool);
            mmap = NULL;
            goto cleanup;
        }
        UpdateHash(context, mmap->mm, mmap->size);
        offset += mmap->size;
        status = apr_mmap_delete(mmap);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, PfnOutput, pool);
            mmap = NULL;
            goto cleanup;
        }
        mmap = NULL;
    } while (offset < filePartSize + startOffset && offset < fileSize);
    FinalHash(context, digest);
cleanup:
    if (mmap == NULL) {
        return;
    }
    status = apr_mmap_delete(mmap);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
    }
}

void OutputDigest(apr_byte_t* digest, DataContext* ctx, apr_size_t sz, apr_pool_t* pool)
{
    OutputContext output = { 0 };
    output.IsFinishLine = TRUE;
    output.IsPrintSeparator = FALSE;
    output.StringToPrint = HashToString(digest, ctx->IsPrintLowCase, sz, pool);
    ctx->PfnOutput(&output);
}