/*!
 * \brief   The file contains directory traverse implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "traverse.h"
#include "encoding.h"

#define COMPOSITE_PATTERN_INIT_SZ 8 // composite pattern array init size
#define SUBDIRS_ARRAY_INIT_SZ 16 // subdirectories array init size
#define PATTERN_SEPARATOR ";"

int MatchToCompositePattern(const char* str, apr_array_header_t* pattern)
{
    int i = 0;

    if (!pattern) {
        return TRUE;    // important
    }
    if (!str) {
        return FALSE;   // important
    }

    for (; i < pattern->nelts; ++i) {
        const char* p = ((const char**)pattern->elts)[i];
        if (apr_fnmatch(p, str, APR_FNM_CASE_BLIND) == APR_SUCCESS) {
            return TRUE;
        }
    }

    return FALSE;
}

const char* HackRootPath(const char* path, apr_pool_t* pool)
{
    size_t len = 0;
    
    if (path == NULL) {
        return path;
    }
    len = strlen(path);
    return path[len - 1] == ':' ? apr_pstrcat(pool, path, "\\", NULL) : path;
}

void CompilePattern(const char* pattern, apr_array_header_t** newpattern, apr_pool_t* pool)
{
    char* parts = NULL;
    char* last = NULL;
    char* p = NULL;

    if (!pattern) {
        return; // important
    }

    *newpattern = apr_array_make(pool, COMPOSITE_PATTERN_INIT_SZ, sizeof(const char*));

    parts = apr_pstrdup(pool, pattern);    /* strtok wants non-const data */
    p = apr_strtok(parts, PATTERN_SEPARATOR, &last);
    while (p) {
        *(const char**)apr_array_push(*newpattern) = p;
        p = apr_strtok(NULL, PATTERN_SEPARATOR, &last);
    }
}

void TraverseDirectory(
    const char* dir,
    TraverseContext* ctx,
    BOOL (*filter)(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* pool), 
    apr_pool_t* pool)
{
    apr_finfo_t info = { 0 };
    apr_dir_t* d = NULL;
    apr_status_t status = APR_SUCCESS;
    char* fullPath = NULL; // Full path to file or subdirectory
    apr_pool_t* iterPool = NULL;
    apr_array_header_t* subdirs = NULL;
    OutputContext output = { 0 };

    if (ctx->PfnFileHandler == NULL || dir == NULL) {
        return;
    }

    status = apr_dir_open(&d, dir, pool);
    if (status != APR_SUCCESS) {
        output.StringToPrint = FromUtf8ToAnsi(dir, pool);
        output.IsPrintSeparator = TRUE;
        ((DataContext*)ctx->DataCtx)->PfnOutput(&output);

        OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
        return;
    }

    if (ctx->IsScanDirRecursively) {
        subdirs = apr_array_make(pool, SUBDIRS_ARRAY_INIT_SZ, sizeof(const char*));
    }

    apr_pool_create(&iterPool, pool);
    for (;;) {
        apr_pool_clear(iterPool);  // cleanup file allocated memory
        status = apr_dir_read(&info, APR_FINFO_NAME | APR_FINFO_MIN, d);
        if (APR_STATUS_IS_ENOENT(status)) {
            break;
        }
        if (info.name == NULL) { // to avoid access violation
            OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
            continue;
        }
        // Subdirectory handling code
        if ((info.filetype == APR_DIR) && ctx->IsScanDirRecursively) {
            // skip current and parent dir
            if (((info.name[0] == '.') && (info.name[1] == '\0'))
                || ((info.name[0] == '.') && (info.name[1] == '.') && (info.name[2] == '\0'))) {
                continue;
            }

            status = apr_filepath_merge(&fullPath,
                                        dir,
                                        info.name,
                                        APR_FILEPATH_NATIVE,
                                        pool); // IMPORTANT: so as not to use strdup
            if (status != APR_SUCCESS) {
                OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
                continue;
            }
            *(const char**)apr_array_push(subdirs) = fullPath;
        } // End subdirectory handling code

        if ((status != APR_SUCCESS) || (info.filetype != APR_REG)) {
            continue;
        }

        if(filter != NULL && !filter(&info, dir, ctx, iterPool) ) {
            continue;
        }

        status = apr_filepath_merge(&fullPath,
                                    dir,
                                    info.name,
                                    APR_FILEPATH_NATIVE,
                                    iterPool);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
            continue;
        }

        if (ctx->PfnFileHandler(fullPath, ctx->DataCtx, iterPool) != APR_SUCCESS) {
            continue; // or break if you want to interrupt in case of any file handling error
        }
    }

    status = apr_dir_close(d);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
    }

    // scan subdirectories found
    if (ctx->IsScanDirRecursively) {
        size_t i = 0;
        for (; i < subdirs->nelts; ++i) {
            const char* path = ((const char**)subdirs->elts)[i];
            apr_pool_clear(iterPool);
            TraverseDirectory(path, ctx, filter, iterPool);
        }
    }

    apr_pool_destroy(iterPool);
}

BOOL FilterByName(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* pool)
{
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(dir);
    UNREFERENCED_PARAMETER(pool);
#endif
    if (!MatchToCompositePattern(info->name, ctx->IncludePattern)) {
        return FALSE;
    }
    // IMPORTANT: check pointer here otherwise the logic will fail
    if (ctx->ExcludePattern &&
        MatchToCompositePattern(info->name, ctx->ExcludePattern)) {
        return FALSE;
    }
    return TRUE;
}