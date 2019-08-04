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
#include <apr_tables.h>
#include <apr_strings.h>
#include <lib.h>
#include "backend.h"
#include "processor.h"

pcre2_general_context* pcre_context = NULL;

static apr_pool_t* proc_pool = NULL;

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
    proc_pool = pool;
    pcre_context = pcre2_general_context_create(&pcre_alloc, &pcre_free, NULL);
}

void proc_complete() {
    pcre2_general_context_free(pcre_context);
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
