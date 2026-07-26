#ifndef HC_APR_SHIM_APR_GENERAL_H
#define HC_APR_SHIM_APR_GENERAL_H

/* See apr.h: translate-c-only shim. Mirrors the init API bf.zig uses. */

#include "apr.h"

#define APR_SUCCESS 0

apr_status_t apr_initialize(void);
void apr_terminate(void);

#endif /* HC_APR_SHIM_APR_GENERAL_H */
