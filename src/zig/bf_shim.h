#ifndef HC_BF_SHIM_H_
#define HC_BF_SHIM_H_

#include "apr_pools.h"
#include "apr.h"

#ifdef __cplusplus
extern "C" {
#endif

void bf_shim_set(void (*digest)(apr_byte_t *digest, const void *string, const apr_size_t input_len),
                 apr_size_t hash_len);

#ifdef __cplusplus
}
#endif

#endif /* HC_BF_SHIM_H_ */
