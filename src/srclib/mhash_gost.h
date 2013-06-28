/*
 *  gosthash.h 
 *  21 Apr 1998  Markku-Juhani Saarinen <mjos@ssh.fi>
 * 
 *  GOST R 34.11-94, Russian Standard Hash Function 
 *  header with function prototypes.
 *
 *  Copyright (c) 1998 SSH Communications Security, Finland
 *  All rights reserved.                    
 */


#if !defined(GOSTHASH_H)
#define GOSTHASH_H

/*
   State structure 
 */

#include "lib.h"

typedef struct {
	uint32_t sum[8];
	uint32_t hash[8];
	uint32_t len[8];
	uint8_t partial[32];
	uint32_t partial_bytes;
} GostHashCtx;

/*
   Compute some lookup-tables that are needed by all other functions. 
 */

#if 0
void gosthash_init(void);
#endif

/*
   Clear the state of the given context structure. 
 */

void gosthash_reset(GostHashCtx * ctx);

/*
   Mix in len bytes of data for the given buffer. 
 */

void gosthash_update(GostHashCtx * ctx, const uint8_t * buf, uint32_t len);

/*
   Compute and save the 32-byte digest. 
 */

void gosthash_final(GostHashCtx * ctx, uint8_t * digest);

#endif /*
	      GOSTHASH_H 
	    */

