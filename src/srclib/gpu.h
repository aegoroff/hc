/*!
 * \brief   GPU related code interface (thin wrapper around gpu_abi.h)
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#ifndef LINQ2HASH_GPU_H_
#define LINQ2HASH_GPU_H_

/*
 * Canonical GPU ABI (structs, decls, CUDA macros under __CUDACC__) lives in
 * src/zig/abi/gpu_abi.h. Keep this header as a stable include name for any
 * leftover callers; new code should include gpu_abi.h directly.
 */
#include "gpu_abi.h"

#endif // LINQ2HASH_GPU_H_
