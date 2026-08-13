#ifndef HC_GPU_C_H_
#define HC_GPU_C_H_

/*
 * translate-c root for gpu.zig — the canonical GPU ABI (gpu_abi.h) plus the
 * per-algorithm CUDA/stub entry points (md5.h, crc32cu.h, sha*.h, ...). Include
 * paths are set on the TranslateC step in build.zig (src/abi + src/cuda_include).
 */

#include "gpu_abi.h"

#include "blake2b.h"
#include "blake2s.h"
#include "blake3.h"
#include "md2.h"
#include "md4.h"
#include "md5.h"
#include "rmd128.h"
#include "rmd160.h"
#include "rmd256.h"
#include "rmd320.h"
#include "sha1.h"
#include "sha3.h"
#include "tiger.h"
#include "sha224.h"
#include "sha256.h"
#include "sha384.h"
#include "sha512.h"
#include "whirlpool.h"
#include "crc32cu.h"

#endif /* HC_GPU_C_H_ */
