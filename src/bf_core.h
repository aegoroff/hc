/*!
 * Pool-free brute-force core for the Zig hc build (H1 hybrid).
 * Hot loops only — orchestration (threads, alloc, probe, I/O) lives in bf.zig.
 * CMake continues to use src/srclib/bf.c + APR; this header is Zig-only.
 */
#ifndef HC_BF_CORE_H_
#define HC_BF_CORE_H_

#include "gpu_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* UTF-16 code units for the wide (NTLM) path. Matches Windows wchar_t width
 * and unix char16_t without pulling wchar.h/uchar.h into translate-c. */
typedef uint16_t bf_wide_char_t;

#ifndef GPU_ATTEMPT_SIZE
#define GPU_ATTEMPT_SIZE 16
#endif

#define BF_DIGITS "0123456789"
#define BF_ASCII_TPL "ASCII"
#define BF_DIGITS_TPL "0-9"
#define BF_LOW_CASE "abcdefghijklmnopqrstuvwxyz"
#define BF_LOW_CASE_TPL "a-z"
#define BF_UPPER_CASE "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#define BF_UPPER_CASE_TPL "A-Z"
#define BF_MAX_DEFAULT 10

typedef struct bf_cpu_ctx {
    unsigned char *pass_;
    bf_wide_char_t *wide_pass_;
    size_t *chars_indexes_;
    uint64_t num_of_attempts_;
    size_t thread_num_;
    uint32_t passmin_;
    uint32_t passmax_;
    uint32_t pass_length_;
    uint32_t work_thread_;
    uint32_t num_of_threads;
    BOOL use_wide_pass_;
    BOOL found_in_the_thread_;
} bf_cpu_ctx_t;

typedef int (*bf_hash_compare_fn)(void *hash, const void *pass, const uint32_t length);

/** Bind dictionary + target hash + compare callback for the next run. */
void bf_core_set_context(const unsigned char *dict, size_t dict_len, void *hash_to_find,
                         bf_hash_compare_fn compare);

/** Clear found flag and attempt counter before a run. */
void bf_core_reset(void);

BOOL bf_core_is_found(void);
void bf_core_set_found(BOOL found);
uint64_t bf_core_get_attempts(void);
void bf_core_add_attempts(uint64_t n);

/** CPU worker body (call from std.Thread). */
void bf_core_cpu_worker(bf_cpu_ctx_t *ctx);

/** GPU worker body (call from std.Thread). variants_ must be preallocated. */
void bf_core_gpu_worker(gpu_tread_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* HC_BF_CORE_H_ */
