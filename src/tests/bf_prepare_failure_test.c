#include "bf_core.h"

#include <assert.h>

static unsigned int prepare_calls;
static unsigned int run_calls;
static unsigned int cleanup_calls;

BOOL gpu_init_pipeline(gpu_tread_ctx_t *ctx) {
    (void)ctx;
    return TRUE;
}

void gpu_synchronize(gpu_tread_ctx_t *ctx) {
    (void)ctx;
    assert(0 && "failed preparation must not synchronize a launch");
}

void gpu_cleanup(gpu_tread_ctx_t *ctx) {
    (void)ctx;
    ++cleanup_calls;
}

static void fail_prepare(int device_ix, const unsigned char *dict, size_t dict_len,
                         const unsigned char *hash, gpu_tread_ctx_t *ctx) {
    (void)device_ix;
    (void)dict;
    (void)dict_len;
    (void)hash;
    ++prepare_calls;
    ctx->dev_result_ = NULL;
}

static void count_run(void *ctx, size_t dict_len) {
    (void)ctx;
    (void)dict_len;
    ++run_calls;
}

int main(void) {
    /* Arrange */
    gpu_context_t gpu = { .pfn_prepare_ = fail_prepare, .pfn_run_ = count_run };
    gpu_tread_ctx_t ctx = {
        .gpu_context_ = &gpu,
        .passmax_ = 5,
        .max_gpu_blocks_number_ = 1,
        .max_threads_per_block_ = 1,
    };
    bf_core_set_context((const unsigned char *)"ab", 2, NULL, NULL);
    bf_core_reset();

    /* Act */
    bf_core_gpu_worker(&ctx);

    /* Assert */
    assert(prepare_calls == 1);
    assert(run_calls == 0);
    assert(cleanup_calls == 1);
    assert(bf_core_get_attempts() == 0);
    assert(!ctx.found_in_the_thread_);
    return 0;
}
