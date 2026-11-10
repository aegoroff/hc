#include "bf_core.h"

#include <assert.h>
#include <string.h>

static unsigned long long counts[16];
static unsigned int run_calls;
static uint32_t last_pass_length;
static unsigned char buffer[GPU_ATTEMPT_SIZE];

BOOL gpu_init_pipeline(gpu_tread_ctx_t *ctx) {
    (void)ctx;
    return TRUE;
}

void gpu_synchronize(gpu_tread_ctx_t *ctx) {
    (void)ctx;
}

void gpu_cleanup(gpu_tread_ctx_t *ctx) {
    (void)ctx;
}

static void prepare(int device_ix, const unsigned char *dict, size_t dict_len,
                    const unsigned char *hash, gpu_tread_ctx_t *ctx) {
    (void)device_ix;
    (void)dict;
    (void)dict_len;
    (void)hash;
    ctx->dev_result_ = buffer;
}

static void run(void *context, size_t dict_len) {
    gpu_tread_ctx_t *ctx = context;
    (void)dict_len;
    ++run_calls;
    last_pass_length = ctx->pass_length_;
    counts[ctx->pass_length_] += ctx->batch_count_;
    if (ctx->pass_length_ + 2 >= ctx->passmin_) ctx->found_in_the_thread_ = TRUE;
}

static void run_case(uint32_t passmin, uint32_t passmax) {
    /* Arrange */
    gpu_context_t gpu = { .pfn_prepare_ = prepare, .pfn_run_ = run };
    gpu_tread_ctx_t ctx = {
        .gpu_context_ = &gpu,
        .passmin_ = passmin,
        .passmax_ = passmax,
        .max_gpu_blocks_number_ = 1024,
        .max_threads_per_block_ = 256,
    };
    memset(counts, 0, sizeof(counts));
    run_calls = 0;
    last_pass_length = 0;
    bf_core_set_context((const unsigned char *)"0123456789", 10, NULL, NULL);
    bf_core_reset();

    /* Act */
    bf_core_gpu_worker(&ctx);
}

int main(void) {
    run_case(10, 10);

    /* Assert: no prefix below passmin-2 may reach the GPU. */
    for (uint32_t i = 3; i < 8; ++i) assert(counts[i] == 0);
    assert(counts[8] == 4096);
    assert(run_calls == 1);

    run_case(4, 4);

    /* Assert: prefix 3 is the special four-character candidate launch. */
    assert(last_pass_length == 3);
    assert(counts[3] == 1000);
    assert(run_calls == 1);
    return 0;
}
