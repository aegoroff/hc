#include "bf_core.h"
#include "ocl_common.h"

#include <assert.h>

static unsigned int run_calls;
static unsigned int sync_calls;
static unsigned int cleanup_calls;
static unsigned int fail_on_run;
static hc_ocl_algo_t algo;
static uint32_t max_len_arg;

static cl_int write_buffer(cl_command_queue queue, cl_mem buffer, cl_bool blocking,
                           size_t offset, size_t size, const void *data, cl_uint count,
                           const cl_event *events, cl_event *event) {
    return CL_SUCCESS;
}

static cl_int set_arg(cl_kernel kernel, cl_uint index, size_t size, const void *value) {
    if (index == 9 && size == sizeof(max_len_arg) && value) max_len_arg = *(const uint32_t *)value;
    return CL_SUCCESS;
}

static cl_int enqueue_kernel(cl_command_queue queue, cl_kernel kernel, cl_uint dims,
                             const size_t *offset, const size_t *global, const size_t *local,
                             cl_uint count, const cl_event *events, cl_event *event) {
    ++run_calls;
    return run_calls == fail_on_run ? -54 /* CL_INVALID_WORK_GROUP_SIZE */ : CL_SUCCESS;
}

static cl_int read_buffer(cl_command_queue queue, cl_mem buffer, cl_bool blocking,
                          size_t offset, size_t size, void *data, cl_uint count,
                          const cl_event *events, cl_event *event) {
    return CL_SUCCESS;
}

static hc_ocl_api_t api = {
    .clEnqueueWriteBuffer = write_buffer,
    .clSetKernelArg = set_arg,
    .clEnqueueNDRangeKernel = enqueue_kernel,
    .clEnqueueReadBuffer = read_buffer,
};

hc_ocl_api_t *hc_ocl_runtime_api(void) { return &api; }
cl_context hc_ocl_runtime_context(void) { return NULL; }
cl_device_id hc_ocl_runtime_device(void) { return NULL; }
cl_command_queue hc_ocl_runtime_queue(void) { return (cl_command_queue)&api; }

BOOL gpu_init_pipeline(gpu_tread_ctx_t *ctx) { return TRUE; }

void gpu_run(gpu_tread_ctx_t *ctx, size_t dict_len,
             void (*kernel)(gpu_tread_ctx_t *, size_t)) {
    kernel(ctx, dict_len);
}

void gpu_synchronize(gpu_tread_ctx_t *ctx) {
    assert(!ctx->launch_failed_);
    ++sync_calls;
    ctx->launch_in_flight_ = FALSE;
}

void gpu_cleanup(gpu_tread_ctx_t *ctx) {
    assert(!ctx->launch_in_flight_);
    ++cleanup_calls;
    hc_ocl_run_active_cleanup();
    ctx->dev_result_ = NULL;
}

static void prepare(int device_ix, const unsigned char *dict, size_t dict_len,
                    const unsigned char *hash, gpu_tread_ctx_t *ctx) {
    algo.ready = 1;
    algo.kernel = (cl_kernel)&algo;
    hc_ocl_set_active_algo(&algo);
    ctx->dev_result_ = (unsigned char *)&algo;
}

static void run(void *ctx, size_t dict_len) {
    hc_ocl_algo_entry_run(ctx, dict_len);
}

int main(void) {
    for (fail_on_run = 1; fail_on_run <= 2; ++fail_on_run) {
        /* Arrange */
        run_calls = sync_calls = cleanup_calls = 0;
        gpu_context_t gpu = {
            .pfn_prepare_ = prepare,
            .pfn_run_ = run,
        };
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
        assert(run_calls == fail_on_run);
        assert(sync_calls == fail_on_run - 1);
        assert(cleanup_calls == 1);
        assert(max_len_arg == 5);
        assert(bf_core_get_attempts() == (fail_on_run - 1) * 5);
        assert(!ctx.found_in_the_thread_);
        assert(ctx.dev_result_ == NULL);
    }
    return 0;
}
