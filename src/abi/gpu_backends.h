#ifndef HC_GPU_BACKENDS_H_
#define HC_GPU_BACKENDS_H_
#include "gpu_abi.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Prefixed CUDA symbols (dual build). */
void cuda_gpu_get_props(device_props_t* prop);
BOOL cuda_gpu_get_device_props(int device_ix, device_props_t* prop);
BOOL cuda_gpu_can_use_gpu(void);
int cuda_gpu_driver_version(void);
int cuda_gpu_runtime_version(void);
gpu_versions_t cuda_gpu_number_to_version(int version_number);
BOOL cuda_gpu_init_pipeline(gpu_tread_ctx_t* ctx);
void cuda_gpu_synchronize(gpu_tread_ctx_t* ctx);
void cuda_gpu_cleanup(gpu_tread_ctx_t* ctx);
void cuda_gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl));

/* Prefixed OpenCL symbols. */
void ocl_gpu_get_props(device_props_t* prop);
BOOL ocl_gpu_get_device_props(int device_ix, device_props_t* prop);
BOOL ocl_gpu_can_use_gpu(void);
int ocl_gpu_driver_version(void);
int ocl_gpu_runtime_version(void);
gpu_versions_t ocl_gpu_number_to_version(int version_number);
BOOL ocl_gpu_init_pipeline(gpu_tread_ctx_t* ctx);
void ocl_gpu_synchronize(gpu_tread_ctx_t* ctx);
void ocl_gpu_cleanup(gpu_tread_ctx_t* ctx);
void ocl_gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl));

#define HC_GPU_DECL_CUDA(name) HC_GPU_HASH_DECL_PFX(cuda_, name)
#define HC_GPU_DECL_OCL(name) HC_GPU_HASH_DECL_PFX(ocl_, name)
HC_GPU_HASHES(HC_GPU_DECL_CUDA)
HC_GPU_HASHES(HC_GPU_DECL_OCL)
#undef HC_GPU_DECL_CUDA
#undef HC_GPU_DECL_OCL

#ifdef __cplusplus
}
#endif
#endif /* HC_GPU_BACKENDS_H_ */
