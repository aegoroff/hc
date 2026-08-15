#ifndef HC_GPU_PREFIX_H_
#define HC_GPU_PREFIX_H_

/*
 * Dual-backend symbol rename. Compile with -DHC_GPU_NS_CUDA or -DHC_GPU_NS_OCL
 * and -include this header. Hash ABI names are wrapped in source with HC_GPU_FN;
 * gpu_* runtime names need object-like #defines so calls (e.g. gpu_run in .cu)
 * rename too. Do not funnel those through a PRE() helper: the argument expands
 * before pasting and the macro recurses.
 */
#if defined(HC_GPU_NS_CUDA)
#define gpu_get_props cuda_gpu_get_props
#define gpu_get_device_props cuda_gpu_get_device_props
#define gpu_can_use_gpu cuda_gpu_can_use_gpu
#define gpu_is_opencl cuda_gpu_is_opencl
#define gpu_driver_version cuda_gpu_driver_version
#define gpu_runtime_version cuda_gpu_runtime_version
#define gpu_number_to_version cuda_gpu_number_to_version
#define gpu_init_pipeline cuda_gpu_init_pipeline
#define gpu_synchronize cuda_gpu_synchronize
#define gpu_cleanup cuda_gpu_cleanup
#define gpu_run cuda_gpu_run
#define HC_GPU_FN(name) cuda_##name
#elif defined(HC_GPU_NS_OCL)
#define gpu_get_props ocl_gpu_get_props
#define gpu_get_device_props ocl_gpu_get_device_props
#define gpu_can_use_gpu ocl_gpu_can_use_gpu
#define gpu_is_opencl ocl_gpu_is_opencl
#define gpu_driver_version ocl_gpu_driver_version
#define gpu_runtime_version ocl_gpu_runtime_version
#define gpu_number_to_version ocl_gpu_number_to_version
#define gpu_init_pipeline ocl_gpu_init_pipeline
#define gpu_synchronize ocl_gpu_synchronize
#define gpu_cleanup ocl_gpu_cleanup
#define gpu_run ocl_gpu_run
#define HC_GPU_FN(name) ocl_##name
#else
#error "compile with -DHC_GPU_NS_CUDA or -DHC_GPU_NS_OCL"
#endif

#endif /* HC_GPU_PREFIX_H_ */
