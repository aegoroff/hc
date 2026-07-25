/* Marker TU so the hc-gpu static lib is non-empty when CUDA is enabled.
 * Real GPU symbols come from zig-out/cuda/libhc-cuda.a (linked separately).
 */
int hc_cuda_archive_linked = 1;
