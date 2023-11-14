#include <gputk.h>

#define NUM_BINS 4096
#define BLOCK_SIZE 1024

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void histogram(unsigned int *input, unsigned int *bins,
                          unsigned int num_elements)
{
  __shared__ unsigned int private_histo[NUM_BINS];
  int shared_idx = threadIdx.x;
  while (shared_idx < NUM_BINS)
  {
    private_histo[shared_idx] = 0;
    shared_idx += blockDim.x;
  }
  __syncthreads();

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x;
  while (thread_idx < num_elements)
  {
    atomicAdd(&(private_histo[input[thread_idx]]), 1);
    thread_idx += stride;
  }
  __syncthreads();

  int shared_idx2 = threadIdx.x;
  while (shared_idx2 < NUM_BINS)
  {
    atomicAdd(&(bins[shared_idx2]), private_histo[shared_idx2]);
    shared_idx2 += blockDim.x;
  }
  __syncthreads();

  shared_idx2 = threadIdx.x;
  while (shared_idx2 < NUM_BINS)
  {
    atomicMin(&(bins[shared_idx2]), 127);
    shared_idx2 += blockDim.x;
  }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)gpuTKImport(gpuTKArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);
  gpuTKLog(TRACE, "The number of bins is ", NUM_BINS);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  gpuTKLog(TRACE, "Launching kernel");
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(1, 1, 1);
  histogram<<<dimGrid, dimBlock>>>(deviceInput, deviceBins, inputLength);
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  gpuTKSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}
