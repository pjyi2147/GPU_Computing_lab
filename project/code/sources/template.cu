#include <gputk.h>

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

struct point {
  float x;
  float y;
};

struct pt_idx {
  unsigned idx;
  point p;
};

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  int inputLength;
  float *hostX;
  float *hostY;
  pt_idx *hostPts;
  pt_idx *deviceInput;
  unsigned int *deviceAnswer;
  unsigned int *hostAnswer;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostX = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostY = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  hostPts = (pt_idx *)malloc(inputLength * sizeof(pt_idx));
  hostAnswer = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  gpuTKTime_start(Generic, "Create data");
  for (int i = 0; i < inputLength; i++)
  {
    hostPts[i].idx = i;
    hostPts[i].p.x = hostX[i];
    hostPts[i].p.y = hostY[i];
  }
  gpuTKTime_stop(Generic, "Create data");

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(pt_idx));
  cudaMalloc((void **)&deviceAnswer, inputLength * sizeof(unsigned int));
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostPts, inputLength * sizeof(pt_idx),
             cudaMemcpyHostToDevice);
  cudaMemset(deviceAnswer, 0, inputLength * sizeof(unsigned int));
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  gpuTKLog(TRACE, "Launching kernel");
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here

  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostAnswer, deviceAnswer, inputLength * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceAnswer);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  gpuTKSolution(args, hostAnswer, inputLength);

  free(hostPts);
  free(hostAnswer);
  return 0;
}
