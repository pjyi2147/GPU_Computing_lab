// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <gputk.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// __global__ void scan_add(float *output, float *auxOutput, int lenOutput, int lenAuxOutput)
// {
//   if (blockIdx.x > 0 && blockIdx.x < lenAuxOutput - 1)
//   {
//     int idx = SECTION_SIZE * blockIdx.x + threadIdx.x;
//     output[idx] += auxOutput[blockIdx.x - 1];
//     output[idx + 1] += auxOutput[blockIdx.x - 1];
//   }
// }

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here

  __shared__ float T[BLOCK_SIZE];
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < len)
  {
    T[threadIdx.x] = input[idx];
  }
  else
  {
    T[threadIdx.x] = 0;
  }
  __syncthreads();

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
  {
    __syncthreads();

    int index = (threadIdx.x + 1) * 2 * stride - 1;

    if (index < blockDim.x)
    {
      T[index] += T[index - stride];
    }
  }

  // postscan
  for (unsigned int stride = blockDim.x / 4; stride > 0; stride /= 2)
  {
    __syncthreads();

    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index + stride < blockDim.x)
    {
      T[index + stride] += T[index];
    }
  }

  // put back to output array
  if (idx < len)
  {
    output[idx] = T[threadIdx.x];
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The number of input elements in the input is ",
        numElements);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  gpuTKCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Clearing output memory.");
  gpuTKCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  gpuTKTime_stop(GPU, "Clearing output memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  gpuTKCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(BLOCK_SIZE, 1, 1);
  dim3 gridDim((numElements - 1) / BLOCK_SIZE + 1, 1, 1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements);

  if (numElements > BLOCK_SIZE)
  {
    // int numAuxArray = numElements / SECTION_SIZE;
    // float *deviceAuxInput, *deviceAuxOutput;
    // cudaMalloc(&deviceAuxInput, numAuxArray * sizeof(float));
    // cudaMalloc(&deviceAuxOutput, numAuxArray * sizeof(float));
    // for (int i = 1; i < numAuxArray; i++)
    // {
    //   deviceAuxInput[i] = deviceOutput[i * SECTION_SIZE - 1];
    // }
    // dim3 gridDim2((numAuxArray - 1) / SECTION_SIZE + 1, 1, 1);
    // scan<<<gridDim2, blockDim>>>(deviceAuxInput, deviceAuxOutput, numAuxArray);

    // // add scanned block sum i
    // scan_add<<<gridDim, blockDim>>>(deviceOutput, deviceAuxOutput, numElements, numAuxArray);
  }

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  gpuTKCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
