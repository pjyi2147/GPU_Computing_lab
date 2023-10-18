#include <gputk.h>


__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len) {
    out[index] = in1[index] + in2[index];
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  unsigned int numStreams;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  gpuTKTime_start(GPU, "Allocating Pinned memory.");

  //@@ Allocate GPU memory here using pinned memory here

  //@@ Create and setup streams 
  //@@ Calculate data segment size of input data processed by each stream 

 
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Perform parallel vector addition with different streams. 
  for (unsigned int s = 0; s<numStreams; s++){
          //@@ Asynchronous copy data to the device memory in segments 
          //@@ Calculate starting and ending indices for per-stream data

          //@@ Invoke CUDA Kernel
          //@@ Determine grid and thread block sizes (consider ococupancy)     

          //@@ Asynchronous copy data from the device memory in segments 

  }

  //@@ Synchronize

  gpuTKTime_stop(Compute, "Performing CUDA computation");


  gpuTKTime_start(GPU, "Freeing Pinned Memory");
  //@@ Destory cudaStream


  //@@ Free the GPU memory here


  gpuTKTime_stop(GPU, "Freeing Pinned Memory");

  gpuTKSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
