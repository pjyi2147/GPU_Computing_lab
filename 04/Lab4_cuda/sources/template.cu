#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
__global__ void convolution(float* deviceInputImageData, float* deviceOutputImageData,
                            const float * __restrict__ deviceMaskData, int imageChannels, int imageWidth,
                            int imageHeight) {
  __shared__ float ds_Mask[Mask_width][Mask_width];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_o = blockIdx.y * TILE_WIDTH + ty;
  int col_o = blockIdx.x * TILE_WIDTH + tx;
  int row_i = row_o - Mask_radius;
  int col_i = col_o - Mask_radius;
  if ((row_i >= 0) && (row_i < imageHeight) && (col_i >= 0) &&
      (col_i < imageWidth)) {
    ds_Mask[ty][tx] = deviceMaskData[ty * Mask_width + tx];
  } else {
    ds_Mask[ty][tx] = 0.0;
  }
  __syncthreads();
  float output = 0.0;
  if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
    for (int i = 0; i < Mask_width; i++) {
      for (int j = 0; j < Mask_width; j++) {
        int row = row_i + i;
        int col = col_i + j;
        if (row >= 0 && row < imageHeight && col >= 0 && col < imageWidth) {
          output += deviceInputImageData[(row * imageWidth + col) * imageChannels +
                                         blockIdx.z] *
                    ds_Mask[i][j];
        }
      }
    }
    deviceOutputImageData[(row_o * imageWidth + col_o) * imageChannels +
                          blockIdx.z] = clamp(output);
  }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage   = gpuTKImport(inputImageFile);
  hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = gpuTKImage_getWidth(inputImage);
  imageHeight   = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);

  outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
              imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float),
              cudaMemcpyHostToDevice);
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimGrid(ceil(imageWidth / TILE_WIDTH), ceil(imageHeight / TILE_WIDTH),
               1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}
