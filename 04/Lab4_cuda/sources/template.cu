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
#define Image_channels 3

//@@ INSERT CODE HERE
__global__ void convolution(
  float* deviceInputImageData, float* deviceOutputImageData, const float * __restrict__ deviceMaskData,
  int imageWidth, int imageHeight)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // this is the output tile
  int row_o = by * blockDim.y + ty;
  int col_o = bx * blockDim.x + tx;

  __shared__ float ds_image[w][w][Image_channels];
  memset(ds_image, 0.0, w * w * Image_channels * sizeof(float));

  if (row_o >= imageHeight || col_o >= imageWidth)
  {
    return;
  }

  // copy current tile first
  ds_image[ty + Mask_radius][tx + Mask_radius][tz] = deviceInputImageData[(row_o * imageWidth + col_o) * Image_channels + tz];

  // copy left
  if (tx == 0)
  {
    for (int i = -Mask_radius; i < 0; i++)
    {
      if (col_o + i >= 0)
      {
        ds_image[ty + Mask_radius][tx + Mask_radius + i][tz] = deviceInputImageData[(row_o * imageWidth + col_o + i) * Image_channels + tz];
      }
      else
      {
        ds_image[ty + Mask_radius][tx + Mask_radius + i][tz] = 0.0;
      }
    }
  }
  // copy right
  if (tx == TILE_WIDTH - 1)
  {
    for (int i = 1; i <= Mask_radius; i++)
    {
      if (col_o + i < imageWidth)
      {
        ds_image[ty + Mask_radius][tx + Mask_radius + i][tz] = deviceInputImageData[(row_o * imageWidth + col_o + i) * Image_channels + tz];
      }
      else
      {
        ds_image[ty + Mask_radius][tx + Mask_radius + i][tz] = 0.0;
      }
    }
  }

  // copy top
  if (ty == 0)
  {
    for (int i = -Mask_radius; i < 0; i++)
    {
      if (row_o + i >= 0)
      {
        ds_image[ty + Mask_radius + i][tx + Mask_radius][tz] = deviceInputImageData[((row_o + i) * imageWidth + col_o) * Image_channels + tz];
      }
      else
      {
        ds_image[ty + Mask_radius + i][tx + Mask_radius][tz] = 0.0;
      }
    }
  }
  // copy bottom
  if (ty == TILE_WIDTH - 1)
  {
    for (int i = 1; i <= Mask_radius; i++)
    {
      if (row_o + i < imageHeight)
      {
        ds_image[ty + Mask_radius + i][tx + Mask_radius][tz] = deviceInputImageData[((row_o + i) * imageWidth + col_o) * Image_channels + tz];
      }
      else
      {
        ds_image[ty + Mask_radius + i][tx + Mask_radius][tz] = 0.0;
      }
    }
  }
  __syncthreads();

  // copy top left
  if (ty == 0 && tx == 0)
  {
    for (int i = -Mask_radius; i < 0; i++)
    {
      for (int j = -Mask_radius; j < 0; j++)
      {
        if (row_o + i >= 0 && col_o + j >= 0)
        {
          ds_image[ty + Mask_radius + i][tx + Mask_radius + j][tz] = deviceInputImageData[((row_o + i) * imageWidth + col_o + j) * Image_channels + tz];
        }
        else
        {
          ds_image[ty + Mask_radius + i][tx + Mask_radius + j][tz] = 0.0;
        }
      }
    }
  }
  // copy top right
  if (ty == 0 && tx == TILE_WIDTH - 1)
  {
    for (int i = -Mask_radius; i < 0; i++)
    {
      for (int j = 1; j <= Mask_radius; j++)
      {
        if (row_o + i >= 0 && col_o + j < imageWidth)
        {
          ds_image[ty + Mask_radius + i][tx + Mask_radius + j][tz] = deviceInputImageData[((row_o + i) * imageWidth + col_o + j) * Image_channels + tz];
        }
        else
        {
          ds_image[ty + Mask_radius + i][tx + Mask_radius + j][tz] = 0.0;
        }
      }
    }
  }
  // copy bottom left
  if (ty == TILE_WIDTH - 1 && tx == 0)
  {
    for (int i = 1; i <= Mask_radius; i++)
    {
      for (int j = -Mask_radius; j < 0; j++)
      {
        if (row_o + i < imageHeight && col_o + j >= 0)
        {
          ds_image[ty + Mask_radius + i][tx + Mask_radius + j][tz] = deviceInputImageData[((row_o + i) * imageWidth + col_o + j) * Image_channels + tz];
        }
        else
        {
          ds_image[ty + Mask_radius + i][tx + Mask_radius + j][tz] = 0.0;
        }
      }
    }
  }
  // copy bottom right
  if (ty == TILE_WIDTH - 1 && tx == TILE_WIDTH - 1)
  {
    for (int i = 1; i <= Mask_radius; i++)
    {
      for (int j = 1; j <= Mask_radius; j++)
      {
        if (row_o + i < imageHeight && col_o + j < imageWidth)
        {
          ds_image[ty + Mask_radius + i][tx + Mask_radius + j][tz] = deviceInputImageData[((row_o + i) * imageWidth + col_o + j) * Image_channels + tz];
        }
        else
        {
          ds_image[ty + Mask_radius + i][tx + Mask_radius + j][tz] = 0.0;
        }
      }
    }
  }
  __syncthreads();

  float output = 0.0;
  for (int i = 0; i < Mask_width; i++)
  {
    for (int j = 0; j < Mask_width; j++)
    {
      int row = ty + i;
      int col = tx + j;
      output += ds_image[row][col][tz] * deviceMaskData[i * Mask_width + j];
    }
  }
  deviceOutputImageData[(row_o * imageWidth + col_o) * Image_channels + tz] = clamp(output);
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

  assert(imageChannels == Image_channels);

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
  int gridx = (imageWidth - 1) / TILE_WIDTH + 1;
  int gridy = (imageHeight - 1) / TILE_WIDTH + 1;
  dim3 dimGrid(gridx, gridy, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 3);
  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData,
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
