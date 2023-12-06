#include <gputk.h>
#include <utility>
#include <algorithm>

#define BLOCK_SIZE 512

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Insert code to implement SPMV using JDS with transposed input here
struct JDS_T
{
  float *data;
  int data_size;
  int *jds_col_idx;
  int jds_col_idx_size;
  int *jds_row_idx;
  int jds_row_idx_size;
  int *jds_row_ptr;
  int jds_row_ptr_size;
  int *jds_t_col_ptr;
  int jds_t_col_ptr_size;
};

__global__ void spmv_jds_transposed(JDS_T A, float *B, float *C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns)
{
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float temp[BLOCK_SIZE];
  temp[threadIdx.x] = 0;
  for (int i = 0; i < A.jds_t_col_ptr_size - 1; i++)
  {
    int col_idx = A.jds_t_col_ptr[i];
    int num_threads = A.jds_t_col_ptr[i + 1] - A.jds_t_col_ptr[i];
    if (row_idx < num_threads)
    {
      float data = A.data[col_idx + row_idx];
      int data_col = A.jds_col_idx[col_idx + row_idx];
      temp[threadIdx.x] += data * B[data_col];
    }
  }
  if (row_idx < numARows)
  {
    C[A.jds_row_idx[row_idx]] = temp[threadIdx.x];
  }
}


int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  JDS_T deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(Generic, "Converting matrix A to JDS format (transposed).");
  vector<std::pair<int, int>> mat_count;
  int total_non_zero = 0;
  for (int i = 0; i < numARows; i++)
  {
    int count = 0;
    for (int j = 0; j < numAColumns; j++)
    {
      if (hostA[i * numAColumns + j] != 0)
      {
        count++;
        total_non_zero++;
      }
    }
    mat_count.push_back(std::make_pair(count, i));
  }

  std::sort(mat_count.begin(), mat_count.end(), std::greater<std::pair<int, int>>());

  vector<float> data;
  vector<int> jds_col_idx;
  vector<int> jds_row_idx;
  vector<int> jds_row_ptr;

  for (auto& i : mat_count)
  {
    jds_row_ptr.push_back(data.size());
    jds_row_idx.push_back(i.second);
    for (int j = 0; j < numAColumns; j++)
    {
      if (hostA[i.second * numAColumns + j] != 0)
      {
        data.push_back(hostA[i.second * numAColumns + j]);
        jds_col_idx.push_back(j);
      }
    }
  }
  jds_row_ptr.push_back(data.size());

  vector<float> data2;
  vector<int> jds_col_idx2;
  vector<int> jds_col_ptr;
  // transpose??
  int tc = 0;
  // this will be the block size
  int max_count = 0;
  // this is the number of rows (transposed)
  for (int i = 0; i < mat_count[0].first; i++)
  {
    // find how many pairs have the first value bigger than i
    jds_col_ptr.push_back(tc);

    // count is the number of columns for this row i
    int count = 0;
    for (int j = 0; j < mat_count.size(); j++)
    {
      if (mat_count[j].first > i)
      {
        count++;
      }
      else
      {
        // it is sorted, so we can break
        break;
      }
    }
    if (count > max_count)
    {
      max_count = count;
    }
    tc += count;

    // retrieve the elements from data
    for (int k = 0; k < count; k++)
    {
      data2.push_back(data[jds_row_ptr[k] + i]);
      jds_col_idx2.push_back(jds_col_idx[jds_row_ptr[k] + i]);
    }
  }
  jds_col_ptr.push_back(tc);

  JDS_T jds;
  jds.data = data2.data();
  jds.data_size = data2.size();
  jds.jds_col_idx = jds_col_idx2.data();
  jds.jds_col_idx_size = jds_col_idx2.size();
  jds.jds_row_idx = jds_row_idx.data();
  jds.jds_row_idx_size = jds_row_idx.size();
  jds.jds_row_ptr = jds_row_ptr.data();
  jds.jds_row_ptr_size = jds_row_ptr.size();
  jds.jds_t_col_ptr = jds_col_ptr.data();
  jds.jds_t_col_ptr_size = jds_col_ptr.size();
  gpuTKTime_stop(Generic, "Converting matrix A to JDS format (transposed).");

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA.data, jds.data_size * sizeof(float));
  cudaMalloc((void**)&deviceA.jds_col_idx, jds.jds_col_idx_size * sizeof(int));
  cudaMalloc((void**)&deviceA.jds_row_idx, jds.jds_row_idx_size * sizeof(int));
  cudaMalloc((void**)&deviceA.jds_row_ptr, jds.jds_row_ptr_size * sizeof(int));
  cudaMalloc((void**)&deviceA.jds_t_col_ptr, jds.jds_t_col_ptr_size * sizeof(int));

  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA.data, jds.data, jds.data_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceA.jds_col_idx, jds.jds_col_idx, jds.jds_col_idx_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceA.jds_row_idx, jds.jds_row_idx, jds.jds_row_idx_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceA.jds_row_ptr, jds.jds_row_ptr, jds.jds_row_ptr_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceA.jds_t_col_ptr, jds.jds_t_col_ptr, jds.jds_t_col_ptr_size * sizeof(int), cudaMemcpyHostToDevice);
  deviceA.data_size = jds.data_size;
  deviceA.jds_col_idx_size = jds.jds_col_idx_size;
  deviceA.jds_row_idx_size = jds.jds_row_idx_size;
  deviceA.jds_row_ptr_size = jds.jds_row_ptr_size;
  deviceA.jds_t_col_ptr_size = jds.jds_t_col_ptr_size;

  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemset(deviceC, 0, numCRows * numCColumns * sizeof(float));
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((max_count - 1) / BLOCK_SIZE + 1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  spmv_jds_transposed<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
                                             numARows, numAColumns,
                                             numBRows, numBColumns);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");


  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceB);
  cudaFree(deviceC);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
