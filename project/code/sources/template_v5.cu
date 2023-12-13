#include <gputk.h>
#include <algorithm>

#define BLOCK_SIZE 1024
#define THRESHOLD (BLOCK_SIZE * 32)

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

struct pt_idx {
  unsigned idx;
  float x, y;

  bool operator==(const pt_idx& rhs) const
  {
    return x == rhs.x && y == rhs.y && idx == rhs.idx;
  }
};

int ccw(pt_idx p1, pt_idx p2, pt_idx p)
{
  float prod = (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);
  if (prod > 0)
  {
    return 1;
  }
  else if (prod < 0)
  {
    return -1;
  }
  else
  {
    return 0;
  }
}

float dist(pt_idx p1, pt_idx p2, pt_idx p)
{
  return abs((p.y - p1.y) * (p2.x - p1.x) - (p.x - p1.x) * (p2.y - p1.y));
}

__global__ void dist(pt_idx* p, float* d, int len, pt_idx p1, pt_idx p2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len)
  {
    pt_idx p3 = p[idx];
    d[idx] = abs((p3.y - p1.y) * (p2.x - p1.x) - (p3.x - p1.x) * (p2.y - p1.y));
  }
}

__global__ void max_dist(pt_idx* from_p, int from_len, pt_idx* to_p, int to_len, pt_idx p1, pt_idx p2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float s_dist[BLOCK_SIZE];
  __shared__ pt_idx s_p[BLOCK_SIZE];
  s_dist[threadIdx.x] = 0;
  __syncthreads();

  if (idx < from_len)
  {
    s_p[threadIdx.x] = from_p[idx];
    pt_idx p3 = s_p[threadIdx.x];
    s_dist[threadIdx.x] = abs((p3.y - p1.y) * (p2.x - p1.x) - (p3.x - p1.x) * (p2.y - p1.y));
  }
  __syncthreads();

  // find max dist
  int s = blockDim.x / 2;
  while (s != 0)
  {
    if (threadIdx.x < s)
    {
      if (s_dist[threadIdx.x] < s_dist[threadIdx.x + s])
      {
        s_dist[threadIdx.x] = s_dist[threadIdx.x + s];
        s_p[threadIdx.x] = s_p[threadIdx.x + s];
      }
    }
    __syncthreads();
    s /= 2;
  }

  if (threadIdx.x == 0)
  {
    to_p[blockIdx.x] = s_p[0];
  }
}

__global__ void ccw(pt_idx* p, int* d, int len, pt_idx p1, pt_idx p2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len)
  {
    pt_idx p3 = p[idx];
    float prod = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    // printf("p1 = (%f, %f), p2 = (%f, %f), p = (%f, %f), prod = %f\n", p1->x, p1->y, p2->x, p2->y, p3.x, p3.y, prod);
    if (prod > 0)
    {
      d[idx] = 1;
    }
    else if (prod < 0)
    {
      d[idx] = -1;
    }
    else
    {
      d[idx] = 0;
    }
  }
}

static void find_hull(vector<pt_idx>& pts, pt_idx p1, pt_idx p2, vector<unsigned>& indices)
{
  if (pts.size() == 0)
  {
    // printf("p1 = (%f, %f) idx = %u\n", p1->p.x, p1->p.y, p1->idx);
    indices.push_back(p1.idx);
    return;
  }

  vector<pt_idx> ac;
  vector<pt_idx> cb;
  auto p = pts[0];
  if (pts.size() > THRESHOLD)
  {
    pt_idx* d_p;
    pt_idx* d_p1;
    pt_idx* d_p2;
    cudaMallocManaged(&d_p, sizeof(pt_idx) * pts.size());
    cudaMallocManaged(&d_p1, sizeof(pt_idx) * pts.size());
    cudaMallocManaged(&d_p2, sizeof(pt_idx) * pts.size());

    memcpy(d_p, pts.data(), sizeof(pt_idx) * pts.size());
    memcpy(d_p1, pts.data(), sizeof(pt_idx) * pts.size());
    memset(d_p2, 0, sizeof(pt_idx) * pts.size());

    int d_size = pts.size();
    while (d_size > 128)
    {
      //printf("d_size = %d\n", d_size);
      dim3 dimBlock(BLOCK_SIZE);
      int next_d_size = (d_size - 1) / BLOCK_SIZE + 1;
      dim3 dimGrid(next_d_size);
      max_dist<<<dimGrid, dimBlock>>>(d_p1, d_size, d_p2, next_d_size, p1, p2);
      cudaDeviceSynchronize();
      pt_idx* temp = d_p1;
      d_p1 = d_p2;
      d_p2 = temp;
      d_size = next_d_size;
    }

    if (d_size > 1)
    {
      p = d_p1[0];
      float max_dist = dist(p1, p2, p);
      for (int i = 1; i < d_size; i++)
      {
        float d = dist(p1, p2, d_p1[i]);
        if (d > max_dist)
        {
          p = d_p1[i];
          max_dist = d;
        }
      }
    }
    else
    {
      p = d_p1[0];
    }

    int * ccw_ac;
    cudaMallocManaged(&ccw_ac, sizeof(int) * pts.size());

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((pts.size() - 1) / BLOCK_SIZE + 1);
    ccw<<<dimGrid, dimBlock>>>(d_p, ccw_ac, pts.size(), p1, p);

    int * ccw_cb;
    cudaMallocManaged(&ccw_cb, sizeof(int) * pts.size());

    ccw<<<dimGrid, dimBlock>>>(d_p, ccw_cb, pts.size(), p, p2);
    cudaDeviceSynchronize();

    for (int i = 0; i < pts.size(); i++)
    {
      if (pts[i] == p)
      {
        continue;
      }

      if (ccw_ac[i] == 1)
      {
        ac.push_back(pts[i]);
      }

      if (ccw_cb[i] == 1)
      {
        cb.push_back(pts[i]);
      }
    }

    cudaFree(d_p);
    cudaFree(d_p1);
    cudaFree(d_p2);
    cudaFree(ccw_ac);
    cudaFree(ccw_cb);
  }
  else
  {
    float max_dist = dist(p1, p2, p);
    for (int i = 1; i < pts.size(); i++)
    {
      float d = dist(p1, p2, pts[i]);
      if (d > max_dist)
      {
        p = pts[i];
        max_dist = d;
      }
    }

    for (int i = 0; i < pts.size(); i++)
    {
      if (pts[i] == p)
      {
        continue;
      }

      int side = ccw(p1, p, pts[i]);
      if (side == 1)
      {
        ac.push_back(pts[i]);
      }

      side = ccw(p, p2, pts[i]);
      if (side == 1)
      {
        cb.push_back(pts[i]);
      }
    }
  }

  find_hull(ac, p1, p, indices);
  find_hull(cb, p, p2, indices);
}

static int compute(vector<pt_idx>& pts, vector<unsigned>& indices)
{
  auto left = pts[0];
  auto right = pts[pts.size() - 1];

  vector<pt_idx> ccw_pts;
  vector<pt_idx> cw_pts;

  if (pts.size() > THRESHOLD)
  {
    pt_idx* p_uni;
    cudaMallocManaged(&p_uni, sizeof(pt_idx) * pts.size());
    memcpy(p_uni, pts.data(), sizeof(pt_idx) * pts.size());

    int * ccw_uni;
    cudaMallocManaged(&ccw_uni, sizeof(int) * pts.size());
    memcpy(ccw_uni, pts.data(), sizeof(int) * pts.size());

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((pts.size() - 1) / BLOCK_SIZE + 1);
    ccw<<<dimGrid, dimBlock>>>(p_uni, ccw_uni, pts.size(), left, right);
    cudaDeviceSynchronize();
    for (int i = 1; i < pts.size() - 1; i++)
    {
      int side = ccw_uni[i];
      if (side == 1)
      {
        // printf("pts[%d] = (%f, %f) idx = %u, side = 1\n", i, pts[i]->p.x, pts[i]->p.y, pts[i]->idx);
        ccw_pts.push_back(pts[i]);
      }
      else if (side == -1)
      {
        // printf("pts[%d] = (%f, %f) idx = %u, side = -1\n", i, pts[i]->p.x, pts[i]->p.y, pts[i]->idx);
        cw_pts.push_back(pts[i]);
      }
    }
    cudaFree(p_uni);
    cudaFree(ccw_uni);
  }
  else
  {
    for (int i = 1; i < pts.size() - 1; i++)
    {
      int side = ccw(left, right, pts[i]);
      if (side == 1)
      {
        // printf("pts[%d] = (%f, %f) idx = %u, side = 1\n", i, pts[i]->p.x, pts[i]->p.y, pts[i]->idx);
        ccw_pts.push_back(pts[i]);
      }
      else if (side == -1)
      {
        // printf("pts[%d] = (%f, %f) idx = %u, side = -1\n", i, pts[i]->p.x, pts[i]->p.y, pts[i]->idx);
        cw_pts.push_back(pts[i]);
      }
    }
  }

  find_hull(ccw_pts, left, right, indices);
  find_hull(cw_pts, right, left, indices);

  // for (int i = 0; i < indices.size(); i++)
  // {
  //   printf("indices[%d] = %u\n");
  // }
  return indices.size();
}

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  int inputLength;
  float *hostX;
  float *hostY;
  vector<pt_idx> hostPts;
  vector<unsigned> hostAnswer;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostX = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostY = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  hostPts = vector<pt_idx>(inputLength);
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  gpuTKTime_start(Generic, "Total Computation");
  gpuTKTime_start(Generic, "Create data");
  for (unsigned i = 0; i < inputLength; i++)
  {
    hostPts[i] = pt_idx{i, hostX[i], hostY[i]};
  }
  std::sort(hostPts.begin(), hostPts.end(), [](const pt_idx& a, const pt_idx& b) {
    if (a.x < b.x) {
      return true;
    } else if (a.x > b.x) {
      return false;
    } else {
      return a.y < b.y;
    }
  });
  gpuTKTime_stop(Generic, "Create data");

  // Launch kernel
  // ----------------------------------------------------------
  gpuTKLog(TRACE, "Launching kernel");
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here
  compute(hostPts, hostAnswer);
  gpuTKTime_stop(Compute, "Performing CUDA computation");
  gpuTKTime_stop(Generic, "Total Computation");

  // Verify correctness
  // -----------------------------------------------------
  gpuTKSolution(args, hostAnswer.data(), hostAnswer.size());

  // Free memory
  free(hostX);
  free(hostY);
  return 0;
}
