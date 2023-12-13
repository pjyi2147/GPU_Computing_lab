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
    cudaMalloc(&d_p, sizeof(pt_idx) * pts.size());
    cudaMemcpy(d_p, pts.data(), sizeof(pt_idx) * pts.size(), cudaMemcpyHostToDevice);

    float * d_dist;
    cudaMalloc(&d_dist, sizeof(float) * pts.size());
    cudaMemset(d_dist, 0, sizeof(float) * pts.size());

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((pts.size() - 1) / BLOCK_SIZE + 1);
    dist<<<dimGrid, dimBlock>>>(d_p, d_dist, pts.size(), p1, p2);
    float *h_dist;
    h_dist = (float *)malloc(sizeof(float) * pts.size());
    cudaMemcpy(h_dist, d_dist, sizeof(float) * pts.size(), cudaMemcpyDeviceToHost);

    float max_dist = h_dist[0];
    for (int i = 1; i < pts.size(); i++)
    {
      if (h_dist[i] > max_dist)
      {
        max_dist = h_dist[i];
        p = pts[i];
      }
    }

    int * d_ccw_ac;
    cudaMalloc(&d_ccw_ac, sizeof(int) * pts.size());
    cudaMemset(d_ccw_ac, 0, sizeof(int) * pts.size());

    ccw<<<dimGrid, dimBlock>>>(d_p, d_ccw_ac, pts.size(), p1, p);
    int * h_ccw_ac = (int *)malloc(sizeof(int) * pts.size());
    cudaMemcpy(h_ccw_ac, d_ccw_ac, sizeof(int) * pts.size(), cudaMemcpyDeviceToHost);

    int * d_ccw_cb;
    cudaMalloc(&d_ccw_cb, sizeof(int) * pts.size());
    cudaMemset(d_ccw_cb, 0, sizeof(int) * pts.size());

    ccw<<<dimGrid, dimBlock>>>(d_p, d_ccw_cb, pts.size(), p, p2);
    int * h_ccw_cb = (int *)malloc(sizeof(int) * pts.size());
    cudaMemcpy(h_ccw_cb, d_ccw_cb, sizeof(int) * pts.size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < pts.size(); i++)
    {
      if (pts[i] == p)
      {
        continue;
      }

      if (h_ccw_ac[i] == 1)
      {
        ac.push_back(pts[i]);
      }

      if (h_ccw_cb[i] == 1)
      {
        cb.push_back(pts[i]);
      }
    }

    cudaFree(d_p);
    cudaFree(d_dist);
    cudaFree(d_ccw_ac);
    cudaFree(d_ccw_cb);
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
    pt_idx* d_p;
    cudaMalloc(&d_p, sizeof(pt_idx) * pts.size());
    cudaMemcpy(d_p, pts.data(), sizeof(pt_idx) * pts.size(), cudaMemcpyHostToDevice);

    int * d_ccw;
    cudaMalloc(&d_ccw, sizeof(int) * pts.size());
    cudaMemset(d_ccw, 0, sizeof(int) * pts.size());

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((pts.size() - 1) / BLOCK_SIZE + 1);
    ccw<<<dimGrid, dimBlock>>>(d_p, d_ccw, pts.size(), left, right);

    int * h_ccw;
    h_ccw = (int *)malloc(sizeof(int) * pts.size());
    cudaMemcpy(h_ccw, d_ccw, sizeof(int) * pts.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_p);
    cudaFree(d_ccw);

    for (int i = 1; i < pts.size() - 1; i++)
    {
      int side = h_ccw[i];
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
