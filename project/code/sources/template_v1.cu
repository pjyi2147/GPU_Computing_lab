#include <gputk.h>
#include <algorithm>

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

int ccw(point p1, point p2, point p)
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

float dist(point p1, point p2, point p)
{
  return abs((p.y - p1.y) * (p2.x - p1.x) - (p.x - p1.x) * (p2.y - p1.y));
}

__global__ void dist(pt_idx* p, float* d, int len, point* p1, point* p2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len)
  {
    point p3 = p[idx].p;
    d[idx] = abs((p3.y - p1->y) * (p2->x - p1->x) - (p3.x - p1->x) * (p2->y - p1->y));
  }
}

__global__ void ccw(pt_idx* p, int* d, int len, point* p1, point* p2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len)
  {
    point p3 = p[idx].p;
    float prod = (p2->x - p1->x) * (p3.y - p1->y) - (p2->y - p1->y) * (p3.x - p1->x);
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

static void find_hull(vector<pt_idx*>& pts, pt_idx* p1, pt_idx* p2, vector<unsigned>& indices)
{
  if (pts.size() == 0)
  {
    // printf("p1 = (%f, %f) idx = %u\n", p1->p.x, p1->p.y, p1->idx);
    indices.push_back(p1->idx);
    return;
  }

  pt_idx* d_p;
  cudaMalloc(&d_p, sizeof(pt_idx) * pts.size());
  for (int i = 0; i < pts.size(); i++)
  {
    cudaMemcpy(&d_p[i], pts[i], sizeof(pt_idx), cudaMemcpyHostToDevice);
  }
  // cudaMemcpy(d_p, pts.data(), sizeof(pt_idx) * pts.size(), cudaMemcpyHostToDevice);

  float * d_dist;
  cudaMalloc(&d_dist, sizeof(float) * pts.size());
  cudaMemset(d_dist, 0, sizeof(float) * pts.size());

  point* d_p1;
  point* d_p2;
  cudaMalloc(&d_p1, sizeof(point));
  cudaMalloc(&d_p2, sizeof(point));
  cudaMemcpy(d_p1, &p1->p, sizeof(point), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2, &p2->p, sizeof(point), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((pts.size() - 1) / BLOCK_SIZE + 1);
  dist<<<dimGrid, dimBlock>>>(d_p, d_dist, pts.size(), d_p1, d_p2);
  float * h_dist;
  h_dist = (float *)malloc(sizeof(float) * pts.size());
  cudaMemcpy(h_dist, d_dist, sizeof(float) * pts.size(), cudaMemcpyDeviceToHost);

  float max_dist = h_dist[0];
  auto p = pts[0];
  for (int i = 1; i < pts.size(); i++)
  {
    if (h_dist[i] > max_dist)
    {
      max_dist = h_dist[i];
      p = pts[i];
    }
  }

  vector<pt_idx*> ac;
  vector<pt_idx*> cb;

  int * d_ccw_ac;
  cudaMalloc(&d_ccw_ac, sizeof(int) * pts.size());
  cudaMemset(d_ccw_ac, 0, sizeof(int) * pts.size());

  point * d_p3;
  cudaMalloc(&d_p3, sizeof(point));
  cudaMemcpy(d_p3, &p->p, sizeof(point), cudaMemcpyHostToDevice);

  ccw<<<dimGrid, dimBlock>>>(d_p, d_ccw_ac, pts.size(), d_p1, d_p3);
  int * h_ccw_ac = (int *)malloc(sizeof(int) * pts.size());
  cudaMemcpy(h_ccw_ac, d_ccw_ac, sizeof(int) * pts.size(), cudaMemcpyDeviceToHost);


  int * d_ccw_cb;
  cudaMalloc(&d_ccw_cb, sizeof(int) * pts.size());
  cudaMemset(d_ccw_cb, 0, sizeof(int) * pts.size());

  ccw<<<dimGrid, dimBlock>>>(d_p, d_ccw_cb, pts.size(), d_p3, d_p2);
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
  cudaFree(d_p1);
  cudaFree(d_p2);
  cudaFree(d_p3);

  find_hull(ac, p1, p, indices);
  find_hull(cb, p, p2, indices);
}

static int compute(vector<pt_idx*>& pts, vector<unsigned>& indices)
{
  auto left = pts[0];
  auto right = pts[pts.size() - 1];

  vector<pt_idx*> ccw_pts;
  vector<pt_idx*> cw_pts;

  pt_idx* d_p;
  cudaMalloc(&d_p, sizeof(pt_idx) * pts.size());
  for (int i = 0; i < pts.size(); i++)
  {
    cudaMemcpy(&d_p[i], pts[i], sizeof(pt_idx), cudaMemcpyHostToDevice);
  }

  int * d_ccw;
  cudaMalloc(&d_ccw, sizeof(int) * pts.size());
  cudaMemset(d_ccw, 0, sizeof(int) * pts.size());

  point * d_p1;
  cudaMalloc(&d_p1, sizeof(point));
  cudaMemcpy(d_p1, &left->p, sizeof(point), cudaMemcpyHostToDevice);

  point * d_p2;
  cudaMalloc(&d_p2, sizeof(point));
  cudaMemcpy(d_p2, &right->p, sizeof(point), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((pts.size() - 1) / BLOCK_SIZE + 1);
  ccw<<<dimGrid, dimBlock>>>(d_p, d_ccw, pts.size(), d_p1, d_p2);

  int * h_ccw;
  h_ccw = (int *)malloc(sizeof(int) * pts.size());
  cudaMemcpy(h_ccw, d_ccw, sizeof(int) * pts.size(), cudaMemcpyDeviceToHost);

  cudaFree(d_p);
  cudaFree(d_ccw);
  cudaFree(d_p1);
  cudaFree(d_p2);

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

  find_hull(ccw_pts, left, right, indices);
  find_hull(cw_pts, right, left, indices);

  return indices.size();
}

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  int inputLength;
  float *hostX;
  float *hostY;
  vector<pt_idx *> hostPts;
  vector<unsigned> hostAnswer;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostX = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostY = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  hostPts = vector<pt_idx *>(inputLength);
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  gpuTKTime_start(Generic, "Total Computation");
  gpuTKTime_start(Generic, "Create data");
  for (unsigned i = 0; i < inputLength; i++)
  {
    hostPts[i] = new pt_idx{i, {hostX[i], hostY[i]}};
  }
  std::sort(hostPts.begin(), hostPts.end(), [](const pt_idx* a, const pt_idx* b) {
    if (a->p.x < b->p.x) {
      return true;
    } else if (a->p.x > b->p.x) {
      return false;
    } else {
      return a->p.y < b->p.y;
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
  for (unsigned i = 0; i < inputLength; i++)
  {
    delete hostPts[i];
  }
  free(hostX);
  free(hostY);
  return 0;
}
