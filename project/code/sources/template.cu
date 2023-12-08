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

static void find_hull(vector<pt_idx*>& pts, pt_idx* p1, pt_idx* p2, vector<unsigned>& indices)
{
  if (pts.size() == 0)
  {
    printf("p1 = (%f, %f) idx = %u\n", p1->p.x, p1->p.y, p1->idx);
    indices.push_back(p1->idx);
    return;
  }

  auto p = pts[0];
  float max_dist = dist(p1->p, p2->p, p->p);
  for (int i = 1; i < pts.size(); i++)
  {
    float d = dist(p1->p, p2->p, pts[i]->p);
    if (d > max_dist)
    {
      p = pts[i];
      max_dist = d;
    }
  }

  vector<pt_idx*> ac;
  vector<pt_idx*> cb;
  for (int i = 0; i < pts.size(); i++)
  {
    if (pts[i] == p)
    {
      continue;
    }

    int side = ccw(p1->p, p->p, pts[i]->p);
    if (side == 1)
    {
      ac.push_back(pts[i]);
    }

    side = ccw(p->p, p2->p, pts[i]->p);
    if (side == 1)
    {
      cb.push_back(pts[i]);
    }
  }

  find_hull(ac, p1, p, indices);
  find_hull(cb, p, p2, indices);
}

static int compute(vector<pt_idx*>& pts, vector<unsigned>& indices)
{
  auto left = pts[0];
  auto right = pts[pts.size() - 1];

  vector<pt_idx*> ccw_pts;
  vector<pt_idx*> cw_pts;

  for (int i = 1; i < pts.size() - 1; i++)
  {
    int side = ccw(left->p, right->p, pts[i]->p);
    if (side == 1)
    {
      ccw_pts.push_back(pts[i]);
    }
    else if (side == -1)
    {
      cw_pts.push_back(pts[i]);
    }
  }

  find_hull(ccw_pts, left, right, indices);
  find_hull(cw_pts, right, left, indices);

  for (int i = 0; i < indices.size(); i++)
  {
    // printf("indices[%d] = %u\n");
  }
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

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  gpuTKLog(TRACE, "Launching kernel");
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here
  compute(hostPts, hostAnswer);
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

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
