
#include "gputk.h"
#include <algorithm>

static char *base_dir;

struct point {
  float x;
  float y;
};

struct pt_idx {
  unsigned idx;
  point p;
};

static int ccw(point p1, point p2, point p)
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

static float dist(point p1, point p2, point p)
{
  return abs((p.y - p1.y) * (p2.x - p1.x) - (p.x - p1.x) * (p2.y - p1.y));
}

static void find_hull(vector<pt_idx*>& pts, pt_idx* p1, pt_idx* p2, vector<unsigned>& indices)
{
  if (pts.size() == 0)
  {
    // printf("p1 = (%f, %f) idx = %u\n", p1->p.x, p1->p.y, p1->idx);
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

static int compute(vector<unsigned int>& indices, float *inputX_data, float *inputY_data, size_t input_length) {
  // convex hull algorithm on the CPU with inputX and inputY
  // output is sorted in indices array
  // return value is the number of points in the convex hull

  // sort the points by x coordinate
  std::vector<pt_idx*> pts(input_length);
  for (unsigned int i = 0; i < input_length; i++) {
    pts[i] = new pt_idx{i, {inputX_data[i], inputY_data[i]}};
    // printf("pts[%d] = (%f, %f)\n", i, pts[i]->p.x, pts[i]->p.y);
  }
  std::sort(pts.begin(), pts.end(), [](const pt_idx* a, const pt_idx* b) {
    if (a->p.x < b->p.x) {
      return true;
    } else if (a->p.x > b->p.x) {
      return false;
    } else {
      return a->p.y < b->p.y;
    }
  });

  // ret
  vector<unsigned int> v_indices;
  int ret = compute(pts, v_indices);
  // printf("\n");

  // copy the indices to the output array
  for (int i = 0; i < ret; i++) {
    indices.push_back(v_indices[i]);
  }
  return ret;
}

static float *generate_data(size_t n) {
  float *data = (float *)malloc(sizeof(float) * n);
  for (unsigned int i = 0; i < n; i++) {
    data[i] = -1.0f + (float)rand() / ((float)RAND_MAX / 2.0f);
  }
  return data;
}

static void write_data(char *file_name, vector<unsigned int>& data) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%lu", data.size());
  for (int ii = 0; ii < data.size(); ii++) {
    fprintf(handle, "\n%d", data[ii]);
  }
  fflush(handle);
  fclose(handle);
}

static void write_data(char *file_name, float *data, int num) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d", num);
  for (int ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%.6f", *data++);
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, size_t input_length) {

  const char *dir_name =
      gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

  char *inputX_file_name  = gpuTKPath_join(dir_name, "input1.raw");
  char *inputY_file_name  = gpuTKPath_join(dir_name, "input2.raw");
  char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

  float *inputX_data = generate_data(input_length);
  float *inputY_data = generate_data(input_length);
  vector<unsigned int> output_data;

  compute(output_data, inputX_data, inputY_data, input_length);
  write_data(inputX_file_name, inputX_data, input_length);
  write_data(inputY_file_name, inputY_data, input_length);
  write_data(output_file_name, output_data);

  free(inputX_data);
  free(inputY_data);
}

int main() {
  base_dir = gpuTKPath_join(gpuTKDirectory_current(), "ConvexHull", "Dataset");

  create_dataset(0, 16);
  create_dataset(1, 200);
  create_dataset(2, 100000);
  create_dataset(3, 1000000);
  create_dataset(4, 10000000);
  return 0;
}
