
#include "gputk.h"
#include <algorithm>

static char *base_dir;

struct pt_idx {
  unsigned idx;
  double x, y;

  bool operator==(const pt_idx& rhs) const
  {
    return x == rhs.x && y == rhs.y && idx == rhs.idx;
  }
};

static int ccw(pt_idx p1, pt_idx p2, pt_idx p)
{
  double prod = (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x);

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

static double dist(pt_idx p1, pt_idx p2, pt_idx p)
{
  return abs((p.y - p1.y) * (p2.x - p1.x) - (p.x - p1.x) * (p2.y - p1.y));
}

static void find_hull(vector<pt_idx>& pts, pt_idx p1, pt_idx p2, vector<unsigned>& indices)
{
  if (pts.size() == 0)
  {
    // printf("p1 = (%f, %f) idx = %u\n", p1->p.x, p1->p.y, p1->idx);
    indices.push_back(p1.idx);
    return;
  }

  auto p = pts[0];
  double max_dist = dist(p1, p2, p);
  for (int i = 1; i < pts.size(); i++)
  {
    double d = dist(p1, p2, pts[i]);
    if (d > max_dist)
    {
      p = pts[i];
      max_dist = d;
    }
  }

  vector<pt_idx> ac;
  vector<pt_idx> cb;
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

  find_hull(ac, p1, p, indices);
  find_hull(cb, p, p2, indices);
}

static int compute(vector<pt_idx>& pts, vector<unsigned>& indices)
{
  auto left = pts[0];
  auto right = pts[pts.size() - 1];

  vector<pt_idx> ccw_pts;
  vector<pt_idx> cw_pts;

  for (int i = 1; i < pts.size() - 1; i++)
  {
    int side = ccw(left, right, pts[i]);
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

static int compute(vector<unsigned int>& indices, double *inputX_data, double *inputY_data, size_t input_length) {
  // convex hull algorithm on the CPU with inputX and inputY
  // output is sorted in indices array
  // return value is the number of points in the convex hull

  // sort the points by x coordinate
  std::vector<pt_idx> pts(input_length);
  // printf("input_length = %lu\n", input_length);
  for (unsigned int i = 0; i < input_length; i++) {
    pts[i] = pt_idx{i, inputX_data[i], inputY_data[i]};
    // printf("pts[%d] = (%f, %f)\n", i, pts[i]->p.x, pts[i]->p.y);
  }
  std::sort(pts.begin(), pts.end(), [](const pt_idx& a, const pt_idx& b) {
    if (a.x < b.x) {
      return true;
    } else if (a.x > b.x) {
      return false;
    } else {
      return a.y < b.y;
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

static void generate_data(double *&X, double *&Y, size_t n) {
  X = (double *)malloc(sizeof(double) * n);
  Y = (double *)malloc(sizeof(double) * n);
  srand((unsigned)time(NULL));
  for (unsigned int i = 0; i < n; i++) {
    double radius = (double)rand() / RAND_MAX;
    double angle = (double)rand() / RAND_MAX * 2 * M_PI;

    X[i] = roundf(radius * cos(angle) * 1e6) / 1e6;
    Y[i] = roundf(radius * sin(angle) * 1e6) / 1e6;
  }
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

static void write_data(char *file_name, double *data, int num) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d", num);
  for (int ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%lf", *data++);
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

  double *inputX_data;
  double *inputY_data;
  generate_data(inputX_data, inputY_data, input_length);
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

  create_dataset(0, 5000000);
  create_dataset(1, 5000000);
  create_dataset(2, 5000000);
  create_dataset(3, 5000000);
  create_dataset(4, 5000000);
  create_dataset(5, 5000000);
  create_dataset(6, 5000000);
  return 0;
}
