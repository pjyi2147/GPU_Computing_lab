
#include "gputk.h"

static char *base_dir;

static void compute(int *output, int *input, int num) {
  int accum = 0;
  int ii;
  for (ii = 0; ii < num; ++ii) {
    accum += input[ii];
    output[ii] = accum;
  }
}

static int *generate_data(int n) {
  int *data = (int *)malloc(sizeof(int) * n);
  int i;
  for (i = 0; i < n; i++) {
    data[i] = rand() % 4;
  }
  return data;
}

static void write_data(char *file_name, int *data, int num) {
  int ii;
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d", num);
  for (ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%d", *data++);
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, int dim) {
  const char *dir_name =
      gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

  char *input_file_name  = gpuTKPath_join(dir_name, "input.raw");
  char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

  int *input_data  = generate_data(dim);
  int *output_data = (int *)calloc(sizeof(int), dim);

  compute(output_data, input_data, dim);

  write_data(input_file_name, input_data, dim);
  write_data(output_file_name, output_data, dim);

  free(input_data);
  free(output_data);
}

int main() {
  base_dir = gpuTKPath_join(gpuTKDirectory_current(), "ListScan", "Dataset");
  create_dataset(0, 64);
  create_dataset(1, 112);
  create_dataset(2, 1120);
  create_dataset(3, 9921);
  create_dataset(4, 4098);
  create_dataset(5, 16656);
  create_dataset(6, 30000);
  create_dataset(7, 96000);
  create_dataset(8, 120000);
  create_dataset(9, 262144);
  return 0;
}
