[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fpjyi2147%2FGPU_Computing_lab&count_bg=%23FFBC93&title_bg=%238FC2FF&icon=&icon_color=%23E7E7E7&title=visits&edge_flat=false)](https://hits.seeyoufarm.com)

## GPU Computing Labs

### Preface 

Exercies on how to use Nvidia GPUs using CUDA and learn optimization methods specific for GPU-parallelized computing.

### How to run? 

In each chapter folder, there is a `readme.md` file explaining on how to build and run the template files for each lab.

### Topics 

#### A1 

* Use CUDA APIs to implement vector addition
* Learn basic transfer of data between CPU and GPU and memory allocation

#### A2

* Implement tiled dense matrix multiplication using CUDA
* Learn how to allocate memory on GPU and transfer data between GPU and CPU
* Use shared memory to optimize computation and memory latency
* Find the difference of performance on usage of shared memory

#### A3

* Use pinned memory with CUDA streams by implementing vector addition
* Benchmark the performance on the usage of CUDA streams by hiding memory latency

#### A4

* Apply convolution to a ppm image using CUDA APIs
* Find overhead in using output tiling algorithm
* Evaluate performance of output tiling algorithm with different tiling sizes including extreme ones

#### A5 

* Perform Histogram Reduction using CUDA APIs
* Learn how to use atomics for memory address in GPU through CUDA memory APIs
* Analyze performance impact of using atomics 

### A6

* Implement 1D inclusive parallel scan using CUDA and work-efficient algorithm (Brent-Kung)
* Learn the restriction of the algorithm based on the commutative feature of binary operator for scan

### A7

* Implement sparse matrix-vector (SPMV) mutiplication using CUDA and a transposed JDS (Jagged Diagonal Sparse) formatted matrix
* Implement the conversion from 2D array to JDS-formatted matrix using C++ and STL vector
* Compare the performance difference of the kernel between the version using shared memory and not using shared memory

### Project

* Compare the performance between the serial (CPU) and parallel verions of Quick Hull algorithm for Convex Hull problem.
* Introduced parallelized sorting, reduction, and computation to reduce the runtime of the algorithm
* Check submodule for more information
