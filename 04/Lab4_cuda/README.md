# CSE490C Heterogeneous Parallel Computing

# Lab assignment 4 - CUDA 2D Convolution 

This lab is based on the "GPU Teaching Kit Labs". The kit and associated lab are produced jointly by NVIDIA and University of Illinois (UIUC). 

# System and Software Requirements

You must use an [NVIDIA CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) to use the compiled binaries.
The labs in the Teaching Kit require a CUDA supported operating system,
C compiler, and the CUDA Toolkit version 8 or later. 
The CUDA Toolkit can be downloadedfrom the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page.
Instructions on how to install the CUDA Toolkit are available in the
[Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
Aside from a C compiler and the CUDA Toolkit, [CMake](https://cmake.org/) 3.17 or later is required
to generate build scripts for your target IDE and compiler.


I strongly recommend that you use the computing cluster at the department of computer science and/or the graduate school or artificial intelligence. 
If you do not have access to either, you can request an account using the following form: 
[CSE cluster](https://forms.gle/1sh2noQfKghFcYvU6)
[AIGS cluster](https://forms.gle/N1mJqPdujT5fcvi4A)


# Compile and running the lab
If you compile and run the lab, all the software required is already installed on the cluster. 
Otherwise, the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [CMake](https://cmake.org/) must be installed.


1. Build libgputk 
The following procedure will build `libgputk` (the support library) that will be linked with your template file for processing command-line arguments, logging time, and checking the correctness of your solution. 


Create the target build directory

~~~
mkdir build-dir
cd build-dir
~~~

We will use `ccmake`

~~~
ccmake /path/to/Lab1 
~~~

You will see the following screen

![ccmake](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+11.58.27.png)

Pressing `c` would configure the build to your system (in the process detecting
  the compiler, the CUDA Toolkit location, etc...).

![ccmake-config](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+12.03.26.png)

~~~
BUILD_LIBgpuTK_LIBRARY          *ON
BUILD_LOGTIME                   *ON
~~~

If you have modified the above, then you should type `g` to regenerate the Makefile and then `e` to quit out of `ccmake`.
You can then use the `make` command to build the labs.

![make](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+12.11.15.png)


2. Build the data generator and template 

The following will compile the template file that you will modify to implement 2D convolution in CUDA, and the data generator that will generate input files. 

~~~
cd sources 
make template 
make dataset_generator 
~~~

You can generate input data with

~~
./dataset_generator
~~

This will create a directory that contains multiple pairs of input data. You can modify the file to generate input data of different sizes.


