### CSED 490C
### LAB 6
### Jeongseop Yi (49004543)

#### Q1

There are four global memory reads being performed by the kernels.
Two reads are from `scan` kernel call.
One is from `scan_add` kernel call.
Last read is from writing back to host memory.

#### Q2

There are five global memory writes are being performed by the kernels.
One is to copy the input from host to device.
Two copies are from `scan` kernel call to write output.
There is one copy to create the auxiliary input array.
Last copy is from `scan_add` kernel call to write output.

#### Q3

A single thread block synchronizes $2 \cdot \log(n)$ times to reduce its portion of the array to a single value.

#### Q4

Yes, as long as the associativity of the binary operator holds. Parallel scan does not change the order of the binary operation. Therefore, the result should not change even though the binary operator is not commutative.

#### Q5

If the binary operator does not have associativity, it is possible to get the different results between the serial version and parallel version of the scan kernel.
As parallel version keeps the temporary version of the result, the correct order of computation may not be observed, which can lead to an different result between the serial and parallel version.

#### Q7

Computation was run on cse-edu cluster with `srun -p titanxp -N 1 -n 6 --mem=32G --gres=gpu:2 --pty /bin/bash -l` command.

![](https://github.com/pjyi2147/acmicpc/assets/21299683/0428cc0b-b53c-4588-81d6-845fb95fe2ca)


| Size   | Import data to host | Allocate GPU memory | Clearing output memory | Copy data to device | Compute  | Copy output to host | Free GPU memory |   |   |
|--------|---------------------|---------------------|------------------------|---------------------|----------|---------------------|-----------------|---|---|
|     64 |            0.465333 |            0.168594 |               0.027842 |            0.028051 | 0.056307 |            0.023963 |        0.019892 |   |   |
|    112 |            0.961088 |            0.173824 |               0.028366 |             0.02857 | 0.050512 |            0.023903 |        0.019806 |   |   |
|   1120 |              4.9488 |            0.141861 |               0.024812 |            0.025049 | 0.064424 |             0.01875 |        0.017355 |   |   |
|   9921 |             37.2342 |            0.128931 |               0.021751 |            0.031856 | 0.078189 |            0.024813 |        0.015359 |   |   |
|   4098 |              15.492 |            0.129943 |               0.021313 |            0.025839 |  0.06666 |            0.019558 |        0.015853 |   |   |
|  16656 |             90.7267 |            0.172094 |               0.048947 |            0.042547 | 0.094513 |            0.035255 |        0.019973 |   |   |
|  30000 |             115.511 |            0.128909 |               0.039437 |            0.045752 | 0.110386 |            0.041493 |        0.016569 |   |   |
|  96000 |                 383 |            0.125953 |               0.040708 |            0.102128 | 0.244163 |             0.09438 |        0.014041 |   |   |
| 120000 |             493.241 |            0.131338 |               0.049725 |            0.123185 | 0.296202 |            0.114116 |        0.016124 |   |   |
| 262144 |             1093.57 |            0.227106 |               0.041595 |            0.258549 | 0.593499 |            0.224495 |        0.097628 |   |   |
