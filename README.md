# CUDA C/C++ programming

This repository is meant to provide open source resources for educational purposes about CUDA C/C++ programming, which is the C/C++ interface to the CUDA parallel computing platform.
In CUDA, the host refers to the CPU and its memory, while the device refers to the GPU and its memory.
Code run on the host can manage memory on both the host and device, and also launches kernels which are functions executed on the device by many GPU threads in parallel.

**NOTE**: it is assumed that you have access to a computer with a CUDA-enabled NVIDIA GPU.

## List of the exercises
Here you can find the solutions for different simple exercises about GPU programming in CUDA C/C++.
The source code is well commented and easy to follow, though a minimum knowledge of parallel architectures is recommended.

* [exercise 00](./exercises/ex00.cu): hello, world!
* [exercise 01](./exercises/ex01.cu): print devices properties
* [exercise 02](./exercises/ex02.cu): addition
* [exercise 03](./exercises/ex03.cu): vector addition using parallel blocks
* [exercise 04](./exercises/ex04.cu): vector addition using parallel threads
* [exercise 05](./exercises/ex05.cu): vector addition combining blocks and threads
* [exercise 06](./exercises/ex06.cu): single-precision A*X Plus Y
* [exercise 07](./exercises/ex07.cu): time, bandwidth, and throughput computation (single-precision A*X Plus Y)
* [exercise 08](./exercises/ex08.cu): multiplication of square matrices
* [exercise 09](./exercises/ex09.cu): transpose of a square matrix
* [exercise 10](./exercises/ex10.cu): dot product (with shared memory)
## Compiling and running the code

The CUDA C/C++ compiler `nvcc` is part of the NVIDIA CUDA Toolkit which is used to separate source code into host and device components. Then, you can compile the code with `nvcc`.

**NOTE**: to find out how long the kernel takes to run or to check the memory usage, you can type `nvprof ./<binary>` or `cuda-memcheck ./<binary>` on the command line, respectively.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
