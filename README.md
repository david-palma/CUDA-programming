# CUDA C/C++ programming

This repository is meant to provide open source resources for educational purposes about CUDA C/C++ programming, which is the C/C++ interface to the CUDA parallel computing platform.
In CUDA, the host refers to the CPU and its memory, while the device refers to the GPU and its memory.
Code run on the host can manage memory on both the host and device, and also launches kernels which are functions executed on the device by many GPU threads in parallel.

**NOTE**: it is assumed that you have access to a computer with a CUDA-enabled NVIDIA GPU.

## List of the exercises
Here you can find the solutions for different simple exercises about GPU programming in CUDA C/C++.
The source code is well commented and easy to follow, though a minimum knowledge of parallel architectures is recommended.

* [exercise 1](https://github.com/david-palma/CUDA_programming/tree/master/exercises/ex1): hello, world!
* [exercise 2](https://github.com/david-palma/CUDA_programming/tree/master/exercises/ex2): addition
* [exercise 3](https://github.com/david-palma/CUDA_programming/tree/master/exercises/ex3): vector addition using parallel blocks
* [exercise 4](https://github.com/david-palma/CUDA_programming/tree/master/exercises/ex4): vector addition using parallel threads
* [exercise 5](https://github.com/david-palma/CUDA_programming/tree/master/exercises/ex5): vector addition combining blocks and threads

## Compiling and running the code

The CUDA C/C++ compiler (nvcc) is part of the NVIDIA CUDA Toolkit which is used to separate source code into host and device components.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
