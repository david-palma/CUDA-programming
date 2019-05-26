// CUDA programming
// Exercise n. 02

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  1
#define THREADS 1

// Prototype
__global__ void add(int *a, int *b, int *c);

int main(void)
{
    int a, b, c;            // host copies of a, b, c
    int *d_a, *d_b, *d_c;   // device copies of a, b, c
    int size = sizeof(int);

    // Setup input values
    a = 5;
    b = 9;

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // Call the kernel on GPU
    add<<< BLOCKS, THREADS >>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return(EXIT_SUCCESS);
}

// Addition (on device)
__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}
