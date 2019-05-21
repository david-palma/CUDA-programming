// CUDA programming
// Exercise n. 3

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  512
#define THREADS 1

// Prototype
__global__ add(int *a, int *b, int *c);

int main(void)
{
    int *a, *b, *c;         // host copies of a, b, c
    int *d_a, *d_b, *d_c;   // device copies of a, b, c

    int    N = BLOCKS * THREADS;
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for device copies of a, b, c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Setup input values
    random_ints(a, N);
    random_ints(b, N);

    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // Call the add() kernel on GPU
    add<<< BLOCKS, THREADS >>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return(EXIT_SUCCESS);
}

// Vector addition
__global__ add(int *a, int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
