// CUDA programming
// Exercise n. 5

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  256
#define THREADS 256

// Prototype
__global__ add(int *a, int *b, int *c);
__host__ void ints(int* m, int N);

int main(void)
{
    int *a, *b, *c;         // host copies of a, b, c
    int *d_a, *d_b, *d_c;   // device copies of a, b, c

    int    N = 2 * BLOCKS * THREADS;
    int size = N * sizeof(int);

    // Allocate space for host copies of a, b, c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Setup input values
    ints(a, N);
    ints(b, N);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    c[index]  = a[index] + b[index];
}

// Initialisation
__host__ void ints(int* m, int N)
{
    int i;
    for (i = 0; i < N; i++)
        m[i] = i;
}
