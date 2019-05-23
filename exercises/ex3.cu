// CUDA programming
// Exercise n. 3

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  8
#define THREADS 1

// Prototype
__global__ void add(int *a, int *b, int *c);
__host__ void ints(int *m, int N);
__host__ void print_array(int *a, int N);

int main(void)
{
    int *a, *b, *c;         // host copies of a, b, c
    int *d_a, *d_b, *d_c;   // device copies of a, b, c

    int    N = BLOCKS * THREADS;
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
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Call the kernel on GPU
    add<<< BLOCKS, THREADS >>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Check the result
    print_array(a, N);
    print_array(b, N);
    print_array(c, N);

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
__global__ void add(int *a, int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// Initialisation
__host__ void ints(int *m, int N)
{
    int i;
    for(i = 0; i < N; i++)
        m[i] = i;
}

// Print the elements of the array
__host__ void print_array(int *a, int N)
{
    for(int i = 0; i < N; i++)
    {
        printf("%d\t", a[i]);
    }
    printf("\n");
}
