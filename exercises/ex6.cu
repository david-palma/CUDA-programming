// CUDA programming
// Exercise n. 6

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  256
#define THREADS 256

// Prototype
__global__ saxpy(int a, int *x, int *y, int N);
__host__ void ints(int* m, int N);

int main(void)
{
    float *x, *y, a;    // host copies of x, y, a
    float *d_x, *d_y;   // device copies of x, y

    int    N = 2 * BLOCKS * THREADS;
    int size = N * sizeof(float);

    // Allocate space for host copies of x, y
    x = (float *)malloc(size);
    y = (float *)malloc(size);

    // Setup input values
    random_ints(x, N);
    random_ints(y, N);
    a = 3.0;

    // Allocate space for device copies of x, y
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    // Copy inputs to device
    cudaMemcpy(d_x, &x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y, size, cudaMemcpyHostToDevice);

    // Call the saxpy() kernel on GPU
    saxpy<<< BLOCKS, THREADS >>>(a, d_x, d_y, N);

    // Copy result back to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);

    return(EXIT_SUCCESS);
}

// Single-precision A*X Plus Y
__global__ saxpy(float a, float *x, float *y, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accessing beyond the end of the arrays
    if (index < N)
    {
        y[index] = a*x[index] + y[index];
    }
}

// Initialisation
__host__ void ints(int* m, int N)
{
    int i;
    for (i = 0; i < N; i++)
        m[i] = i;
}
