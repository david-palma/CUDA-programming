// CUDA programming
// Exercise n. 06

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  4
#define THREADS 4

// Prototype
__global__ void saxpy(float a, float *x, float *y, float *z, int N);
__host__ void initialize_array(float *m, int N);
__host__ void print_saxpy(float a, float *x, float *y, float *z, int N);

int main(void)
{
    float *x, *y, *z, a;     // host copies of x, y, a
    float *d_x, *d_y, *d_z;  // device copies of x, y

    int    N = BLOCKS * THREADS;
    int size = N * sizeof(float);

    // Allocate space for host copies of x, y
    x = (float *)malloc(size);
    y = (float *)malloc(size);
    z = (float *)malloc(size);

    // Setup input values
    initialize_array(x, N);
    initialize_array(y, N);
    a = 3.0/2.5;

    // Allocate space for device copies of x, y
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_z, size);

    // Copy inputs to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Call the kernel on GPU
    saxpy<<< BLOCKS, THREADS >>>(a, d_x, d_y, d_z, N);

    // Copy result back to host
    cudaMemcpy(z, d_z, size, cudaMemcpyDeviceToHost);

    print_saxpy(a, x, y, z, N);

    // Cleanup
    free(x);
    free(y);
    free(z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return(EXIT_SUCCESS);
}

// Single-precision A*X Plus Y (on device)
__global__ void saxpy(float a, float *x, float *y, float *z, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accessing beyond the end of the arrays
    if(index < N)
    {
        z[index] = a * x[index] + y[index];
    }
}

// Host function to initialize an array
__host__ void initialize_array(float *m, int N)
{
    for(int i = 0; i < N; i++)
        m[i] = i/(i + 1.0);
}

// Host function to print the elements of the equation
__host__ void print_saxpy(float a, float *x, float *y, float *z, int N)
{
    for(int i = 0; i < N; i++)
    {
        printf("%5.2f = %5.2f x %5.2f + %5.2f\n", z[i], a, x[i], y[i]);
    }
    printf("\n");
}
