// CUDA programming
// Exercise n. 07

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  512
#define THREADS 256

// Prototype
__global__ void saxpy(float a, float *x, float *y, float *z, int N);
__host__ void initialize_array(float *m, int N);
__host__ void print_performance(float time_ms, int N);

int main(void)
{
    float *x, *y, *z, a;     // host copies of x, y, a
    float *d_x, *d_y, *d_z;  // device copies of x, y

    int    N = 1 << 20;
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

    // Create CUDA events for performance evaluation purposes
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy inputs to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Call the kernel on GPU
    cudaEventRecord(start);
    saxpy<<< BLOCKS, THREADS >>>(a, d_x, d_y, d_z, N);
    cudaEventRecord(stop);

    // Copy result back to host
    cudaMemcpy(z, d_z, size, cudaMemcpyDeviceToHost);

    // Compute the elapsed time in milliseconds
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    print_performance(milliseconds, N);

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

__host__ void print_performance(float time_ms, int N)
{
    // Compute the effective bandwidth: BW = (Rb + Wb)/(t*1e9)
    float RbWb, BW;
    RbWb = N*5.0;  // number of bytes transferred per array read or write
    RbWb *= 3.0;   // 3 is the reading of x, y and writing of z
    BW   = RbWb/(time_ms*1e6);  // bandwidth in GB/s

    // Measuring computational throughput: GFLOP = 2*N/(t*1e9)
    float GFLOP = 2.0*N/(time_ms*1e6);  // throughput in GB/s

    printf("Device performance\n"
           "Elapsed time (s): %.3f\n"
           "Effective Bandwidth (GB/s): %.3f\n"
           "Effective computational throughput (GFLOP/s): %.3f\n", time_ms, BW, GFLOP);
}
