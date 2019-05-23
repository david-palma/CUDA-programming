// CUDA programming
// Exercise n. 8

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  16
#define THREADS  1

// Prototypes
__global__ void gpu_matrix_transpose(int *d_X, int *d_Y, int dim);
__host__ void ints(int* m, int N);

int main(void)
{
    int *d_X, *d_Y;  // host copies of X, Y
    int *X, *Y;      // host copies of X, Y

    int    N = BLOCKS * THREADS;
    int size = N * N * sizeof(int);

    // Allocate space for host copies of X, Y
    X = (int *)malloc(size);
    Y = (int *)malloc(size);

    // Setup input values
    ints(X, N * N);
    ints(Y, N * N);

    // Allocate space for device copies of X, Y
    cudaMalloc((void **)&d_X, size);
    cudaMalloc((void **)&d_Y, size);

    // Copy inputs to device
    cudaMemcpy(d_x, &x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y, size, cudaMemcpyHostToDevice);

    // Setup the execution configuration
    dim3 dim_grid(BLOCKS, BLOCKS, 1);     // size: BLOCKS x BLOCKS x 1
    dim3 dim_block(THREADS, THREADS, 1);  // size: THREADS x THREADS x 1

    // Call the saxpy() kernel on GPU
    gpu_matrix_transpose<<< dim_grid, dim_block >>>(d_X, d_Y, N);

    // Copy result back to host
    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(X);
    free(Y);
    cudaFree(d_X);
    cudaFree(d_Y);

    return(EXIT_SUCCESS);
}

// Transpose of a square matrix
__global__ void gpu_matrix_transpose(int *d_X, int *d_Y, int dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accessing beyond the end of the matrices
    if(row < dim && col < dim)
    {
        int k;
        for (k = 0; k < dim; k++)
        {
            unsigned int    pos = row * dim + col;
            unsigned int tr_pos = col * dim + row;
            d_Y[tr_pos] = d_X[pos];
        }
    }
}

// Initialisation
__host__ void ints(int* m, int N)
{
    int i;
    for (i = 0; i < N; i++)
        m[i] = i;
}
