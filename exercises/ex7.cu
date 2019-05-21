// CUDA programming
// Exercise n. 7

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  16
#define THREADS 1

// Prototypes
__global__ void gpu_square_matrix_mult(int *d_A, int *d_B, int *d_P, int dim);
__host__ void ints(int* m, int N);

int main(void)
{
    int *d_A, *d_B, *d_P;    // host copies of A, B, P
    int *A, *B, *P;          // host copies of A, B, P

    int    N = BLOCKS * THREADS;
    int size = N * N * sizeof(int);

    // Allocate space for host copies of A, B, P
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    P = (int *)malloc(size);

    // Setup input values
    ints(A, N * N);
    ints(B, N * N);
    ints(P, N * N);

    // Allocate space for device copies of A, B, P
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_P, size);

    // Copy inputs to device
    cudaMemcpy(d_x, &x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y, size, cudaMemcpyHostToDevice);

    // Setup the execution configuration
    dim3 dim_grid(BLOCKS, BLOCKS, 1);     // size: BLOCKS x BLOCKS x 1
    dim3 dim_block(THREADS, THREADS, 1);  // size: THREADS x THREADS x 1

    // Call the saxpy() kernel on GPU
    gpu_square_matrix_mult<<< dim_grid, dim_block >>>(d_A, d_B, d_P, N);

    // Copy result back to host
    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(A);
    free(B);
    free(P);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_P);

    return(EXIT_SUCCESS);
}

// Square matrix multiplication
__global__ void gpu_square_matrix_mult(int *d_A, int *d_B, int *d_P, int dim)
{
    // Pvalue is the element of the matrix that is computed by the thread
    int Pvalue = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accessing beyond the end of the matrices
    if(row < dim && col < dim)
    {
        int k;
        for (k = 0; k < dim; k++)
        {
            Pvalue += d_A[row * dim + k] * d_B[k * dim + col];
        }

        d_P[row * dim + col] = Pvalue;
    }
}

// Initialisation
__host__ void ints(int* m, int N)
{
    int i;
    for (i = 0; i < N; i++)
        m[i] = i;
}
