// CUDA programming
// Exercise n. 8

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  2
#define THREADS 2

// Prototypes
__global__ void gpu_square_matrix_transpose(int *d_X, int *d_Y, int N);
__host__ void ints(int *m, int N);
__host__ void print_matrix(int *A, int N);

int main(void)
{
    int *d_A, *d_B;    // host copies of A, B
    int *A, *B;        // host copies of A, B

    int    N = BLOCKS * THREADS;
    int size = N * N * sizeof(int);

    // Allocate space for host copies of A, B
    A = (int *)malloc(size);
    B = (int *)malloc(size);

    // Setup input values
    ints(A, N * N);

    // Allocate space for device copies of A, B
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    // Copy inputs to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Setup the execution configuration
    dim3 dim_grid(BLOCKS, BLOCKS, 1);     // size: BLOCKS x BLOCKS x 1
    dim3 dim_block(THREADS, THREADS, 1);  // size: THREADS x THREADS x 1

    // Call the kernel on GPU
    gpu_square_matrix_transpose<<< dim_grid, dim_block >>>(d_A, d_B, N);

    // Copy result back to host
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    // Check the result
    print_matrix(A, N);
    print_matrix(B, N);

    // Cleanup
    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);

    return(EXIT_SUCCESS);
}

// Transpose of a square matrix
__global__ void gpu_square_matrix_transpose(int *d_X, int *d_Y, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accessing beyond the end of the matrices
    if(row < N && col < N)
    {
        for(int k = 0; k < N; k++)
        {
            unsigned int    pos = row * N + col;
            unsigned int tr_pos = col * N + row;
            d_Y[tr_pos] = d_X[pos];
        }
    }
}

// Initialisation
__host__ void ints(int *m, int N)
{
    int i;
    for(i = 0; i < N; i++)
        m[i] = i;
}

// Print the elements of the matrix
__host__ void print_matrix(int *A, int N)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%d\t", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}
