// CUDA programming
// Exercise n. 08

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  2
#define THREADS 2

// Prototypes
__global__ void square_matrix_mult(int *d_A, int *d_B, int *d_P, int N);
__host__ void initialize_array(int *array, int N)
__host__ void eye(int *M, int N);
__host__ void print_matrix(int *A, int N);

int main(void)
{
    int *A, *B, *P;          // host copies of A, B, P
    int *d_A, *d_B, *d_P;    // device copies of A, B, P

    int    N = BLOCKS * THREADS;
    int size = N * N * sizeof(int);

    // Allocate space for host copies of A, B, P
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    P = (int *)malloc(size);

    // Setup input values
    initialize_array(A, N * N);
    eye(B, N);
    initialize_array(P, N * N);

    // Allocate space for device copies of A, B, P
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_P, size);

    // Copy inputs to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, size, cudaMemcpyHostToDevice);

    // Setup the execution configuration
    dim3 dim_grid(BLOCKS, BLOCKS, 1);     // size: BLOCKS x BLOCKS x 1
    dim3 dim_block(THREADS, THREADS, 1);  // size: THREADS x THREADS x 1

    // Call the kernel on GPU
    square_matrix_mult<<< dim_grid, dim_block >>>(d_A, d_B, d_P, N);

    // Copy result back to host
    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    // Check the result
    print_matrix(A, N);
    print_matrix(B, N);
    print_matrix(P, N);

    // Cleanup
    free(A);
    free(B);
    free(P);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_P);

    return(EXIT_SUCCESS);
}

// Square matrix multiplication (on device)
__global__ void square_matrix_mult(int *d_A, int *d_B, int *d_P, int N)
{
    // Pvalue is the element of the matrix that is computed by the thread
    int Pvalue = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accessing beyond the end of the matrices
    if(row < N && col < N)
    {
        for(int k = 0; k < N; k++)
        {
            Pvalue += d_A[row * N + k] * d_B[k * N + col];
        }

        d_P[row * N + col] = Pvalue;
    }
}

// Host function to initialize an array
__host__ void initialize_array(int *array, int N)
{
    for (int i = 0; i < N; i++)
    {
        array[i] = i + 1;  // Sequential integers
    }
}

// Host function to build an identity matrix
__host__ void eye(int *M, int N)
{
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            M[i*N + j] = (i == j);
}

// Host function to print a matrix
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
