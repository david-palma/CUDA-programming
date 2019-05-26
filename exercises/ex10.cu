// CUDA programming
// Exercise n. 10

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define N_ELEMS 16
#define THREADS  4

// Prototype
__global__ void dot_prod(int *a, int *b, int *c);
__host__ void ints(int *m, int N);
__host__ void print_array(int *a, int N);

int main(void)
{
    int *a, *b, *c;         // host copies of a, b, c
    int *d_a, *d_b, *d_c;   // device copies of a, b, c
    int   size = N_ELEMS * sizeof(int);

    // Allocate space for host copies of a, b, c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(sizeof(int));

    // Setup input values
    ints(a, N_ELEMS);
    ints(b, N_ELEMS);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Call the kernel on GPU
    dot_prod<<< N_ELEMS/THREADS, THREADS >>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Check the result
    print_array(a, N_ELEMS);
    print_array(b, N_ELEMS);
    printf("%d\n", *c);

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return(EXIT_SUCCESS);
}

// Vector addition (on device)
__global__ void dot_prod(int *a, int *b, int *c)
{
    __shared__ int tmp[THREADS];
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    tmp[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if(0 == threadIdx.x)
    {
        int sum = 0;
        for(int i = 0; i < THREADS; i++)
        {
            sum += tmp[i];
        }
        atomicAdd(c, sum);  // atomic operation to avoid race condition
    }
}

// Initialisation
__host__ void ints(int *m, int N)
{
    int i;
    for(i = 0; i < N; i++)
        m[i] = 1;
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
