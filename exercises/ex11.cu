// CUDA programming
// Exercise n. 11

#include <cuda.h>
#include <stdio.h>

#define N 16        // Number of elements
#define THREADS 8   // Threads per block

// Prototypes
__global__ void prefix_sum(int *input, int *output, int n);
__host__ void initialize_array(int *a, int n);
__host__ void print_array(int *a, int n);

int main(void)
{
    int *input, *output;         // Host copies
    int *d_input, *d_output;     // Device copies
    int size = N * sizeof(int);

    // Allocate space for host arrays
    input = (int *)malloc(size);
    output = (int *)malloc(size);

    // Initialize input array
    initialize_array(input, N);

    // Allocate space for device arrays
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    // Copy input array to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Call the kernel
    prefix_sum<<<1, THREADS>>>(d_input, d_output, N);

    // Copy results back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Input Array:\n");
    print_array(input, N);
    printf("Prefix Sum (Exclusive):\n");
    print_array(output, N);

    // Cleanup
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}

// Kernel: Prefix sum (exclusive scan)
__global__ void prefix_sum(int *input, int *output, int n)
{
    __shared__ int temp[THREADS];  // Shared memory for computation
    int index = threadIdx.x;

    // Load elements into shared memory
    if (index < n)
        temp[index] = input[index];
    __syncthreads();

    // Compute prefix sum using the shared memory
    for (int offset = 1; offset < n; offset *= 2)
    {
        int value = 0;
        if (index >= offset)
            value = temp[index - offset];
        __syncthreads();
        temp[index] += value;
        __syncthreads();
    }

    // Write results back to the output array
    if (index < n)
        output[index] = (index == 0) ? 0 : temp[index - 1];  // Exclusive scan
}

// Host function to initialize an array
__host__ void initialize_array(int *a, int N)
{
    for(int i = 0; i < N; i++)
        a[i] = i + 1;  // Sequential integers
}

// Host function to print an array
__host__ void print_array(int *a, int N)
{
    for(int i = 0; i < N; i++)
    {
        printf("%d\t", a[i]);
    }
    printf("\n");
}
