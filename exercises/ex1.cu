// CUDA programming
// Exercise n. 1

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS  1
#define THREADS 32

// Prototypes
void say_hello_from_cpu (void);
__global__ say_hello_from_gpu (void);

int main(void)
{
    dim3 grid (BLOCKS);     // blocks in the grid
    dim3 block (THREADS);   // threads per block

    // Call the CPU version
    cpu_helloworld();

    // Call the GPU version
    gpu_helloworld <<< grid, block >>> ();

    return(EXIT_SUCCESS);
}

// CPU version of hello world!
void say_hello_from_cpu (void)
{
    printf ("Hello from the CPU!\n");
}

// GPU version of hello world!
__global__ void say_hello_from_gpu (void)
{
    int threadId = threadIdx.x;
    printf ("Hello from the GPU! My threadId is %d\n", threadId);
}
