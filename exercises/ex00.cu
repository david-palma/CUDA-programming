// CUDA programming
// Exercise n. 00

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCKS   1
#define THREADS 32

// Prototypes
void cpu_hello_world(void);
__global__ void gpu_hello_world(void);

int main(void)
{
    // Call the CPU version
    cpu_hello_world();

    // Call the GPU version
    gpu_hello_world<<< BLOCKS, THREADS >>>();

    return(EXIT_SUCCESS);
}

// CPU version of hello world!
void cpu_hello_world(void)
{
    printf("Hello from the CPU!\n");
}

// GPU version of hello world!
__global__ void gpu_hello_world(void)
{
    int threadId = threadIdx.x;
    printf("Hello from the GPU! My threadId is %d\n", threadId);
}
