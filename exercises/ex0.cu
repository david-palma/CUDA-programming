// CUDA programming
// Exercise n. 0

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

// Prototype
__host__ void printDevProp(cudaDeviceProp devProp);

int main(void)
{
    // Number of CUDA-capable devices attached to this system
    int devCount;
    cudaGetDeviceCount(&devCount);

    // Calculate the theoretical peak bandwidth for each device
    for(int i = 0; i < devCount; i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Device Number: %d\n", i);
        printDevProp(devProp);
    }
  }

// Print device properties
__host__ void printDevProp(cudaDeviceProp devProp)
{
    printf("  Major revision number:         %d\n",  devProp.major);
    printf("  Minor revision number:         %d\n",  devProp.minor);
    printf("  Name:                          %s\n",  devProp.name);
    printf("  Total global memory:           %zu\n", devProp.totalGlobalMem);
    printf("  Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
    printf("  Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("  Warp size:                     %d\n",  devProp.warpSize);
    printf("  Maximum memory pitch:          %zu\n", devProp.memPitch);
    printf("  Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);

    for(int i = 0; i < 3; ++i)
        printf("  Maximum block dimension #%02d:   %d\n", i, devProp.maxThreadsDim[i]);

    for(int i = 0; i < 3; ++i)
        printf("  Maximum grid dimension #%02d:    %d\n", i, devProp.maxGridSize[i]);

    printf("  Clock rate:                    %d\n",  devProp.clockRate);
    printf("  Memory Bus Width (bits):       %d\n",  devProp.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s):  %f\n\n", 2.0 * devProp.memoryClockRate * (devProp.memoryBusWidth / 8) / 1.0e6);
    printf("  Total constant memory:         %zu\n", devProp.totalConstMem);
    printf("  Texture alignment:             %zu\n", devProp.textureAlignment);
    printf("  Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("  Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("  Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));

   return;
}
