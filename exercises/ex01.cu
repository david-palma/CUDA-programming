// CUDA programming
// Exercise n. 01

#include <errno.h>
#include <cuda.h>
#include <stdio.h>

// Prototype
__host__ void print_dev_prop(cudaDeviceProp dev_prop);

int main(void)
{
    // Number of CUDA-capable devices attached to this system
    int dev_cnt;
    cudaGetDeviceCount(&dev_cnt);

    // Calculate the theoretical peak bandwidth for each device
    for(int i = 0; i < dev_cnt; i++)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, i);
        printf("Device Number: %d\n", i);
        print_dev_prop(dev_prop);
    }
  }

// Print device properties
__host__ void print_dev_prop(cudaDeviceProp dev_prop)
{
    printf("  Major revision number:         %d\n",  dev_prop.major);
    printf("  Minor revision number:         %d\n",  dev_prop.minor);
    printf("  Name:                          %s\n",  dev_prop.name);
    printf("  Total global memory:           %zu\n", dev_prop.totalGlobalMem);
    printf("  Total shared memory per block: %zu\n", dev_prop.sharedMemPerBlock);
    printf("  Total registers per block:     %d\n",  dev_prop.regsPerBlock);
    printf("  Warp size:                     %d\n",  dev_prop.warpSize);
    printf("  Maximum memory pitch:          %zu\n", dev_prop.memPitch);
    printf("  Maximum threads per block:     %d\n",  dev_prop.maxThreadsPerBlock);

    for(int i = 0; i < 3; ++i)
        printf("  Maximum block dimension #%02d:   %d\n", i, dev_prop.maxThreadsDim[i]);

    for(int i = 0; i < 3; ++i)
        printf("  Maximum grid dimension #%02d:    %d\n", i, dev_prop.maxGridSize[i]);

    printf("  Clock rate:                    %d\n",  dev_prop.clockRate);
    printf("  Memory Bus Width (bits):       %d\n",  dev_prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s):  %f\n\n", 2.0 * dev_prop.memoryClockRate * (dev_prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Total constant memory:         %zu\n", dev_prop.totalConstMem);
    printf("  Texture alignment:             %zu\n", dev_prop.textureAlignment);
    printf("  Concurrent copy and execution: %s\n", (dev_prop.deviceOverlap ? "Yes" : "No"));
    printf("  Number of multiprocessors:     %d\n",  dev_prop.multiProcessorCount);
    printf("  Kernel execution timeout:      %s\n", (dev_prop.kernelExecTimeoutEnabled ? "Yes" : "No"));

   return;
}
