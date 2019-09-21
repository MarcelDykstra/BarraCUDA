#include <stdio.h>
#include <string.h>
#include <cuda.h>

//------------------------------------------------------------------------------
extern "C" int cuDeviceCount(void)
{
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

//------------------------------------------------------------------------------
extern "C" cudaDeviceProp cuDeviceProperties(void)
{
    int count;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&count);
    for (int i=0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
    }
    return prop;
}

