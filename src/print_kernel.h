#ifndef PRINT_H
#define PRINT_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCKSIZE 256

#define checkCudaKernel(...)                                                                                           \
    __VA_ARGS__;                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = cudaPeekAtLastError();                                                                \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            std::cout << "launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;                             \
        }                                                                                                              \
    } while (0);

void print_dla_addr(half *buffer, int size, int total, cudaStream_t stream);



#endif