#include "print_kernel.h"

static __global__ void print_kernel(half *buffer, int size, int total)
{
    int idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if(idx >= total)
    {
        return;
    }
    if(idx < size)
    {
        printf("idx : %d, val : %lf \n", idx, __half2float(buffer[idx]));
    }
}

void print_dla_addr(half *buffer, int size, int total, cudaStream_t stream)
{
    dim3 grid((total+BLOCKSIZE-1) / float(BLOCKSIZE));
    dim3 block(BLOCKSIZE);
    checkCudaKernel(print_kernel<<<grid, block, 0, stream>>>(buffer, size, total));
    return;
}

