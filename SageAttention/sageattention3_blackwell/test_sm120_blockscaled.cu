// test_fp4_gemm.cu
#include <stdio.h>
#include "cute/arch/mma_sm120.hpp"
#include "cute/tensor.hpp"
using namespace cute;

__global__ void test_sm120_kernel()
{
    printf("Launching SM120 BlockScaled kernel test!\\n");

    // 声明MMA算子 (FP8 compatible)
    using MMA = MMA_Atom<SM120_16x8x64_TN>;
    MMA mma;
    printf("SM120 MMA Atom instantiated successfully.\\n");
}

int main()
{
    printf("=== Blackwell SM120 TensorCore Compile Test ===\\n");
    test_sm120_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
