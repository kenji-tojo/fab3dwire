#pragma once

#include "common.h"


namespace wiregrad {
namespace cuda {
namespace kernel {

template<typename Func>
__global__ void parallel_for(int num, Func func)
{
    auto tid = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
    func(num, /*idx=*/tid);
}

} // namespace kernel

template<typename Func> inline
void parallel_for(int num, Func func)
{
    constexpr auto num_threads = 512;
    const auto num_blocks = num / num_threads + 1;

    kernel::parallel_for<<<num_blocks, num_threads>>>(num, func);
}

} // namespace cuda
} // namespace wiregrad


