#pragma once

#include <cmath>
#include <cassert>
#include <cstdint>

#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

#if defined(COMPILE_WITH_CUDA)
#include <cuda_runtime.h>
#define DEVICE __device__
// #define HOST_DEVICE __host__ __device__
#define HOST_DEVICE __device__
#define ASSERT(x)
#define ATOMIC_ADD(data, offset, value) atomicAdd(data + offset, value)
#else
#define DEVICE
#define HOST_DEVICE
#define ASSERT(x) assert(x)
#define ATOMIC_ADD(data, offset, value) data[offset] += value
#endif


namespace wiregrad {

template<typename T> HOST_DEVICE constexpr auto Pi = static_cast<T>(M_PI);
template<typename T> HOST_DEVICE constexpr auto TwoPi = static_cast<T>(2.0f * M_PI);
template<typename T> HOST_DEVICE constexpr auto InvPi = static_cast<T>(1.0f / M_PI);

namespace helper {

template<typename T> HOST_DEVICE constexpr auto min(T a, T b) { return a < b ? a : b; }
template<typename T> HOST_DEVICE constexpr auto max(T a, T b) { return a > b ? a : b; }
template<typename T> HOST_DEVICE constexpr auto clip(T val, T lo, T hi) { return helper::max(lo, helper::min(hi, val)); }
template<typename T> HOST_DEVICE constexpr auto relu(T a) { return helper::max<T>(a, T(0.0f)); }
template<typename T> HOST_DEVICE constexpr auto sign(T a) { return a >= 0.0f ? 1.0f : -1.0f; }

} // namespace helper

} // namespace wiregrad
