#pragma once

#include "common.h"


namespace wiregrad {

template<typename T1, typename T2> HOST_DEVICE inline
auto dot2(const T1 a[2], const T2 b[2]) -> decltype(a[0] * b[0])
{
    return a[0] * b[0] +
           a[1] * b[1];
}

template<typename T1, typename T2> HOST_DEVICE inline
auto dot3(const T1 a[3], const T2 b[3]) -> decltype(a[0] * b[0])
{
    return a[0] * b[0] +
           a[1] * b[1] +
           a[2] * b[2];
}

template<typename T> HOST_DEVICE inline
auto norm2(const T a[2]) -> T
{
    return std::sqrt(dot2(a, a));
}

template<typename T> HOST_DEVICE inline
auto norm3(const T a[3]) -> T
{
    return std::sqrt(dot3(a, a));
}

template<typename T1, typename T2> HOST_DEVICE inline
auto cross2(const T1 a[2], const T2 b[2]) -> decltype(a[0] * b[1])
{
    return a[0] * b[1] -
           a[1] * b[0];
}

template<typename T> HOST_DEVICE inline
void copy2(const T a[2], T out[2])
{
    out[0] = a[0];
    out[1] = a[1];
}

template<typename T> HOST_DEVICE inline
void copy3(const T a[3], T out[3])
{
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
}

template<typename T> HOST_DEVICE inline
void atomic_add3(T a[3], const T b[3], const T weight = T(1.0f))
{
#if !defined(COMPILE_WITH_CUDA)
    a[0] += weight * b[0];
    a[1] += weight * b[1];
    a[2] += weight * b[2];
#else
    atomicAdd(a + 0, weight * b[0]);
    atomicAdd(a + 1, weight * b[1]);
    atomicAdd(a + 2, weight * b[2]);
#endif
}

template<typename T> HOST_DEVICE inline
void sub2(const T a[2], const T b[2], T out[2])
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
}

template<typename T> HOST_DEVICE inline
void sub2(T a[2], const T b[2])
{
    a[0] -= b[0];
    a[1] -= b[1];
}

template<typename T> HOST_DEVICE inline
void sub3(const T a[3], const T b[3], T out[3])
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

template<typename T> HOST_DEVICE inline
void add2(const T a[2], const T b[2], T out[2])
{
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
}

template<typename T> HOST_DEVICE inline
void add2(T a[2], const T b[2])
{
    a[0] += b[0];
    a[1] += b[1];
}

template<typename T> HOST_DEVICE inline
void mul2(T a[2], const T b)
{
    a[0] *= b;
    a[1] *= b;
}

template<typename T> HOST_DEVICE inline
void div2(T a[2], const T b)
{
    a[0] /= b;
    a[1] /= b;
}

template<typename T> HOST_DEVICE inline
void div3(T a[3], const T b)
{
    a[0] /= b;
    a[1] /= b;
    a[2] /= b;
}

template<typename T> HOST_DEVICE inline
void local_coords(const T ez[3], T ex[3], T ey[3])
{
    const T sign = ez[2] >= 0.0f ? 1.0f : -1.0f;
    const T a = -1.0f / (sign + ez[2]);
    const T b = ez[0] * ez[1] * a;
    ex[0] = 1.0f + sign * ez[0] * ez[0] * a;
    ex[1] = sign * b;
    ex[2] = -sign * ez[0];
    ey[0] = b;
    ey[1] = sign + ez[1] * ez[1] * a;
    ey[2] = -ez[1];
    ASSERT(std::abs(norm3(ex) - 1.0f) < 1e-5f);
    ASSERT(std::abs(norm3(ey) - 1.0f) < 1e-5f);
    ASSERT(std::abs(dot3(ex, ey)) < 1e-5f);
    ASSERT(std::abs(dot3(ey, ez)) < 1e-5f);
    ASSERT(std::abs(dot3(ez, ex)) < 1e-5f);
}

} // namespace wiregrad



