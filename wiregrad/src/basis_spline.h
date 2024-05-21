#pragma once

#include "common.h"


namespace wiregrad {

void cubic_basis_spline(unsigned long num_points,
                        unsigned long dim_points,
                        const float *points,
                        unsigned long num_knots,
                        const float *knots,
                        float *nodes,
                        bool cyclic = true,
                        bool is_cuda = false);

void d_cubic_basis_spline(const float *d_nodes,
                          unsigned long num_knots,
                          const float *knots,
                          unsigned long num_points,
                          unsigned long dim_points,
                          float *d_points,
                          bool cyclic = true,
                          bool is_cuda = false,
                          int num_cpu_threads = 20);

struct cubic_basis_spline_op {
private:
    const int num_points;
    const int dim_points;
    const float *const points;
    const int num_knots;
    const float *const knots;
    float *const nodes;
    const bool cyclic;

public:
    cubic_basis_spline_op(unsigned long num_points, unsigned long dim_points, const float *points,
                          unsigned long num_knots, const float *knots, float *nodes,
                          bool cyclic)
            : num_points{ static_cast<int>(num_points) }, dim_points{ static_cast<int>(dim_points) }, points{ points }
            , num_knots{ static_cast<int>(num_knots) }, knots{ knots }, nodes{ nodes }
            , cyclic{ cyclic } { }

    HOST_DEVICE inline
    void operator()(int num, int idx)
    {
        ASSERT(num <= num_knots);
        if (idx >= num)
            return;

        const auto t = knots[idx];
        ASSERT(t >= 0.0f && t <= num_points);

        auto i0 = static_cast<int>(std::floor(t));
        auto i1 = i0 + 1;
        auto i2 = i0 + 2;
        auto i3 = i0 + 3;

        if (cyclic) {
            i0 = i0 % num_points;
            i1 = i1 % num_points;
            i2 = i2 % num_points;
            i3 = i3 % num_points;
        }

        i0 = helper::clip(i0, 0, num_points - 1);
        i1 = helper::clip(i1, 0, num_points - 1);
        i2 = helper::clip(i2, 0, num_points - 1);
        i3 = helper::clip(i3, 0, num_points - 1);

        const auto t1 = t - std::floor(t);
        const auto t2 = t1 * t1;
        const auto t3 = t2 * t1;
        const auto w0 = (1.0f / 6.0f) * (                                      t3);
        const auto w1 = (1.0f / 6.0f) * (1.0f + 3.0f * t1 + 3.0f * t2 - 3.0f * t3);
        const auto w2 = (1.0f / 6.0f) * (4.0f             - 6.0f * t2 + 3.0f * t3);
        const auto w3 = (1.0f / 6.0f) * (1.0f - 3.0f * t1 + 3.0f * t2 -        t3);

        for (int d = 0; d < dim_points; ++d)
            nodes[idx * dim_points + d] = w3 * points[i0 * dim_points + d] +
                                          w2 * points[i1 * dim_points + d] +
                                          w1 * points[i2 * dim_points + d] +
                                          w0 * points[i3 * dim_points + d];
    }

};

struct d_cubic_basis_spline_op {
private:
    const float *const d_nodes;
    const int num_knots;
    const float *const knots;
    const int num_points;
    const int dim_points;
    float *const d_points;
    const bool cyclic;

public:
    d_cubic_basis_spline_op(const float *d_nodes, unsigned long num_knots, const float *knots,
                            unsigned long num_points, unsigned long dim_points, float *d_points,
                            bool cyclic)
            : d_nodes{ d_nodes }, num_knots{ static_cast<int>(num_knots) }, knots{ knots }
            , num_points{ static_cast<int>(num_points) }, dim_points{ static_cast<int>(dim_points) }, d_points{ d_points }
            , cyclic{ cyclic } { }

    HOST_DEVICE inline
    void operator()(int num, int idx)
    {
        ASSERT(num <= num_knots);
        if (idx >= num)
            return;

        const auto t = knots[idx];
        ASSERT(t >= 0.0f && t <= num_points);

        auto i0 = static_cast<int>(std::floor(t));
        auto i1 = i0 + 1;
        auto i2 = i0 + 2;
        auto i3 = i0 + 3;

        if (cyclic) {
            i0 = i0 % num_points;
            i1 = i1 % num_points;
            i2 = i2 % num_points;
            i3 = i3 % num_points;
        }

        i0 = helper::clip(i0, 0, num_points - 1);
        i1 = helper::clip(i1, 0, num_points - 1);
        i2 = helper::clip(i2, 0, num_points - 1);
        i3 = helper::clip(i3, 0, num_points - 1);

        const auto t1 = t - std::floor(t);
        const auto t2 = t1 * t1;
        const auto t3 = t2 * t1;
        const auto w0 = (1.0f / 6.0f) * (                                      t3);
        const auto w1 = (1.0f / 6.0f) * (1.0f + 3.0f * t1 + 3.0f * t2 - 3.0f * t3);
        const auto w2 = (1.0f / 6.0f) * (4.0f             - 6.0f * t2 + 3.0f * t3);
        const auto w3 = (1.0f / 6.0f) * (1.0f - 3.0f * t1 + 3.0f * t2 -        t3);

        for (int d = 0; d < dim_points; ++d) {
            ATOMIC_ADD(d_points, i0 * dim_points + d, d_nodes[idx * dim_points + d] * w3);
            ATOMIC_ADD(d_points, i1 * dim_points + d, d_nodes[idx * dim_points + d] * w2);
            ATOMIC_ADD(d_points, i2 * dim_points + d, d_nodes[idx * dim_points + d] * w1);
            ATOMIC_ADD(d_points, i3 * dim_points + d, d_nodes[idx * dim_points + d] * w0);
        }
    }

};

} // namespace wiregrad
