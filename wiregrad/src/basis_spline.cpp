#include "basis_spline.h"

#include <cstdio>
#include <vector>

#include <Eigen/Core>

#include "thread.h"


namespace wiregrad {

#if defined(WIREGRAD_CUDA)
namespace cuda {

extern void cubic_basis_spline(unsigned long num_points,
                               unsigned long dim_points,
                               const float *points,
                               unsigned long num_knots,
                               const float *knots,
                               float *nodes,
                               bool cyclic);

extern void d_cubic_basis_spline(const float *d_nodes,
                                 unsigned long num_knots,
                                 const float *knots,
                                 unsigned long num_points,
                                 unsigned long dim_points,
                                 float *d_points,
                                 bool cyclic);

} // namespace cuda
#endif

void cubic_basis_spline(unsigned long num_points,
                        unsigned long dim_points,
                        const float *points,
                        unsigned long num_knots,
                        const float *knots,
                        float *nodes,
                        bool cyclic,
                        bool is_cuda)
{
    ASSERT(num_knots >= num_points);

    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::cubic_basis_spline(num_points, dim_points, points,
                                 num_knots, knots, nodes,
                                 cyclic);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        cubic_basis_spline_op func{
                num_points, dim_points, points,
                num_knots, knots, nodes,
                cyclic
        };

        const auto num = static_cast<int>(num_knots);
        cpu::parallel_for(num, [&] (int idx, int tid) { func(num, idx); });
    }
}

void d_cubic_basis_spline(const float *d_nodes,
                          unsigned long num_knots,
                          const float *knots,
                          unsigned long num_points,
                          unsigned long dim_points,
                          float *d_points,
                          bool cyclic,
                          bool is_cuda,
                          [[maybe_unused]] int num_cpu_threads)
{
    ASSERT(num_knots >= num_points);

    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::d_cubic_basis_spline(d_nodes, num_knots, knots,
                                   num_points, dim_points, d_points,
                                   cyclic);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        using namespace Eigen;

        const auto num_threads = helper::max(1, num_cpu_threads);

        Matrix<float, Dynamic, Dynamic, RowMajor> d_points_buffer;
        d_points_buffer.setZero(num_threads, static_cast<long>(num_points * dim_points));

        std::vector<d_cubic_basis_spline_op> funcs;
        funcs.reserve(num_threads);

        for (int tid = 0; tid < num_threads; ++tid)
            funcs.emplace_back(d_nodes, num_knots, knots,
                               num_points, dim_points, d_points_buffer.row(tid).data(),
                               cyclic);

        const auto num = static_cast<int>(num_knots);
        cpu::parallel_for(num, [&] (int idx, int tid) { funcs[tid](num, idx); }, num_threads);

        Map<VectorXf>{ d_points, d_points_buffer.cols() } += d_points_buffer.colwise().sum();
    }
}

} // namespace wiregrad
