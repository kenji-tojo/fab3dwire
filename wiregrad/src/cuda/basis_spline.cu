#include "basis_spline.h"

#include "common.cuh"


namespace wiregrad {
namespace cuda {

void cubic_basis_spline(unsigned long num_points,
                        unsigned long dim_points,
                        const float *points,
                        unsigned long num_knots,
                        const float *knots,
                        float *nodes,
                        bool cyclic)
{
    const auto num = static_cast<int>(num_knots);
    cuda::parallel_for(num, cubic_basis_spline_op{ num_points, dim_points, points, num_knots, knots, nodes, cyclic });
}

void d_cubic_basis_spline(const float *d_nodes,
                          unsigned long num_knots,
                          const float *knots,
                          unsigned long num_points,
                          unsigned long dim_points,
                          float *d_points,
                          bool cyclic)
{
    const auto num = static_cast<int>(num_knots);
    cuda::parallel_for(num, d_cubic_basis_spline_op{ d_nodes, num_knots, knots, num_points, dim_points, d_points, cyclic });
}

} // namespace cuda
} // namespace wiregrad
