#include "line_hierarchy2d.h"

#include <thrust/reduce.h>

#include "common.cuh"


namespace wiregrad {
namespace cuda {

void create_line_hierarchy2d(unsigned long num_nodes, const float *nodes,
                              unsigned long num_levels, const int *boxes,
                              unsigned long total_boxes, int *prims, float *bounds,
                              bool cyclic)
{
    assert(total_boxes == thrust::reduce(thrust::device, boxes, boxes + num_levels, 0, thrust::plus<int>{}));

    auto num = static_cast<int>(total_boxes);
    cuda::parallel_for(num, create_line_hierarchy2d_op{ num_nodes, nodes, num_levels, boxes, prims, bounds, cyclic });
}

} // namespace cuda
} // namespace wiregrad


