#include "line_hierarchy3d.h"

#include <thrust/reduce.h>

#include "common.cuh"


namespace wiregrad {
namespace cuda {

void create_line_hierarchy3d(unsigned long num_nodes, const float *nodes,
                             unsigned long num_levels, const int *boxes,
                             unsigned long total_boxes, int *prims, float *box_points,
                             bool cyclic)
{
    assert(total_boxes == thrust::reduce(thrust::device, boxes, boxes + num_levels, 0, thrust::plus<int>{}));

    const auto num = static_cast<int>(total_boxes);
    cuda::parallel_for(num, create_line_hierarchy3d_op{ num_nodes, nodes, num_levels, boxes, prims, box_points, cyclic });
}

} // namespace cuda
} // namespace wiregrad



