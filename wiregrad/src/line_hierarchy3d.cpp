#include "line_hierarchy3d.h"

#include <cstdio>

#include <vector>
#include <numeric>
#include <functional>

#include "thread.h"


namespace wiregrad {

#if defined(WIREGRAD_CUDA)
namespace cuda {

extern void create_line_hierarchy3d(unsigned long num_nodes, const float *nodes,
                                    unsigned long num_levels, const int *boxes,
                                    unsigned long total_boxes, int *prims, float *box_points,
                                    bool cyclic);

} // namespace cuda
#endif

void create_line_hierarchy3d(unsigned long num_nodes, const float *nodes,
                             unsigned long num_levels, const int *boxes,
                             unsigned long total_boxes, int *prims, float *box_points,
                             bool cyclic, bool is_cuda)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::create_line_hierarchy3d(num_nodes, nodes, num_levels, boxes, total_boxes, prims, box_points, cyclic);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        ASSERT(total_boxes == std::accumulate(boxes, boxes + num_levels, 0, std::plus<>{}));

        create_line_hierarchy3d_op func{
                num_nodes, nodes,
                num_levels, boxes,
                prims, box_points,
                cyclic
        };

        const auto num = static_cast<int>(total_boxes);
        cpu::parallel_for(num, [&] (int idx, int tid) { func(num, idx); });
    }
}

} // namespace wiregrad
