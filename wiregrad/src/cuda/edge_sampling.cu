#include "edge_sampling.h"

#include "common.cuh"


namespace wiregrad {
namespace cuda {

void sample_edges(const float *nodes, unsigned long num_edges, const int *edges, const float *length_cdf,
                  unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal)
{
    const auto num = static_cast<int>(num_samples);
    cuda::parallel_for(num, sample_edges_op{
            nodes, num_edges, edges, length_cdf,
            num_samples, samples, node_id, lerp, normal
    });
}

void sample_edges_and_corners(unsigned long num_nodes, const float *nodes, float stroke_width,
                              unsigned long num_edges, const int *edges, const float *length_cdf, const float *corner_normals, const float *angles,
                              unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal, float *position)
{
    const auto num = static_cast<int>(num_samples);
    cuda::parallel_for(num, sample_edges_and_corners_op{
            num_nodes, nodes, stroke_width,
            num_edges, edges, length_cdf, corner_normals, angles,
            num_samples, samples, node_id, lerp, normal, position
    });
}

} // namespace cuda
} // namespace wiregrad




