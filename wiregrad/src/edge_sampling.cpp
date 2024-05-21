#include "edge_sampling.h"

#include <cstdio>

#include "thread.h"


namespace wiregrad {

#if defined(WIREGRAD_CUDA)
namespace cuda {

extern void sample_edges(const float *nodes, unsigned long num_edges, const int *edges, const float *length_cdf,
                         unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal);

extern void sample_edges_and_corners(unsigned long num_nodes, const float *nodes, float stroke_width,
                                     unsigned long num_edges, const int *edges, const float *length_cdf, const float *corner_normals, const float *angles,
                                     unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal, float *position);

} // namespace cuda
#endif
void sample_edges(const float *nodes, unsigned long num_edges, const int *edges, const float *length_cdf,
                  unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal,
                  bool is_cuda)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::sample_edges(nodes, num_edges, edges, length_cdf,
                           num_samples, samples, node_id, lerp, normal);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        sample_edges_op func{
                nodes, num_edges, edges, length_cdf,
                num_samples, samples, node_id, lerp, normal
        };

        const auto num = static_cast<int>(num_samples);
        cpu::parallel_for(num, [&] (int idx, int tid) { func(num, idx); });
    }
}

void sample_edges_and_corners(unsigned long num_nodes, const float *nodes, float stroke_width,
                              unsigned long num_edges, const int *edges, const float *length_cdf, const float *corner_normals, const float *angles,
                              unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal, float *position,
                              bool is_cuda)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::sample_edges_and_corners(num_nodes, nodes, stroke_width,
                                       num_edges, edges, length_cdf, corner_normals, angles,
                                       num_samples, samples, node_id, lerp, normal, position);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        sample_edges_and_corners_op func{
                num_nodes, nodes, stroke_width,
                num_edges, edges, length_cdf, corner_normals, angles,
                num_samples, samples, node_id, lerp, normal, position
        };

        const auto num = static_cast<int>(num_samples);
        cpu::parallel_for(num, [&] (int idx, int tid) { func(num, idx); });
    }
}

} // namespace wiregrad


