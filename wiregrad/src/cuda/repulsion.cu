#include "repulsion.h"

#include "common.cuh"


namespace wiregrad {
namespace cuda {

void repulsion(unsigned long num_nodes, const float *nodes,
               float *energy, float *d_nodes,
               bool cyclic, const RepulsionParameters &parameters)
{
    const auto num = static_cast<int>(num_nodes);
    cuda::parallel_for(num, repulsion_bruteforce_op{ num_nodes, nodes, energy, d_nodes, cyclic, parameters });
}

void repulsion(unsigned long num_nodes, const float *nodes,
               unsigned long num_levels, const int *splits, const int *prims, const float *box_points,
               float *energy, float *d_nodes,
               bool cyclic, const RepulsionParameters &parameters)
{
    const auto num = static_cast<int>(num_nodes);
    cuda::parallel_for(num, repulsion_accelerated_op{
            num_nodes, nodes, num_levels, splits, prims, box_points,
            energy, d_nodes, cyclic, parameters
    });
}

} // namespace cuda
} // namespace wiregrad



