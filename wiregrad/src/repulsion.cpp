#include "repulsion.h"

#include <cstdio>
#include <vector>

#include "thread.h"


namespace wiregrad {

#if defined(WIREGRAD_CUDA)
namespace cuda {

extern void repulsion(unsigned long num_nodes, const float *nodes,
                      float *energy, float *d_nodes,
                      bool cyclic,
                      const RepulsionParameters &parameters);

extern void repulsion(unsigned long num_nodes, const float *nodes,
                      unsigned long num_levels, const int *splits,
                      const int *prims, const float *box_points,
                      float *energy, float *d_nodes,
                      bool cyclic,
                      const RepulsionParameters &parameters);

} // namespace cuda
#endif

void repulsion(unsigned long num_nodes, const float *nodes,
               float *energy, float *d_nodes,
               bool cyclic, bool is_cuda,
               const RepulsionParameters &parameters)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::repulsion(num_nodes, nodes, energy, d_nodes, cyclic, parameters);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        using namespace Eigen;

        Matrix<float, Dynamic, Dynamic, RowMajor> d_nodes_buffer;

        const auto num_threads = helper::max(1, parameters.num_cpu_threads);
        d_nodes_buffer.setZero(num_threads, static_cast<long>(num_nodes * 3));

        std::vector<repulsion_bruteforce_op> funcs;
        funcs.reserve(num_threads);

        for (int tid = 0; tid < num_threads; ++tid)
            funcs.emplace_back(num_nodes, nodes,
                               energy, d_nodes_buffer.row(tid).data(),
                               cyclic, parameters);

        const auto num = static_cast<int>(num_nodes);
        cpu::parallel_for(num, [&] (int idx, int tid) { funcs[tid](num, idx); }, num_threads);

        Map<VectorXf>{ d_nodes, d_nodes_buffer.cols() } += d_nodes_buffer.colwise().sum();
    }
}

void repulsion(unsigned long num_nodes, const float *nodes,
               unsigned long num_levels, const int *splits,
               const int *prims, const float *box_points,
               float *energy, float *d_nodes,
               bool cyclic, bool is_cuda,
               const RepulsionParameters &parameters)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::repulsion(num_nodes, nodes, num_levels, splits, prims, box_points,
                        energy, d_nodes, cyclic, parameters);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        using namespace Eigen;

        Matrix<float, Dynamic, Dynamic, RowMajor> d_nodes_buffer;

        const auto num_threads = helper::max(1, parameters.num_cpu_threads);
        d_nodes_buffer.setZero(num_threads, static_cast<long>(num_nodes * 3));

        std::vector<repulsion_accelerated_op> funcs;
        funcs.reserve(num_threads);

        for (int tid = 0; tid < num_threads; ++tid)
            funcs.emplace_back(num_nodes, nodes,
                               num_levels, splits, prims, box_points,
                               energy, d_nodes_buffer.row(tid).data(),
                               cyclic, parameters);

        const auto num = static_cast<int>(num_nodes);
        cpu::parallel_for(num, [&] (int idx, int tid) { funcs[tid](num, idx); }, num_threads);

        Map<VectorXf>{ d_nodes, d_nodes_buffer.cols() } += d_nodes_buffer.colwise().sum();
    }
}

} // namespace wiregrad
