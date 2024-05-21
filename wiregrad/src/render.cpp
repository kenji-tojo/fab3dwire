#include "render.h"

#include <cstdio>
#include <numeric>

#include "thread.h"


namespace wiregrad {

#if defined(WIREGRAD_CUDA)
namespace cuda {

extern void render_filled_polygons(const FilledPolygons &polygons,
                                   unsigned long num_samples, const float *samples,
                                   int width, int height, float *weight_sum, float *contrib_sum,
                                   const RenderOptions &options);

extern void d_render_filled_polygons(int width, int height, const float *d_image,
                                     const FilledPolygons &polygons,
                                     float length_sum, unsigned long num_samples,
                                     const int *node_id, const float *lerp, const float *normal,
                                     float *d_nodes, const RenderOptions &options);

extern void render_polylines(const PolyLines &polylines,
                             unsigned long num_samples, const float *samples,
                             int width, int height, float *weight_sum, float *contrib_sum,
                             const RenderOptions &options);

extern void d_render_polylines(int width, int height, const float *d_image, const PolyLines &polylines, float length_sum,
                               unsigned long num_samples, const int *node_id, const float *lerp, const float *normal, const float *position,
                               float *d_nodes, float *d_stroke_width,
                               const RenderOptions &options);

} // namespace cuda
#endif

namespace cpu {

template<typename RenderFunc>
void render(const typename RenderFunc::Primitive &primitive,
            unsigned long num_samples, const float *samples,
            int width, int height, float *weight_sum, float *contrib_sum,
            const RenderOptions &options)
{
    using namespace Eigen;

    const auto num_threads = helper::max(1, options.num_cpu_threads);
    const auto num_tiles = num_threads;
    const auto tile_height = height / num_tiles + 1;
    const auto tile_margin = gaussian_filter_weight_op::filter_radius;
    const auto tile_vertical_size = tile_height + 2 * tile_margin;

    Matrix<float, Dynamic, Dynamic, RowMajor> tile_weight_sum, tile_contrib_sum;
    tile_weight_sum.setZero(num_tiles, tile_vertical_size * width);
    tile_contrib_sum.setZero(num_tiles, tile_vertical_size * width * 3);

    std::vector<ImageTile> image_tiles;
    image_tiles.reserve(num_tiles);

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        image_tiles.emplace_back(width, height,
                                 tile_id, tile_height, tile_margin,
                                 tile_weight_sum.row(tile_id).data(),
                                 tile_contrib_sum.row(tile_id).data());
        ASSERT(image_tiles.back().tile_vertical_size() == tile_vertical_size);
    }

    std::mutex image_mutex;

    cpu::parallel_for(num_tiles, [&] (int idx, int tid) {
        const auto &tile = image_tiles[idx];
        RenderFunc func{ primitive, num_samples, samples, tile, options };

        const auto ih_bgn = helper::min<int>(tile_height * (idx + 0), height);
        const auto ih_end = helper::min<int>(tile_height * (idx + 1), height);

        for (int ih = ih_bgn; ih < ih_end; ++ih)
            for (int iw = 0; iw < width; ++iw)
                func(iw, ih);

        std::lock_guard guard(image_mutex);
        tile.merge(weight_sum, contrib_sum);

    }, num_threads);
}

} // namespace cpu

void render_filled_polygons(const FilledPolygons &polygons,
                            unsigned long num_samples, const float *samples,
                            int width, int height, float *weight_sum, float *contrib_sum,
                            const RenderOptions &options,
                            bool is_cuda)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::render_filled_polygons(polygons, num_samples, samples, width, height, weight_sum, contrib_sum, options);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else
        cpu::render<render_filled_polygons_op>(polygons, num_samples, samples, width, height, weight_sum, contrib_sum, options);
}


void d_render_filled_polygons(int width, int height, const float *d_image,
                              const FilledPolygons &polygons,
                              float length_sum, unsigned long num_samples,
                              const int *node_id, const float *lerp, const float *normal,
                              float *d_nodes,
                              const RenderOptions &options,
                              bool is_cuda)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::d_render_filled_polygons(width, height, d_image, polygons,
                                       length_sum, num_samples, node_id, lerp, normal,
                                       d_nodes, options);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        using namespace Eigen;

        const auto num_nodes = std::accumulate(polygons.num_nodes, polygons.num_nodes + polygons.num_polygons, 0, std::plus<>{});
        const auto num_threads = helper::max(1, options.num_cpu_threads);
        Matrix<float, Dynamic, Dynamic, RowMajor> d_nodes_buffer;
        d_nodes_buffer.setZero(num_threads, num_nodes * 3);

        std::vector<d_render_filled_polygons_op> funcs;
        funcs.reserve(num_threads);

        for (int tid = 0; tid < num_threads; ++tid)
            funcs.emplace_back(width, height, d_image, polygons,
                               length_sum, num_samples, node_id, lerp, normal,
                               d_nodes_buffer.row(tid).data(),
                               options);

        const auto num = static_cast<int>(num_samples);
        cpu::parallel_for(num, [&] (int idx, int tid) { funcs[tid](num, idx); }, num_threads);

        Map<VectorXf>{ d_nodes, d_nodes_buffer.cols() } += d_nodes_buffer.colwise().sum();
    }
}

void render_polylines(const PolyLines &polylines,
                      unsigned long num_samples, const float *samples,
                      int width, int height, float *weight_sum, float *contrib_sum,
                      const RenderOptions &options,
                      bool is_cuda)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::render_polylines(polylines, num_samples, samples, width, height, weight_sum, contrib_sum, options);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else
        cpu::render<render_polylines_op>(polylines, num_samples, samples, width, height, weight_sum, contrib_sum, options);
}

void d_render_polylines(int width, int height, const float *d_image, const PolyLines &polylines, float length_sum,
                        unsigned long num_samples, const int *node_id, const float *lerp, const float *normal, const float *position,
                        float *d_nodes, float *d_stroke_width,
                        const RenderOptions &options, bool is_cuda)
{
    if (is_cuda) {
#if defined(WIREGRAD_CUDA)
        cuda::d_render_polylines(width, height, d_image, polylines, length_sum,
                                 num_samples, node_id, lerp, normal, position,
                                 d_nodes, d_stroke_width, options);
#else
        fprintf(stderr, "CUDA is not supported\n");
#endif
    }
    else {
        using namespace Eigen;

        const auto num_nodes = std::accumulate(polylines.num_nodes, polylines.num_nodes + polylines.num_polylines, 0, std::plus<>{});
        const auto num_threads = helper::max(1, options.num_cpu_threads);
        Matrix<float, Dynamic, Dynamic, RowMajor> d_nodes_buffer;
        d_nodes_buffer.setZero(num_threads, num_nodes * 3);

        std::vector<d_render_polylines_op> funcs;
        funcs.reserve(num_threads);

        for (int tid = 0; tid < num_threads; ++tid)
            funcs.emplace_back(width, height, d_image, polylines, length_sum,
                               num_samples, node_id, lerp, normal, position,
                               d_nodes_buffer.row(tid).data(), d_stroke_width,
                               options);

        const auto num = static_cast<int>(num_samples);
        cpu::parallel_for(num, [&] (int idx, int tid) { funcs[tid](num, idx); }, num_threads);

        Map<VectorXf>{ d_nodes, d_nodes_buffer.cols() } += d_nodes_buffer.colwise().sum();
    }
}

} // namespace wiregrad





