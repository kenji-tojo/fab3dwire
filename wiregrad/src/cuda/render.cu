#include "render.h"

#include "common.cuh"


namespace wiregrad {
namespace cuda {

namespace kernel {

template<typename RenderFunc> __global__
void render(RenderFunc func)
{
    const auto tid = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);

    const ImageTile &tile = func.tile;
    const auto width = tile.width;
    const auto height = tile.height;
    const auto num_pixels = height * width;

    if (tid >= num_pixels)
        return;

    const auto iw = tid % width;
    const auto ih = tid / width;

    func(iw, ih);
}

} // kernel

template<typename RenderFunc>
void render(const typename RenderFunc::Primitive &primitive,
            unsigned long num_samples,
            const float *samples,
            const ImageTile &tile,
            const RenderOptions &options)
{
    const auto num_pixels = tile.height * tile.width;
    constexpr auto num_threads = 512;
    const auto num_blocks = num_pixels / num_threads + 1;

    kernel::render<<<num_blocks, num_threads>>>(RenderFunc{ primitive, num_samples, samples, tile, options });
}

void render_filled_polygons(const FilledPolygons &polygons,
                            unsigned long num_samples, const float *samples,
                            int width, int height, float *weight_sum, float *contrib_sum,
                            const RenderOptions &options)
{
    const ImageTile tile{ width, height, 0, height, 0, weight_sum, contrib_sum };
    cuda::render<render_filled_polygons_op>(polygons, num_samples, samples, tile, options);
}

void d_render_filled_polygons(int width, int height, const float *d_image,
                              const FilledPolygons &polygons,
                              float length_sum, unsigned long num_samples,
                              const int *node_id, const float *lerp, const float *normal,
                              float *d_nodes,
                              const RenderOptions &options)
{
    const auto num = static_cast<int>(num_samples);
    cuda::parallel_for(num, d_render_filled_polygons_op{
            width, height, d_image, polygons,
            length_sum, num_samples, node_id, lerp, normal,
            d_nodes, options
    });
}

void render_polylines(const PolyLines &polylines,
                      unsigned long num_samples, const float *samples,
                      int width, int height, float *weight_sum, float *contrib_sum,
                      const RenderOptions &options)
{
    const ImageTile tile{ width, height, 0, height, 0, weight_sum, contrib_sum };
    cuda::render<render_polylines_op>(polylines, num_samples, samples, tile, options);
}

void d_render_polylines(int width, int height, const float *d_image, const PolyLines &polylines, float length_sum,
                        unsigned long num_samples, const int *node_id, const float *lerp, const float *normal, const float *position,
                        float *d_nodes, float *d_stroke_width,
                        const RenderOptions &options)
{
    const auto num = static_cast<int>(num_samples);
    cuda::parallel_for(num, d_render_polylines_op{
            width, height, d_image, polylines, length_sum,
            num_samples, node_id, lerp, normal, position,
            d_nodes, d_stroke_width, options
    });
}

} // namespace cuda
} // namespace wiregrad


