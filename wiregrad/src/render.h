#pragma once

#include "common.h"
#include "filled_polygons.h"
#include "polylines.h"
#include "antialias.h"
#include "image_tile.h"

#include <Eigen/Core>
#include <Eigen/Dense>


namespace wiregrad {

struct RenderOptions {
    int num_cpu_threads = 20;
    float background[3]{ 1.0f, 1.0f, 1.0f };
};

void render_filled_polygons(const FilledPolygons &polygons,
                            unsigned long num_samples, const float *samples,
                            int width, int height, float *weight_sum, float *contrib_sum,
                            const RenderOptions &options,
                            bool is_cuda = false);

void render_polylines(const PolyLines &polylines,
                      unsigned long num_samples, const float *samples,
                      int width, int height, float *weight_sum, float *contrib_sum,
                      const RenderOptions &options,
                      bool is_cuda = false);

template<typename Primitive_>
struct render_primitive_op {
public:
    using Primitive = Primitive_;

    const Primitive_ primitve;
    const int num_samples;
    const float *samples;
    const ImageTile tile;
    const RenderOptions options;

    render_primitive_op(Primitive_ primitve,
                        unsigned long num_samples, const float *samples,
                        ImageTile tile, RenderOptions options)
            : primitve{ primitve }
            , num_samples{ static_cast<int>(num_samples) }, samples{ samples }
            , tile{ tile }, options{ options } { }

    HOST_DEVICE inline
    void operator()(int iw, int ih)
    {
        for (int ip = 0; ip < num_samples; ++ip) {
            gaussian_filter_weight_op filter_weight_func;

            auto &p = filter_weight_func.p;
            p[0] = static_cast<float>(iw) + samples[ip * 2 + 0];
            p[1] = static_cast<float>(ih) + samples[ip * 2 + 1];

            float rgb[3];

            if (primitve.intersect(p))
                copy3(primitve.color, rgb);
            else
                copy3(options.background, rgb);

            tile.accumulate(iw, ih, rgb, filter_weight_func);
        }
    }
};

using render_filled_polygons_op = render_primitive_op<FilledPolygons>;
using render_polylines_op = render_primitive_op<PolyLines>;

void d_render_filled_polygons(int width, int height, const float *d_image,
                              const FilledPolygons &polygons,
                              float length_sum, unsigned long num_samples,
                              const int *node_id, const float *lerp, const float *normal,
                              float *d_nodes,
                              const RenderOptions &options,
                              bool is_cuda);

struct d_render_filled_polygons_op {
public:
    const int width;
    const int height;
    const float *d_image;
    const FilledPolygons polygons;
    const float length_sum;
    const int num_samples;
    const int *const node_id;
    const float *const lerp;
    const float *const normal;
    float *const d_nodes;
    const RenderOptions options;

    d_render_filled_polygons_op(int width, int height, const float *d_image, const FilledPolygons &polygons,
                                float length_sum, unsigned long num_samples,
                                const int *node_id, const float *lerp, const float *normal,
                                float *d_nodes,
                                const RenderOptions &options)
            : width{ width }, height{ height }, d_image{ d_image }, polygons{ polygons }
            , length_sum{ length_sum }, num_samples{ static_cast<int>(num_samples) }
            , node_id{ node_id }, lerp{ lerp }, normal{ normal }
            , d_nodes{ d_nodes }, options{ options } { }

    HOST_DEVICE inline
    void operator()(int num, int idx)
    {
        using namespace Eigen;
        constexpr auto eps = 1e-4f;

        if (idx >= helper::min(num, num_samples))
            return;

        const auto pdf = 1.0f / length_sum;
        const auto sample_weight = 1.0f / (static_cast<float>(num_samples) * pdf);
        ASSERT(std::isfinite(sample_weight));

        const auto ip0 = node_id[idx * 2 + 0];
        const auto ip1 = node_id[idx * 2 + 1];
        const Map<const Vector2f> p0{ polygons.nodes + ip0 * 2 };
        const Map<const Vector2f> p1{ polygons.nodes + ip1 * 2 };

        const auto t = lerp[idx];
        const Map<const Vector2f> n{ normal + idx * 2 };

        gaussian_filter_weight_op filter_weight_func;
        Map<Vector2f> p{ filter_weight_func.p };

        p = (1.0f - t) * p0 + t * p1;
        const Vector2f q0 = p - eps * n;
        const Vector2f q1 = p + eps * n;

        Vector3f c0, c1;
        copy3(options.background, c0.data());
        copy3(options.background, c1.data());

        if (polygons.intersect(q0.data()))
            copy3(polygons.color, c0.data());

        if (polygons.intersect(q1.data()))
            copy3(polygons.color, c1.data());

        const auto iw = helper::clip(static_cast<int>(std::floor(p[0])), 0, width - 1);
        const auto ih = helper::clip(static_cast<int>(std::floor(p[1])), 0, height - 1);

        const auto filter_radius = decltype(filter_weight_func)::filter_radius;
        const auto w_lo = helper::max(iw - filter_radius, 0);
        const auto w_hi = helper::max(iw + filter_radius, width - 1);
        const auto h_lo = helper::max(ih - filter_radius, 0);
        const auto h_hi = helper::max(ih + filter_radius, height - 1);

        const Vector3f c01 = c0 - c1;

        auto weight_sum = 0.0f;

        for (int jh = h_lo; jh <= h_hi; ++jh)
            for (int jw = w_lo; jw <= w_hi; ++jw) {
                const auto filter_weight = sample_weight * filter_weight_func(jw, jh);
                const auto pixel_id = width * jh + jw;
                weight_sum += filter_weight * dot3(d_image + pixel_id * 3, c01.data());
            }

        ATOMIC_ADD(d_nodes, ip0 * 2 + 0, weight_sum * (1.0f - t) * n[0]);
        ATOMIC_ADD(d_nodes, ip0 * 2 + 1, weight_sum * (1.0f - t) * n[1]);
        ATOMIC_ADD(d_nodes, ip1 * 2 + 0, weight_sum * t * n[0]);
        ATOMIC_ADD(d_nodes, ip1 * 2 + 1, weight_sum * t * n[1]);
    }

};

void d_render_polylines(int width, int height, const float *d_image, const PolyLines &polylines, float length_sum,
                        unsigned long num_samples, const int *node_id, const float *lerp, const float *normal, const float *position,
                        float *d_nodes, float *d_stroke_width,
                        const RenderOptions &options, bool is_cuda);

struct d_render_polylines_op {
public:
    const int width;
    const int height;
    const float *d_image;
    const PolyLines polylines;
    const float length_sum;
    const int num_samples;
    const int *const node_id;
    const float *const lerp;
    const float *const normal;
    const float *const position;
    float *const d_nodes;
    float *const d_stroke_width;
    const RenderOptions options;

    d_render_polylines_op(int width, int height, const float *d_image, const PolyLines &polylines, float length_sum,
                          unsigned long num_samples, const int *node_id, const float *lerp, const float *normal, const float *position,
                          float *d_nodes, float *d_stroke_width, const RenderOptions &options)
            : width{ width }, height{ height }, d_image{ d_image }, polylines{ polylines }, length_sum{ length_sum }
            , num_samples{ static_cast<int>(num_samples) }, node_id{ node_id }, lerp{ lerp }, normal{ normal }, position{ position }
            , d_nodes{ d_nodes }, d_stroke_width{d_stroke_width}, options{ options } { }

    HOST_DEVICE inline
    void operator()(int num, int idx)
    {
        using namespace Eigen;
        constexpr auto eps = 1e-4f;

        if (idx >= helper::min(num, num_samples))
            return;

        const auto pdf = 1.0f / length_sum;
        const auto sample_weight = 1.0f / (static_cast<float>(num_samples) * pdf);
        ASSERT(std::isfinite(sample_weight));

        const auto ip0 = node_id[idx * 2 + 0];
        const auto ip1 = node_id[idx * 2 + 1];

        const auto t = lerp[idx];
        const Map<const Vector2f> n{ normal + idx * 2 };

        gaussian_filter_weight_op filter_weight_func;
        Map<Vector2f> p{ filter_weight_func.p };

        p << position[idx * 2 + 0], position[idx * 2 + 1];
        const Vector2f q = p + eps * n;

        const Map<const Vector3f> c0{ polylines.color };
        Vector3f c1;

        if (polylines.intersect(q.data()))
            copy3(polylines.color, c1.data());
        else
            copy3(options.background, c1.data());

        const auto iw = helper::clip(static_cast<int>(std::floor(p[0])), 0, width - 1);
        const auto ih = helper::clip(static_cast<int>(std::floor(p[1])), 0, height - 1);

        const auto filter_radius = decltype(filter_weight_func)::filter_radius;
        const auto w_lo = helper::max(iw - filter_radius, 0);
        const auto w_hi = helper::max(iw + filter_radius, width - 1);
        const auto h_lo = helper::max(ih - filter_radius, 0);
        const auto h_hi = helper::max(ih + filter_radius, height - 1);

        const Vector3f c01 = c0 - c1;

        auto weight_sum = 0.0f;

        for (int jh = h_lo; jh <= h_hi; ++jh)
            for (int jw = w_lo; jw <= w_hi; ++jw) {
                const auto filter_weight = sample_weight * filter_weight_func(jw, jh);
                const auto pixel_id = width * jh + jw;
                weight_sum += filter_weight * dot3(d_image + pixel_id * 3, c01.data());
            }

        ATOMIC_ADD(d_nodes, ip0 * 2 + 0, weight_sum * (1.0f - t) * n[0]);
        ATOMIC_ADD(d_nodes, ip0 * 2 + 1, weight_sum * (1.0f - t) * n[1]);
        ATOMIC_ADD(d_nodes, ip1 * 2 + 0, weight_sum * t * n[0]);
        ATOMIC_ADD(d_nodes, ip1 * 2 + 1, weight_sum * t * n[1]);

        if (d_stroke_width != nullptr)
            d_stroke_width[idx] += 0.5f * weight_sum;
    }

};

} // namespace wiregrad


