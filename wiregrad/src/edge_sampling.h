#pragma once

#include "common.h"

#include <Eigen/Core>


namespace wiregrad {

template<typename T> HOST_DEVICE inline
int sample_cdf(int num, const T *cdf /* cdf[num + 1] */, const T u) {
    auto first = 0;
    auto last = num;

    while (last - first > 1) {
        const auto middle = first + (last - first) / 2;
        ASSERT(middle > first && middle < last);

        if (cdf[middle] >= u)
            last = middle;
        else
            first = middle;
    }

    return first;
}

void sample_edges(const float *nodes, unsigned long num_edges, const int *edges, const float *length_cdf,
                  unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal,
                  bool is_cuda);

struct sample_edges_op {
public:
    const float *const nodes;
    const int num_edges;
    const int *const edges;
    const float *const length_cdf;
    const int num_samples;
    const float *const samples;
    int *const node_id;
    float *const lerp;
    float *const normal;

    sample_edges_op(const float *nodes, unsigned long num_edges, const int *edges, const float *length_cdf,
                    unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal)
            : nodes{ nodes }, num_edges{ static_cast<int>(num_edges) }, edges{ edges }, length_cdf{ length_cdf }
            , num_samples{ static_cast<int>(num_samples) }, samples{ samples }, node_id{ node_id }, lerp{ lerp }, normal{ normal } { }

    HOST_DEVICE inline
    void operator()(int num, int idx) const
    {
        using namespace Eigen;

        if (idx >= helper::min(num, num_samples))
            return;

        const auto u = samples[idx];
        const auto edge_id = sample_cdf(num_edges, length_cdf, u);

        ASSERT(edge_id >= 0 && edge_id < num_edges);
        ASSERT(length_cdf[edge_id] < length_cdf[edge_id + 1]);
        ASSERT(u >= length_cdf[edge_id] && u <= length_cdf[edge_id + 1]);

        const auto ip0 = edges[edge_id * 2 + 0];
        const auto ip1 = edges[edge_id * 2 + 1];
        const Map<const Vector2f> p0{ nodes + ip0 * 2 };
        const Map<const Vector2f> p1{ nodes + ip1 * 2 };

        node_id[idx * 2 + 0] = ip0;
        node_id[idx * 2 + 1] = ip1;
        lerp[idx] = (u - length_cdf[edge_id]) / (length_cdf[edge_id + 1] - length_cdf[edge_id]);
        Map<Vector2f> n{ normal + idx * 2 };

        const Vector2f e = (p1 - p0).normalized();
        n << -e[1], e[0];
    }

};

void sample_edges_and_corners(unsigned long num_nodes, const float *nodes, float stroke_width,
                              unsigned long num_edges, const int *edges, const float *length_cdf, const float *corner_normals, const float *angles,
                              unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal, float *position,
                              bool is_cuda);

struct sample_edges_and_corners_op {
public:
    const int num_nodes;
    const float *const nodes;
    const float stroke_width;
    const int num_edges;
    const int *const edges;
    const float *const length_cdf;
    const float *corner_normals;
    const float *angles;
    const int num_samples;
    const float *const samples;
    int *const node_id;
    float *const lerp;
    float *const normal;
    float *const position;

    sample_edges_and_corners_op(unsigned long num_nodes, const float *nodes, float stroke_width,
                                unsigned long num_edges, const int *edges, const float *length_cdf, const float *corner_normals, const float *angles,
                                unsigned long num_samples, const float *samples, int *node_id, float *lerp, float *normal, float *position)
            : num_nodes{ static_cast<int>(num_nodes) }, nodes{ nodes }, stroke_width{ stroke_width }
            , num_edges{ static_cast<int>(num_edges) }, edges{ edges }, length_cdf{ length_cdf }, corner_normals{ corner_normals }, angles{ angles }
            , num_samples{ static_cast<int>(num_samples) }, samples{ samples }, node_id{ node_id }, lerp{ lerp }, normal{ normal }, position{ position } { }

    HOST_DEVICE inline
    void operator()(int num, int idx) const
    {
        using namespace Eigen;

        if (idx >= helper::min(num, num_samples))
            return;

        auto u = samples[idx];
        const auto num_segments = num_edges * 2 + num_nodes;
        const auto segment_id = sample_cdf(num_segments, length_cdf, u);

        ASSERT(segment_id >= 0 && segment_id < num_segments);
        ASSERT(length_cdf[segment_id] < length_cdf[segment_id + 1]);
        ASSERT(u >= length_cdf[segment_id] && u <= length_cdf[segment_id + 1]);

        u = (u - length_cdf[segment_id]) / (length_cdf[segment_id + 1] - length_cdf[segment_id]);

        if (segment_id < num_edges * 2) {
            const auto edge_id = segment_id % num_edges;

            const auto ip0 = edges[edge_id * 2 + 0];
            const auto ip1 = edges[edge_id * 2 + 1];

            node_id[idx * 2 + 0] = ip0;
            node_id[idx * 2 + 1] = ip1;
            lerp[idx] = u;
            Map<Vector2f> n{ normal + idx * 2 };
            Map<Vector2f> p{ position + idx * 2 };

            const Map<const Vector2f> p0{ nodes + ip0 * 2 };
            const Map<const Vector2f> p1{ nodes + ip1 * 2 };
            Vector2f e = (p1 - p0).normalized();

            if (segment_id >= num_edges)
                e *= -1.0f;

            n << -e[1], e[0];
            p = (1.0f - lerp[idx]) * p0 + lerp[idx] * p1 + 0.5f * stroke_width * n;
        }
        else {
            const auto corner_id = segment_id - num_edges * 2;
            ASSERT(corner_id >= 0 && corner_id < num_nodes);

            node_id[idx * 2 + 0] = corner_id;
            node_id[idx * 2 + 1] = corner_id;
            lerp[idx] = 0.0f;
            Map<Vector2f> n{ normal + idx * 2 };
            Map<Vector2f> p{ position + idx * 2 };

            const Map<const Vector2f> p0{ nodes + corner_id * 2 };
            const Map<const Vector2f> ex{ corner_normals + corner_id * 2 };
            const Vector2f ey{ -ex[1], ex[0] };

            const auto phi = (u - 0.5f) * angles[corner_id];
            const auto cos_phi = std::cos(phi);
            const auto sin_phi = std::sin(phi);

            n = cos_phi * ex + sin_phi * ey;
            p = p0 + 0.5f * stroke_width * n;
        }
    }

};

} // namespace wiregrad


