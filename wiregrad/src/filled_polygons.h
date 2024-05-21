#pragma once

#include "common.h"
#include "vector.h"
#include "intersect.h"


namespace wiregrad {

class FilledPolygons {
public:
    const int num_polygons;
    const int *const num_nodes;
    const float *nodes;
    const int *const num_levels;
    const int *const splits;
    const int *const total_boxes;
    const int *const prims;
    const float *const bounds;
    float color[3]{ 0.7f, 0.3f, 0.3f };

    FilledPolygons(unsigned long num_polygons,
                   const int *num_nodes, const float *nodes,
                   const int *num_levels, const int *splits,
                   const int *total_boxes, const int *prims, const float *bounds)
            : num_polygons{ static_cast<int>(num_polygons) }
            , num_nodes{ num_nodes }, nodes{ nodes }
            , num_levels{ num_levels }, splits{ splits }
            , total_boxes{ total_boxes }, prims{ prims }, bounds{ bounds } { }

    HOST_DEVICE inline
    bool intersect(const float point[2], float thresh = 0.99f) const;

};

template<typename T> HOST_DEVICE inline
void winding_number(const T point[2],
                    const T p0[2],
                    const T p1[2],
                    T &sum)
{
    T c0[2], c1[2];
    sub2(p0, point, c0);
    sub2(p1, point, c1);
    sum += std::atan2(cross2(c0, c1), dot2(c0, c1));
}

template<typename T> HOST_DEVICE inline
auto integrate_winding_number_bruteforce(const T point[2], int num_nodes, const T *nodes) -> T
{
    auto wn_sum = 0.0f;

    for (int ip0 = 0; ip0 < num_nodes; ++ip0) {
        const auto ip1 = (ip0 + 1) % num_nodes;
        const float *const p0 = nodes + ip0 * 2;
        const float *const p1 = nodes + ip1 * 2;
        winding_number(point, p0, p1, wn_sum);
    }

    return wn_sum / TwoPi<float>;
}

template<typename T> HOST_DEVICE inline
auto integrate_winding_number(const T point[2],
                              int num_nodes, const T *nodes,
                              int num_levels, const int *splits,
                              const int *prims, const T *bounds) -> T
{
    auto wn_sum = 0.0f;

    constexpr auto stack_size = 16;
    num_levels = helper::min(num_levels, stack_size);

    int base[stack_size];
    int index[stack_size];
    int offset[stack_size];

    base[0] = 0;
    {
        auto boxes = splits[0];
        for (int i = 1; i < num_levels; ++i) {
            base[i] = base[i - 1] + boxes;
            boxes *= splits[i];
        }
    }

    index[0] = 0;
    offset[0] = 0;

    auto level = 0;

    while (level > 0 || index[0] < splits[0]) {
        if (level > 0 && index[level] >= splits[level]) {
            level -= 1;
            continue;
        }

        const auto address = base[level] + offset[level] + index[level];

        if (intersect_point_aabb2d(point, bounds + address * 4, /*margin=*/0.0f)) {
            if (level < num_levels - 1) {
                level += 1;
                offset[level] = splits[level] * (offset[level - 1] + index[level - 1]);
                index[level - 1] += 1;
                index[level] = 0;
                continue;
            }

            const auto bgn = prims[address * 2 + 0];
            const auto end = prims[address * 2 + 1];

            for (int ip0 = bgn; ip0 < end; ++ip0) {
                const auto ip1 = (ip0 + 1) % num_nodes;
                const float *const p0 = nodes + ip0 * 2;
                const float *const p1 = nodes + ip1 * 2;
                winding_number(point, p0, p1, wn_sum);
            }
        }
        else {
            const auto ip0 = prims[address * 2 + 0] % num_nodes;
            const auto ip1 = prims[address * 2 + 1] % num_nodes;
            const float *const p0 = nodes + ip0 * 2;
            const float *const p1 = nodes + ip1 * 2;
            winding_number(point, p0, p1, wn_sum);
        }

        index[level] += 1;
    }

    return wn_sum / TwoPi<float>;
}

HOST_DEVICE inline
bool FilledPolygons::intersect(const float point[2], float thresh) const
{
    auto node_offset = 0;
    auto tree_offset = 0;
    auto data_offset = 0;

    for (int i = 0; i < num_polygons; ++i) {
        float wn;

        if (num_levels[i] > 0)
            wn = integrate_winding_number(point, num_nodes[i], nodes + node_offset * 2,
                                          num_levels[i], splits + tree_offset,
                                          prims + data_offset * 2, bounds + data_offset * 4);
        else
            wn = integrate_winding_number_bruteforce(point, num_nodes[i], nodes + node_offset * 2);

        if (std::abs(wn) > thresh)
            return true;

        node_offset += num_nodes[i];
        tree_offset += num_levels[i];
        data_offset += total_boxes[i];
    }

    return false;
}

} // namespace wiregrad


