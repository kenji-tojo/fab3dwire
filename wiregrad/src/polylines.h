#pragma once

#include "common.h"
#include "intersect.h"
#include "line_hierarchy2d.h"


namespace wiregrad {

class PolyLines {
public:
    const int num_polylines;
    const int *const num_nodes;
    const float *nodes;
    const int *const num_levels;
    const int *const splits;
    const int *const total_boxes;
    const int *const prims;
    const float *const bounds;
    const bool cyclic;
    float stroke_width = 2.0f;
    float color[3]{ 0.7f, 0.3f, 0.3f };

    PolyLines(unsigned long num_polylines,
              const int *num_nodes, const float *nodes,
              const int *num_levels, const int *splits,
              const int *total_boxes, const int *prims, const float *bounds,
              bool cyclic)
            : num_polylines{ static_cast<int>(num_polylines) }
            , num_nodes{ num_nodes }, nodes{ nodes }
            , num_levels{ num_levels }, splits{ splits }
            , total_boxes{ total_boxes }, prims{ prims }, bounds{ bounds }
            , cyclic{ cyclic } { }

    HOST_DEVICE inline
    bool intersect(const float point[2]) const;

};

struct intersect_polyline_op{
public:
    const float point[2];
    const int num_nodes;
    const float *const nodes;
    const float stroke_width;
    const bool cyclic;

    HOST_DEVICE inline
    intersect_polyline_op(const float point[2], int num_nodes, const float *nodes,
                          float stroke_width, bool cyclic)
            : point{ point[0], point[1] }, num_nodes{ num_nodes }, nodes{ nodes }
            , stroke_width{ stroke_width }, cyclic{ cyclic } { }

    HOST_DEVICE inline
    bool operator()(int prim_id)
    {
        ASSERT(prim_id >= 0 && prim_id < num_nodes);

        if (!cyclic && prim_id == num_nodes - 1)
            return false;

        const auto ip0 = prim_id;
        const auto ip1 = (ip0 + 1) % num_nodes;
        const float *const p0 = nodes + ip0 * 2;
        const float *const p1 = nodes + ip1 * 2;
        float t;
        return intersect_point_round_line(point, p0, p1, stroke_width, t);
    }
};

HOST_DEVICE inline
bool PolyLines::intersect(const float point[2]) const
{
    auto node_offset = 0;
    auto tree_offset = 0;
    auto data_offset = 0;

    for (int i = 0; i < num_polylines; ++i) {
        intersect_polyline_op func{ point, num_nodes[i], nodes + node_offset * 2, stroke_width, cyclic };

        if (num_levels[i] > 0) {
            if (intersect_line_hierarchy2d(point, num_levels[i], splits + tree_offset,
                                            prims + data_offset * 2, bounds + data_offset * 4,
                                            func, /*margin=*/0.5f * stroke_width))
                return true;
        }
        else  {
            const auto num_prims = cyclic ? num_nodes[i] : num_nodes[i] - 1;

            for (int ip0 = 0; ip0 < num_prims; ++ip0)
                if (func(ip0))
                    return true;
        }

        node_offset += num_nodes[i];
        tree_offset += num_levels[i];
        data_offset += total_boxes[i];
    }

    return false;
}

} // namespace wiregrad


