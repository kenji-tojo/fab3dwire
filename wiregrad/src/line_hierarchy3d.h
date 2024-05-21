#pragma once

#include "common.h"
#include "intersect.h"
#include "bbox.h"


namespace wiregrad {

void create_line_hierarchy3d(unsigned long num_nodes, const float *nodes,
                             unsigned long num_levels, const int *boxes,
                             unsigned long total_boxes, int *prims, float *box_points,
                             bool cyclic = true, bool is_cuda = false);

struct create_line_hierarchy3d_op {
private:
    const int num_nodes;
    const float *const nodes;
    const int num_levels;
    const int *const boxes;
    int *const prims;
    float *const box_points;
    const bool cyclic;

public:
    create_line_hierarchy3d_op(unsigned long num_nodes, const float *nodes,
                               unsigned long num_levels, const int *boxes,
                               int *prims, float *box_points,
                               bool cyclic)
            : num_nodes{ static_cast<int>(num_nodes) }, nodes{ nodes }
            , num_levels{ static_cast<int>(num_levels) }, boxes{ boxes }
            , prims{ prims }, box_points{ box_points }
            , cyclic{ cyclic } { }

    HOST_DEVICE inline
    void operator()(int num, int idx)
    {
        if (idx >= num)
            return;

        const auto num_leaves = boxes[num_levels - 1];
        const auto nodes_per_leaf = 1 + num_nodes / num_leaves;

        auto &bgn = prims[idx * 2 + 0];
        auto &end = prims[idx * 2 + 1];
        bgn = end = 0;

        auto index = idx;

        for (int i = 0; i < num_levels; ++i) {
            if (index < boxes[i]) {
                const auto nodes_per_box = (num_leaves / boxes[i]) * nodes_per_leaf;
                bgn = helper::min((index + 0) * nodes_per_box, cyclic ? num_nodes : num_nodes - 1);
                end = helper::min((index + 1) * nodes_per_box, cyclic ? num_nodes : num_nodes - 1);
                break;
            }
            index -= boxes[i];
        }

        if (bgn < end) {
            OrientedBoundingBox3D obb;
            obb.create_from_polyline(num_nodes, nodes, bgn, end);
            obb.get_box_points(box_points + idx * 24);
        }
    }

};

template<typename Func> HOST_DEVICE inline
void intersect_plane_polyline(const float center[3],
                              const float normal[3],
                              int num_nodes, const float *nodes,
                              int num_levels, const int *splits,
                              const int *prims, const float *box_points,
                              Func &&func)
{
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
        const float *const box = box_points + address * 24;

        if (intersect_plane_box(center, normal, box)) {
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
                const float *const p0 = nodes + ip0 * 3;
                const float *const p1 = nodes + ip1 * 3;

                if (intersect_plane_line(center, normal, p0, p1))
                    func(ip0);
            }
        }

        index[level] += 1;
    }
}

} // namespace wiregrad


