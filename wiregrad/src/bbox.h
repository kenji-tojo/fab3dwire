#pragma once

#include "common.h"
#include "vector.h"


namespace wiregrad {

class OrientedBoundingBox3D {
public:
    HOST_DEVICE inline
    void create_from_polyline(int num_nodes,
                              const float *nodes,
                              int begin,
                              int end);

    HOST_DEVICE inline
    void get_box_points(float box_points[24]) const;

private:
    float ex[3]{};
    float ey[3]{};
    float ez[3]{};
    float world_shift[3]{}; // translation in the world space
    float local_scale[3]{}; // box scaling in the local coordinates

};

HOST_DEVICE inline
void OrientedBoundingBox3D::create_from_polyline(int num_nodes,
                                                 const float *nodes,
                                                 int begin,
                                                 int end)
{
    ASSERT(begin >= 0 && begin < num_nodes);
    ASSERT(end > begin && end <= num_nodes);

    const float *const p0 = nodes + begin * 3;
    sub3(nodes + (end % num_nodes) * 3, p0, ez);

    float lo[3], hi[3];
    lo[0] = lo[1] = lo[2] = 0.0f;
    hi[0] = hi[1] = hi[2] = 0.0f;

    const auto norm = norm3(ez);
    hi[2] += norm;

    if (norm < 1e-15f) {
        ez[0] = ez[1] = 0.0f;
        ez[2] = 1.0f;
    }
    else
        div3(ez, norm);

    local_coords(ez, ex, ey);

    for (int ip = begin + 1; ip < end; ++ip) {
        float p[3];
        sub3(nodes + ip * 3, p0, p);
        const auto x = dot3(p, ex);
        const auto y = dot3(p, ey);
        const auto z = dot3(p, ez);
        lo[0] = helper::min(x, lo[0]);
        lo[1] = helper::min(y, lo[1]);
        lo[2] = helper::min(z, lo[2]);
        hi[0] = helper::max(x, hi[0]);
        hi[1] = helper::max(y, hi[1]);
        hi[2] = helper::max(z, hi[2]);
    }
    ASSERT(lo[0] <= hi[0]);
    ASSERT(lo[1] <= hi[1]);
    ASSERT(lo[2] <= hi[2]);

    sub3(hi, lo, local_scale);
    world_shift[0] = p0[0] + lo[0] * ex[0] + lo[1] * ey[0] + lo[2] * ez[0];
    world_shift[1] = p0[1] + lo[0] * ex[1] + lo[1] * ey[1] + lo[2] * ez[1];
    world_shift[2] = p0[2] + lo[0] * ex[2] + lo[1] * ey[2] + lo[2] * ez[2];
}

HOST_DEVICE inline
void OrientedBoundingBox3D::get_box_points(float box_points[24]) const
{
    for (unsigned int i = 0; i < 8; ++i) {
        const auto sx = static_cast<float>((i >> 0) % 2) * local_scale[0];
        const auto sy = static_cast<float>((i >> 1) % 2) * local_scale[1];
        const auto sz = static_cast<float>((i >> 2) % 2) * local_scale[2];
        box_points[i * 3 + 0] = world_shift[0] + sx * ex[0] + sy * ey[0] + sz * ez[0];
        box_points[i * 3 + 1] = world_shift[1] + sx * ex[1] + sy * ey[1] + sz * ez[1];
        box_points[i * 3 + 2] = world_shift[2] + sx * ex[2] + sy * ey[2] + sz * ez[2];
    }
}

class AABB2D {
public:
    HOST_DEVICE inline
    void create_from_polyline(int num_nodes,
                              const float *nodes,
                              int begin,
                              int end);

    HOST_DEVICE inline
    void get_bounds(float bounds[4]) const;

private:
    float x_lo = 0.0f;
    float x_hi = 0.0f;
    float y_lo = 0.0f;
    float y_hi = 0.0f;

};

HOST_DEVICE inline
void AABB2D::create_from_polyline(int num_nodes,
                                  const float *nodes,
                                  int begin,
                                  int end)
{
    ASSERT(begin >= 0 && begin < num_nodes);
    ASSERT(end > begin && end <= num_nodes);

    x_lo = x_hi = nodes[(end % num_nodes) * 2 + 0];
    y_lo = y_hi = nodes[(end % num_nodes) * 2 + 1];

    for (int ip = begin; ip < end; ++ip) {
        const float *const p = nodes + ip * 2;
        x_lo = helper::min(p[0], x_lo);
        x_hi = helper::max(p[0], x_hi);
        y_lo = helper::min(p[1], y_lo);
        y_hi = helper::max(p[1], y_hi);
    }
    ASSERT(x_lo <= x_hi);
    ASSERT(y_lo <= y_hi);
}

HOST_DEVICE inline
void AABB2D::get_bounds(float bounds[4]) const
{
    bounds[0] = x_lo;
    bounds[1] = x_hi;
    bounds[2] = y_lo;
    bounds[3] = y_hi;
}

} // namespace wiregrad




