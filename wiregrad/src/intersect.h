#pragma once

#include "common.h"
#include "vector.h"


namespace wiregrad {

template<typename T> HOST_DEVICE inline
bool intersect_plane_box(const T center[3],
                         const T normal[3],
                         const T box_points[24])
{
    T q[3];
    sub3(box_points, center, q);
    const auto d = dot3(q, normal);
    for (int i = 1; i < 8; ++i) {
        sub3(box_points + i * 3, center, q);
        if (d * dot3(q, normal) <= 0.0f)
            return true;
    }
    return false;
}

template<typename T> HOST_DEVICE inline
bool intersect_plane_line(const T center[3],
                          const T normal[3],
                          const T p0[3],
                          const T p1[3])
{
    T q0[3], q1[3];
    sub3(p0, center, q0);
    sub3(p1, center, q1);
    return dot3(q0, normal) * dot3(q1, normal) <= 0.0f;
}

template<typename T> HOST_DEVICE inline
bool intersect_point_aabb2d(const T point[2],
                            const T bounds[4],
                            const T margin = static_cast<T>(0))
{
    ASSERT(margin >= 0.0f);
    const auto x_lo = bounds[0] - margin;
    const auto x_hi = bounds[1] + margin;
    const auto y_lo = bounds[2] - margin;
    const auto y_hi = bounds[3] + margin;
    return point[0] >= x_lo && point[0] <= x_hi && point[1] >= y_lo && point[1] <= y_hi;
}

template<typename T> HOST_DEVICE inline
bool intersect_point_circle(const T point[2],
                            const T center[2],
                            const T radius)
{
    const auto x = point[0] - center[0];
    const auto y = point[1] - center[1];
    return x * x + y * y <= radius * radius;
}

template<typename T> HOST_DEVICE inline
bool intersect_point_square_line(const T point[2],
                                 const T p0[2],
                                 const T p1[2],
                                 const T stroke_width,
                                 T &t)
{
    T e[2];
    sub2(p1, p0, e);

    const auto length = norm2(e);

    if (length < 1e-15f) {
        t = 0.0f;
        return false;
    }

    div2(e, length);

    T q[2];
    sub2(point, p0, q);

    const auto proj = dot2(q, e);
    t = proj / length;

    if (proj < 0.0f || proj > length)
        return false;

    const auto dist_sq = dot2(q, q) - proj * proj;
    const auto half_width = 0.5f * stroke_width;

    if (dist_sq > half_width * half_width)
        return false;

    return true;
}

template<typename T> HOST_DEVICE inline
bool intersect_point_round_line(const T point[2],
                                const T p0[2],
                                const T p1[2],
                                const T stroke_width,
                                T &t)
{
    if (intersect_point_square_line(point, p0, p1, stroke_width, t))
        return true;

    if (t < 0.5f)
        return intersect_point_circle(point, p0, /*radius=*/0.5f * stroke_width);

    return intersect_point_circle(point, p1, /*radius=*/0.5f * stroke_width);
}

} // namespace wiregrad


