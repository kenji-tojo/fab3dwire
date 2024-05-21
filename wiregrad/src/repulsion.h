#pragma once

#include "common.h"
#include "vector.h"
#include "line_hierarchy3d.h"

#include <Eigen/Core>


namespace wiregrad {

struct RepulsionParameters {
public:
    float d0 = 1.0f;
    float eps = 1e-1f;
    int num_cpu_threads = 20; // ignored when using cuda
};

void repulsion(unsigned long num_nodes, const float *nodes,
               float *energy, float *d_nodes,
               bool cyclic, bool is_cuda,
               const RepulsionParameters &parameters);

void repulsion(unsigned long num_nodes, const float *nodes,
               unsigned long num_levels, const int *splits,
               const int *prims, const float *box_points,
               float *energy, float *d_nodes,
               bool cyclic, bool is_cuda,
               const RepulsionParameters &parameters);


template<typename T> HOST_DEVICE inline
auto isotropic_repulsion_kernel(const Eigen::Vector<T, 3> &p,
                                const Eigen::Vector<T, 3> &q,
                                Eigen::Vector<T, 3> &d_r,
                                const T d0,
                                const T eps) -> T
{
    using Vector3 = Eigen::Vector<T, 3>;

    auto out = static_cast<T>(0);

    const Vector3 r = q - p;
    const auto norm = r.norm() / d0;

    if (norm < 1.0f) {
        const auto w = 1.0f / (1.0f + eps * eps) / (1.0f + eps * eps);
        const auto norm_sq = norm * norm;
        const auto norm_eps_sq = norm_sq + eps * eps;

        out = 1.0f / norm_eps_sq + w * norm_sq - w * (2.0f + eps * eps);

        const auto d_norm_eps_sq = -1.0f / norm_eps_sq / norm_eps_sq;
        const auto d_norm_sq = w + d_norm_eps_sq;
        const auto d_norm = d_norm_sq * 2.0f * norm;

        d_r += d_norm * (1.0f / d0) * (q - p).normalized();
    }

    return out;
}

template<typename T> HOST_DEVICE inline
auto repulsion_kernel(const Eigen::Vector<T, 3> &p,
                      const Eigen::Vector<T, 3> &Tp,
                      const Eigen::Vector<T, 3> &q0,
                      const Eigen::Vector<T, 3> &q1,
                      T &s,
                      Eigen::Vector<T, 3> &d_r,
                      const float d0,
                      const float eps) -> T
{
    using namespace Eigen;

    const Vector<T, 3> r0 = q0 - p;
    const Vector<T, 3> r1 = q1 - p;

    auto s0 = r0.dot(Tp);
    auto s1 = r1.dot(Tp);

    if (s0 * s1 >= 0.0f)
        return static_cast<T>(0);

    s0 = std::abs(s0);
    s1 = std::abs(s1);

    s = s0 / (s0 + s1);
    const Vector<T, 3> q = (1.0f - s) * q0 + s * q1;

    return isotropic_repulsion_kernel(p, q, d_r, d0, eps);
}

struct repulsion_primitive_op {
private:
    const int num_nodes;
    const float *const nodes;
    float *const energy;
    float *const d_nodes;
    const RepulsionParameters parameters;
    const int prim_id;
    const Eigen::Vector3f p;
    const Eigen::Vector3f Tp;
    const float lp;

public:
    HOST_DEVICE
    repulsion_primitive_op(int num_nodes, const float *nodes,
                           float *energy, float *d_nodes,
                           const RepulsionParameters &parameters,
                           int prim_id, const Eigen::Vector3f &p, const Eigen::Vector3f &Tp, float lp)
            : num_nodes{ num_nodes }, nodes{ nodes }
            , energy{ energy }, d_nodes{ d_nodes }, parameters{ parameters }
            , prim_id{ prim_id }, p{ p }, Tp{ Tp }, lp{ lp } { }

    HOST_DEVICE inline
    void operator()(int iq0)
    {
        using namespace Eigen;

        const auto ip0 = prim_id;
        const auto ip1 = (ip0 + 1) % num_nodes;
        const auto iq1 = (iq0 + 1) % num_nodes;

        if (ip0 == iq0 || ip0 == iq1 || ip1 == iq0 || ip1 == iq1)
            return;

        Vector3f q0, q1;
        copy3(nodes + iq0 * 3, q0.data());
        copy3(nodes + iq1 * 3, q1.data());

        float s = -1.0f;
        Vector3f d_r = decltype(d_r)::Zero();

        const auto f = repulsion_kernel(p, Tp, q0, q1, s, d_r, parameters.d0, parameters.eps);

        if (f > 0.0f) {
            ASSERT(s >= 0.0f && s <= 1.0f);
            ASSERT(intersect_plane_line(p.data(), Tp.data(), q0.data(), q1.data()));
            energy[ip0] += lp * f;
            d_r *= lp;
            atomic_add3(d_nodes + iq0 * 3, d_r.data(), /*weight=*/1.0f - s);
            atomic_add3(d_nodes + iq1 * 3, d_r.data(), /*weight=*/       s);
            atomic_add3(d_nodes + ip0 * 3, d_r.data(), /*weight=*/-0.5f);
            atomic_add3(d_nodes + ip1 * 3, d_r.data(), /*weight=*/-0.5f);
        }
    }

};

struct repulsion_bruteforce_op {
private:
    const int num_nodes;
    const float *const nodes;
    float *const energy;
    float *const d_nodes;
    const bool cyclic;
    const RepulsionParameters parameters;

public:
    repulsion_bruteforce_op(unsigned long num_nodes, const float *nodes,
                            float *energy, float *d_nodes,
                            bool cyclic, RepulsionParameters parameters)
            : num_nodes{ static_cast<int>(num_nodes) }, nodes{ nodes }
            , energy{ energy }, d_nodes{ d_nodes }
            , cyclic{ cyclic }, parameters{ parameters } { }

    HOST_DEVICE inline
    void operator()(int num, int idx)
    {
        using namespace Eigen;

        const auto num_prims = cyclic ? num_nodes : num_nodes - 1;

        if (idx >= helper::min(num, num_prims))
            return;

        const auto ip0 = idx;
        const auto ip1 = (ip0 + 1) % num_nodes;

        Vector3f p0, p1;
        copy3(nodes + ip0 * 3, p0.data());
        copy3(nodes + ip1 * 3, p1.data());

        const Vector3f p = 0.5f * (p0 + p1);
        Vector3f Tp = p1 - p0;
        const auto lp = Tp.norm();

        if (lp < 1e-15f)
            return;

        Tp /= lp;

        repulsion_primitive_op func{
                num_nodes, nodes,
                energy, d_nodes, parameters,
                /*prim_id=*/ip0, p, Tp, lp
        };

        for (auto iq0 = 0; iq0 < num_prims; ++iq0)
            func(iq0);
    }
};

struct repulsion_accelerated_op {
private:
    const int num_nodes;
    const float *const nodes;
    const int num_levels;
    const int *const splits;
    const int *const prims;
    const float *const box_points;
    float *const energy;
    float *const d_nodes;
    const bool cyclic;
    const RepulsionParameters parameters;

public:
    repulsion_accelerated_op(unsigned long num_nodes, const float *nodes,
                             unsigned long num_levels, const int *splits, const int *prims, const float *box_points,
                             float *energy, float *d_nodes,
                             bool cyclic, RepulsionParameters parameters)
            : num_nodes{ static_cast<int>(num_nodes) }, nodes{ nodes }
            , num_levels{ static_cast<int>(num_levels) }, splits{ splits }, prims{ prims }, box_points{ box_points }
            , energy{ energy }, d_nodes{ d_nodes }
            , cyclic{ cyclic }, parameters{ parameters } { }

    HOST_DEVICE inline
    void operator()(int num, int idx)
    {
        using namespace Eigen;

        const auto num_prims = cyclic ? num_nodes : num_nodes - 1;

        if (idx >= helper::min(num, num_prims))
            return;

        const auto ip0 = idx;
        const auto ip1 = (ip0 + 1) % num_nodes;

        Vector3f p0, p1;
        copy3(nodes + ip0 * 3, p0.data());
        copy3(nodes + ip1 * 3, p1.data());

        const Vector3f p = 0.5f * (p0 + p1);
        Vector3f Tp = p1 - p0;
        const auto lp = Tp.norm();

        if (lp < 1e-15f)
            return;

        Tp /= lp;

        repulsion_primitive_op func{
                num_nodes, nodes,
                energy, d_nodes, parameters,
                /*prim_id=*/ip0, p, Tp, lp
        };

        intersect_plane_polyline(p.data(), Tp.data(),
                                 num_nodes, nodes,
                                 num_levels, splits, prims, box_points,
                                 func);
    }
};

} // namespace wiregrad
