#include "trimesh.h"

#include <limits>
#include <vector>
#include <mutex>

#include "bvh/v2/bvh.h"
#include "bvh/v2/default_builder.h"
#include "bvh/v2/stack.h"
#include "bvh/v2/tri.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include "vector.h"
#include "antialias.h"
#include "image_tile.h"
#include "thread.h"


namespace {
struct Hit {
    int prim_id = -1;
    float t = std::numeric_limits<float>::max();
    float u = 0.0f;
    float v = 0.0f;
};
} // namespace

namespace wiregrad {

class TriMesh::Bvh_: public bvh::v2::Bvh<bvh::v2::Node<float, 3>> {
private:
    using Scalar  = float;
    using Vec3    = bvh::v2::Vec<Scalar, 3>;
    using BBox    = bvh::v2::BBox<Scalar, 3>;
    using Tri     = bvh::v2::Tri<Scalar, 3>;
    using Node    = bvh::v2::Node<Scalar, 3>;
    using Bvh     = bvh::v2::Bvh<Node>;
    using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;
    const bool should_permute;

public:
    Bvh_([[maybe_unused]] unsigned long num_vertices,
         const float *vertices,
         unsigned long num_triangles,
         const int *triangles,
         bool should_permute = true)
            : should_permute(should_permute)
    {
        std::vector<Tri> tris(num_triangles);
        for (int i = 0; i < num_triangles; ++i) {
            const auto ip0 = triangles[i * 3 + 0];
            const auto ip1 = triangles[i * 3 + 1];
            const auto ip2 = triangles[i * 3 + 2];
            ASSERT(ip0 >= 0 && ip0 < num_vertices);
            ASSERT(ip1 >= 0 && ip1 < num_vertices);
            ASSERT(ip2 >= 0 && ip2 < num_vertices);
            const float *const v = vertices;
            const auto p0 = Vec3{ v[ip0 * 3 + 0], v[ip0 * 3 + 1], v[ip0 * 3 + 2] };
            const auto p1 = Vec3{ v[ip1 * 3 + 0], v[ip1 * 3 + 1], v[ip1 * 3 + 2] };
            const auto p2 = Vec3{ v[ip2 * 3 + 0], v[ip2 * 3 + 1], v[ip2 * 3 + 2] };
            tris[i] = Tri{ p0, p1, p2 };
        }

        bvh::v2::ThreadPool thread_pool;
        bvh::v2::ParallelExecutor executor(thread_pool);

        // Get triangle centers and bounding boxes (required for BVH builder)
        std::vector<BBox> bboxes(tris.size());
        std::vector<Vec3> centers(tris.size());
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                bboxes[i]  = tris[i].get_bbox();
                centers[i] = tris[i].get_center();
            }
        });

        typename bvh::v2::DefaultBuilder<Node>::Config config;
        config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
        bvh_.emplace(bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config));
        const auto &bvh = bvh_.value();

        // This precomputes some data to speed up traversal further.
        precomputed_tris.resize(tris.size());
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto j = should_permute ? bvh.prim_ids[i] : i;
                precomputed_tris[i] = tris[j];
            }
        });
    }

    bool intersect(bvh::v2::Ray<Scalar, 3> ray, ::Hit &output) const
    {
        const auto &bvh = bvh_.value();

        static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
        static constexpr size_t stack_size = 64;
        static constexpr bool use_robust_traversal = false;

        auto prim_id = invalid_id;
        Scalar u, v; // barycentric coords

        // Traverse the BVH and get the u, v coordinates of the closest intersection.
        bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
        bvh.template intersect<false, use_robust_traversal>(ray, bvh.get_root().index, stack,
                                                            [&] (size_t begin, size_t end) {
                                                                for (size_t i = begin; i < end; ++i) {
                                                                    size_t j = should_permute ? i : bvh.prim_ids[i];
                                                                    if (auto hit = precomputed_tris[j].intersect(ray)) {
                                                                        prim_id = i;
                                                                        std::tie(u, v) = *hit;
                                                                    }
                                                                }
                                                                return prim_id != invalid_id;
                                                            });

        if (prim_id != invalid_id && ray.tmax < output.t) {
            output.prim_id = static_cast<int>(bvh.prim_ids[prim_id]);
            output.t = ray.tmax;
            output.u = u;
            output.v = v;
            return true;
        }

        return false;
    }

private:
    std::optional<Bvh> bvh_;
    std::vector<PrecomputedTri> precomputed_tris;

};

TriMesh::TriMesh(unsigned long num_vertices,
                 const float *vertices,
                 unsigned long num_triangles,
                 const int *triangles)
        : bvh{ std::make_unique<Bvh_>(num_vertices, vertices, num_triangles, triangles) } { }

TriMesh::~TriMesh() = default;

void TriMesh::intersect(const float mvp[16],
                        const float ndc[2],
                        int &prim_id,
                        float &t,
                        float &u,
                        float &v) const
{
    using namespace Eigen;
    using Mat4 = Matrix<float, 4, 4, RowMajor>;

    Vector4f org4{ ndc[0], ndc[1], -1.0f, 1.0f };
    Vector4f end4{ ndc[0], ndc[1], 1.0f, 1.0f };

    const Mat4 inv = Map<const Mat4>{ mvp }.inverse();
    org4 = inv * org4;
    end4 = inv * end4;

    const Vector3f org = decltype(org){ org4.x(), org4.y(), org4.z() } / org4.w();
    const Vector3f end = decltype(end){ end4.x(), end4.y(), end4.z() } / end4.w();
    const Vector3f dir = (end - org).normalized();
    constexpr auto tmin = 0.0f;
    const auto tmax = (end - org).norm();

    bvh::v2::Ray<float, 3> ray{
            bvh::v2::Vec<float, 3>{ org.x(), org.y(), org.z() },
            bvh::v2::Vec<float, 3>{ dir.x(), dir.y(), dir.z() },
            tmin, tmax
    };

    ::Hit hit;

    bvh->intersect(ray, hit);

    prim_id = hit.prim_id;
    t = hit.t;
    u = hit.u;
    v = hit.v;
}

void render_triangles(const float mvp[16],
                      unsigned long num_vertices,
                      const float *vertices,
                      unsigned long num_triangles,
                      const int *triangles,
                      unsigned long num_colors,
                      const float *colors,
                      const float background[3],
                      int width,
                      int height,
                      float *image,
                      int num_samples,
                      int num_cpu_threads)
{
    using namespace Eigen;

    TriMesh mesh{ num_vertices, vertices, num_triangles, triangles };

    num_samples = helper::max(1, num_samples);
    std::vector<float> samples(num_samples * 2);

    checkerboard_pattern(num_samples, samples.data());
    samples.resize(num_samples * 2);

    VectorXf weight_sum;
    Matrix<float, Dynamic, 3, RowMajor> contrib_sum;

    const auto num_pixels = height * width;
    weight_sum.setZero(num_pixels);
    contrib_sum.setZero(num_pixels, 3);

    const auto num_threads = helper::max(1, num_cpu_threads);
    const auto num_tiles = num_threads;
    const auto tile_height = height / num_tiles + 1;
    constexpr auto tile_margin = 2;
    const auto tile_vertical_size = tile_height + 2 * tile_margin;

    Matrix<float, Dynamic, Dynamic, RowMajor> weight_sum_buffer, contrib_sum_buffer;
    weight_sum_buffer.setZero(num_tiles, tile_vertical_size * width);
    contrib_sum_buffer.setZero(num_tiles, tile_vertical_size * width * 3);

    std::vector<ImageTile> image_tiles;
    image_tiles.reserve(num_tiles);

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        image_tiles.emplace_back(width, height,
                                 tile_id, tile_height, tile_margin,
                                 weight_sum_buffer.row(tile_id).data(),
                                 contrib_sum_buffer.row(tile_id).data());
        ASSERT(image_tiles.back().tile_vertical_size() == tile_vertical_size);
    }

    std::mutex image_mutex;

    cpu::parallel_for(num_tiles, [&] (int idx, int tid) {
        const auto &tile = image_tiles[idx];
        const auto ih_bgn = helper::min<int>(tile_height * (idx + 0), height);
        const auto ih_end = helper::min<int>(tile_height * (idx + 1), height);

        float rgb[3], ndc[2];

        for (int ih = ih_bgn; ih < ih_end; ++ih)
            for (int iw = 0; iw < width; ++iw)
                for (int ip = 0; ip < num_samples; ++ip) {
                    gaussian_filter_weight_op filter_weight_func;
                    auto &p = filter_weight_func.p;
                    p[0] = static_cast<float>(iw) + samples[ip * 2 + 0];
                    p[1] = static_cast<float>(ih) + samples[ip * 2 + 1];

                    ndc[0] = 2.0f * (-0.5f + p[0] / static_cast<float>(width));
                    ndc[1] = 2.0f * (-0.5f + p[1] / static_cast<float>(height));
                    ndc[1] *= -1.0f; // flip vertically

                    int prim_id = -1;
                    float t, u, v;
                    mesh.intersect(mvp, ndc, prim_id, t, u, v);

                    wiregrad::copy3(background, rgb);

                    if (prim_id >= 0) {
                        if (num_colors == 1) {
                            wiregrad::copy3(colors, rgb);
                        }
                        else if (num_colors == num_vertices) {
                            const auto ip0 = triangles[prim_id * 3 + 0];
                            const auto ip1 = triangles[prim_id * 3 + 1];
                            const auto ip2 = triangles[prim_id * 3 + 2];
                            const Map<const Vector3f> c0{ colors + ip0 * 3 };
                            const Map<const Vector3f> c1{ colors + ip1 * 3 };
                            const Map<const Vector3f> c2{ colors + ip2 * 3 };
                            const Vector3f c = (1.0f - u - v) * c0 + u * c1 + v * c2;
                            wiregrad::copy3(c.data(), rgb);
                        }
                    }

                    tile.accumulate(iw, ih, rgb, filter_weight_func);
                }

        std::lock_guard guard(image_mutex);
        tile.merge(weight_sum.data(), contrib_sum.data());
    }, num_threads);

    Map<Matrix<float, Dynamic, 3, RowMajor>> im{ image, num_pixels, 3 };
    im.col(0) += (contrib_sum.col(0).array() / weight_sum.array()).matrix();
    im.col(1) += (contrib_sum.col(1).array() / weight_sum.array()).matrix();
    im.col(2) += (contrib_sum.col(2).array() / weight_sum.array()).matrix();
}

} // namespace wiregrad


