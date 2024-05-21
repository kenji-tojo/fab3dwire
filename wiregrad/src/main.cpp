#include <cstdio>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "wiregrad.h"


namespace nb = nanobind;
using namespace nb::literals;

template<typename T> using TensorX = nb::ndarray<nb::pytorch, T, nb::shape<nb::any>, nb::c_contig>;
template<typename T> using TensorXX = nb::ndarray<nb::pytorch, T, nb::shape<nb::any, nb::any>, nb::c_contig>;
template<typename T, long dim0> using TensorN = nb::ndarray<nb::pytorch, T, nb::shape<dim0>, nb::c_contig>;
template<typename T, long dim1> using TensorXN = nb::ndarray<nb::pytorch, T, nb::shape<nb::any, dim1>, nb::c_contig>;
template<typename T, long dim0, long dim1> using TensorNN = nb::ndarray<nb::pytorch, T, nb::shape<dim0, dim1>, nb::c_contig>;
template<typename T, long dim2> using TensorXXN = nb::ndarray<nb::pytorch, T, nb::shape<nb::any, nb::any, dim2>, nb::c_contig>;

#define CHECK_TENSOR_SIZE(tensor, dim, size) if (tensor.shape(dim) != size) throw std::runtime_error(#tensor " has an invalid tensor size")
#define CHECK_CUDA_SUPPORT(is_cuda) if (is_cuda && !wiregrad::cuda::is_available()) throw std::runtime_error("CUDA is not supported")

namespace wg = wiregrad;

namespace {

void create_line_hierarchy3d(TensorXN<float, 3> nodes,
                             TensorX<int> boxes,
                             TensorXN<int, 2> prims,
                             TensorXN<float, 24> box_points,
                             bool cyclic)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_nodes = nodes.shape(0);
    const auto num_levels = boxes.shape(0);
    const auto total_boxes = prims.shape(0);

    if (num_levels == 0)
        return;

    CHECK_TENSOR_SIZE(box_points, 0, total_boxes);

    wg::create_line_hierarchy3d(num_nodes, nodes.data(),
                              num_levels, boxes.data(),
                              total_boxes, prims.data(), box_points.data(),
                              cyclic, is_cuda);
}

void create_line_hierarchy2d(TensorXN<float, 2> nodes,
                             TensorX<int> boxes,
                             TensorXN<int, 2> prims,
                             TensorXN<float, 4> bounds,
                             bool cyclic)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_nodes = nodes.shape(0);
    const auto num_levels = boxes.shape(0);
    const auto total_boxes = prims.shape(0);

    if (num_levels == 0)
        return;

    CHECK_TENSOR_SIZE(bounds, 0, total_boxes);

    wg::create_line_hierarchy2d(num_nodes, nodes.data(),
                                 num_levels, boxes.data(),
                                 total_boxes, prims.data(), bounds.data(),
                                 cyclic, is_cuda);
}

void repulsion(TensorXN<float, 3> nodes,
               TensorX<int> splits,
               TensorXN<int, 2> prims,
               TensorXN<float, 24> box_points,
               TensorX<float> energy,
               TensorXN<float, 3> d_nodes,
               bool cyclic,
               float d0,
               float eps,
               int num_cpu_threads)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_nodes = nodes.shape(0);
    const auto num_levels = splits.shape(0);

    CHECK_TENSOR_SIZE(energy, 0, num_nodes);
    CHECK_TENSOR_SIZE(d_nodes, 0, num_nodes);

    wg::RepulsionParameters parameters;
    parameters.d0 = d0;
    parameters.eps = eps;
    parameters.num_cpu_threads = num_cpu_threads;

    if (num_levels == 0) {
        wg::repulsion(num_nodes, nodes.data(),
                      energy.data(), d_nodes.data(),
                      cyclic, is_cuda, parameters);
        return;
    }

    wg::repulsion(num_nodes, nodes.data(),
                  num_levels, splits.data(), prims.data(), box_points.data(),
                  energy.data(), d_nodes.data(),
                  cyclic, is_cuda, parameters);
}

void cubic_basis_spline(TensorXX<float> points,
                        TensorX<float> knots,
                        TensorXX<float> nodes,
                        bool cyclic)
{
    const bool is_cuda = points.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_points = points.shape(0);
    const auto dim_points = points.shape(1);
    const auto num_knots = knots.shape(0);

    CHECK_TENSOR_SIZE(nodes, 0, num_knots);
    CHECK_TENSOR_SIZE(nodes, 1, dim_points);

    wg::cubic_basis_spline(num_points, dim_points, points.data(),
                           num_knots, knots.data(), nodes.data(),
                           cyclic, is_cuda);
}

void d_cubic_basis_spline(TensorXX<float> d_nodes,
                          TensorX<float> knots,
                          TensorXX<float> d_points,
                          bool cyclic,
                          int num_cpu_threads)
{
    const bool is_cuda = d_points.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_points = d_points.shape(0);
    const auto dim_points = d_points.shape(1);
    const auto num_knots = knots.shape(0);

    CHECK_TENSOR_SIZE(d_nodes, 0, num_knots);
    CHECK_TENSOR_SIZE(d_nodes, 1, dim_points);

    wg::d_cubic_basis_spline(d_nodes.data(), num_knots, knots.data(),
                             num_points, dim_points, d_points.data(),
                             cyclic, is_cuda, num_cpu_threads);
}

int checkerboard_pattern(int num_samples, TensorXN<float, 2> samples)
{
    wg::checkerboard_pattern(num_samples, samples.data());
    return num_samples;
}

void render_filled_polygons(TensorX<int> num_nodes,
                            TensorXN<float, 2> nodes,
                            TensorX<int> num_levels,
                            TensorX<int> splits,
                            TensorX<int> total_boxes,
                            TensorXN<int, 2> prims,
                            TensorXN<float, 4> bounds,
                            TensorN<float, 3> color,
                            TensorXN<float, 2> pixel_samples,
                            TensorXX<float> weight_sum,
                            TensorXXN<float, 3> contrib_sum,
                            int num_cpu_threads,
                            TensorN<float, 3> background)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_polygons = num_nodes.shape(0);

    CHECK_TENSOR_SIZE(num_levels, 0, num_polygons);
    CHECK_TENSOR_SIZE(total_boxes, 0, num_polygons);

    wg::FilledPolygons polygons {
            num_polygons,
            num_nodes.data(), nodes.data(),
            num_levels.data(), splits.data(),
            total_boxes.data(), prims.data(), bounds.data()
    };

    wg::copy3(color.data(), polygons.color);

    const auto height = static_cast<int>(weight_sum.shape(0));
    const auto width = static_cast<int>(weight_sum.shape(1));

    CHECK_TENSOR_SIZE(contrib_sum, 0, height);
    CHECK_TENSOR_SIZE(contrib_sum, 1, width);

    wg::RenderOptions options;
    options.num_cpu_threads = num_cpu_threads;
    wg::copy3(background.data(), options.background);

    wg::render_filled_polygons(polygons, pixel_samples.shape(0), pixel_samples.data(),
                               width, height, weight_sum.data(), contrib_sum.data(),
                               options, is_cuda);
}

void d_render_filled_polygons(TensorXXN<float, 3> d_image,
                              TensorX<int> num_nodes,
                              TensorXN<float, 2> nodes,
                              TensorX<int> num_levels,
                              TensorX<int> splits,
                              TensorX<int> total_boxes,
                              TensorXN<int, 2> prims,
                              TensorXN<float, 4> bounds,
                              TensorN<float, 3> color,
                              float length_sum,
                              TensorXN<int, 2> node_id,
                              TensorX<float> lerp,
                              TensorXN<float, 2> normal,
                              TensorXN<float, 2> d_nodes,
                              int num_cpu_threads,
                              TensorN<float, 3> background)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_polygons = num_nodes.shape(0);
    const auto num_samples = node_id.shape(0);

    CHECK_TENSOR_SIZE(num_levels, 0, num_polygons);
    CHECK_TENSOR_SIZE(total_boxes, 0, num_polygons);
    CHECK_TENSOR_SIZE(lerp, 0, num_samples);
    CHECK_TENSOR_SIZE(normal, 0, num_samples);

    wg::FilledPolygons polygons {
            num_polygons,
            num_nodes.data(), nodes.data(),
            num_levels.data(), splits.data(),
            total_boxes.data(), prims.data(), bounds.data()
    };

    wg::copy3(color.data(), polygons.color);

    const auto height = static_cast<int>(d_image.shape(0));
    const auto width = static_cast<int>(d_image.shape(1));

    wg::RenderOptions options;
    options.num_cpu_threads = num_cpu_threads;
    wg::copy3(background.data(), options.background);

    wg::d_render_filled_polygons(width, height, d_image.data(), polygons,
                                 length_sum, num_samples,
                                 node_id.data(), lerp.data(), normal.data(),
                                 d_nodes.data(), options, is_cuda);
}

void sample_edges(TensorXN<float, 2> nodes,
                  TensorXN<int, 2> edges,
                  TensorX<float> length_cdf,
                  TensorX<float> samples,
                  TensorXN<int, 2> node_id,
                  TensorX<float> lerp,
                  TensorXN<float, 2> normal)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_edges = edges.shape(0);
    const auto num_samples = samples.shape(0);

    CHECK_TENSOR_SIZE(length_cdf, 0, num_edges + 1);
    CHECK_TENSOR_SIZE(node_id, 0, num_samples);
    CHECK_TENSOR_SIZE(lerp, 0, num_samples);
    CHECK_TENSOR_SIZE(normal, 0, num_samples);

    wg::sample_edges(nodes.data(), num_edges, edges.data(), length_cdf.data(),
                     num_samples, samples.data(), node_id.data(), lerp.data(), normal.data(),
                     is_cuda);
}

void sample_edges_and_corners(TensorXN<float, 2> nodes,
                              float stroke_width,
                              TensorXN<int, 2> edges,
                              TensorX<float> length_cdf,
                              TensorXN<float, 2> corner_normals,
                              TensorX<float> angles,
                              TensorX<float> samples,
                              TensorXN<int, 2> node_id,
                              TensorX<float> lerp,
                              TensorXN<float, 2> normal,
                              TensorXN<float, 2> position)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_nodes = nodes.shape(0);
    const auto num_edges = edges.shape(0);
    const auto num_samples = samples.shape(0);

    CHECK_TENSOR_SIZE(length_cdf, 0, num_edges * 2 + num_nodes + 1);
    CHECK_TENSOR_SIZE(corner_normals, 0, num_nodes);
    CHECK_TENSOR_SIZE(angles, 0, num_nodes);
    CHECK_TENSOR_SIZE(node_id, 0, num_samples);
    CHECK_TENSOR_SIZE(lerp, 0, num_samples);
    CHECK_TENSOR_SIZE(normal, 0, num_samples);
    CHECK_TENSOR_SIZE(position, 0, num_samples);

    wg::sample_edges_and_corners(num_nodes, nodes.data(), stroke_width,
                                 num_edges, edges.data(), length_cdf.data(), corner_normals.data(), angles.data(),
                                 num_samples, samples.data(), node_id.data(), lerp.data(), normal.data(), position.data(),
                                 is_cuda);
}

void render_polylines(TensorX<int> num_nodes,
                      TensorXN<float, 2> nodes,
                      TensorX<int> num_levels,
                      TensorX<int> splits,
                      TensorX<int> total_boxes,
                      TensorXN<int, 2> prims,
                      TensorXN<float, 4> bounds,
                      float stroke_width,
                      TensorN<float, 3> color,
                      bool cyclic,
                      TensorXN<float, 2> pixel_samples,
                      TensorXX<float> weight_sum,
                      TensorXXN<float, 3> contrib_sum,
                      int num_cpu_threads,
                      TensorN<float, 3> background)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_polylines = num_nodes.shape(0);

    CHECK_TENSOR_SIZE(num_levels, 0, num_polylines);
    CHECK_TENSOR_SIZE(total_boxes, 0, num_polylines);

    wg::PolyLines polylines {
            num_polylines,
            num_nodes.data(), nodes.data(),
            num_levels.data(), splits.data(),
            total_boxes.data(), prims.data(), bounds.data(),
            cyclic
    };

    polylines.stroke_width = stroke_width;
    wg::copy3(color.data(), polylines.color);

    const auto height = static_cast<int>(weight_sum.shape(0));
    const auto width = static_cast<int>(weight_sum.shape(1));

    CHECK_TENSOR_SIZE(contrib_sum, 0, height);
    CHECK_TENSOR_SIZE(contrib_sum, 1, width);

    wg::RenderOptions options;
    options.num_cpu_threads = num_cpu_threads;
    wg::copy3(background.data(), options.background);

    wg::render_polylines(polylines, pixel_samples.shape(0), pixel_samples.data(),
                         width, height, weight_sum.data(), contrib_sum.data(),
                         options, is_cuda);
}

void d_render_polylines(TensorXXN<float, 3> d_image,
                        TensorX<int> num_nodes,
                        TensorXN<float, 2> nodes,
                        TensorX<int> num_levels,
                        TensorX<int> splits,
                        TensorX<int> total_boxes,
                        TensorXN<int, 2> prims,
                        TensorXN<float, 4> bounds,
                        float stroke_width,
                        TensorN<float, 3> color,
                        bool cyclic,
                        float length_sum,
                        TensorXN<int, 2> node_id,
                        TensorX<float> lerp,
                        TensorXN<float, 2> normal,
                        TensorXN<float, 2> position,
                        TensorXN<float, 2> d_nodes,
                        TensorX<float> d_stroke_width,
                        int num_cpu_threads,
                        TensorN<float, 3> background)
{
    const bool is_cuda = nodes.device_type() == nb::device::cuda::value;
    CHECK_CUDA_SUPPORT(is_cuda);

    const auto num_polygons = num_nodes.shape(0);
    const auto num_samples = node_id.shape(0);

    CHECK_TENSOR_SIZE(num_levels, 0, num_polygons);
    CHECK_TENSOR_SIZE(total_boxes, 0, num_polygons);
    CHECK_TENSOR_SIZE(lerp, 0, num_samples);
    CHECK_TENSOR_SIZE(normal, 0, num_samples);
    CHECK_TENSOR_SIZE(position, 0, num_samples);
    CHECK_TENSOR_SIZE(d_nodes, 0, nodes.shape(0));

    if (d_stroke_width.size() > 0)
        CHECK_TENSOR_SIZE(d_stroke_width, 0, num_samples);

    wg::PolyLines polylines {
            num_polygons,
            num_nodes.data(), nodes.data(),
            num_levels.data(), splits.data(),
            total_boxes.data(), prims.data(), bounds.data(),
            cyclic
    };

    polylines.stroke_width = stroke_width;
    wg::copy3(color.data(), polylines.color);

    const auto height = static_cast<int>(d_image.shape(0));
    const auto width = static_cast<int>(d_image.shape(1));

    wg::RenderOptions options;
    options.num_cpu_threads = num_cpu_threads;
    wg::copy3(background.data(), options.background);

    wg::d_render_polylines(width, height, d_image.data(), polylines, length_sum,
                           num_samples, node_id.data(), lerp.data(), normal.data(), position.data(),
                           d_nodes.data(),
                           d_stroke_width.size() > 0 ? d_stroke_width.data() : nullptr,
                           options, is_cuda);
}

namespace debug {

auto intersect_plane_polyline(TensorN<float, 3> center,
                              TensorN<float, 3> normal,
                              TensorXN<float, 3> nodes,
                              TensorX<int> splits,
                              TensorXN<int, 2>  prims,
                              TensorXN<float, 24> box_points,
                              TensorX<int> ip_out) -> long
{
    const auto num_nodes = static_cast<int>(nodes.shape(0));
    const auto num_levels = static_cast<int>(splits.shape(0));

    CHECK_TENSOR_SIZE(ip_out, 0, num_nodes);

    std::vector<int> ip;

    if (splits.size() > 0) {
        wg::intersect_plane_polyline(center.data(), normal.data(),
                                     num_nodes, nodes.data(),
                                     num_levels, splits.data(),
                                     prims.data(), box_points.data(),
                                     [&] (int ip0) { ip.push_back(ip0); });
    }
    else {
        // brute-force
        for (int ip0 = 0; ip0 < nodes.shape(0); ++ip0) {
            const auto ip1 = ip0 < nodes.shape(0) - 1 ? ip0 + 1 : 0;
            const float *const p0 = nodes.data() + ip0 * 3;
            const float *const p1 = nodes.data() + ip1 * 3;
            if (wg::intersect_plane_line(center.data(), normal.data(), p0, p1))
                ip.push_back(ip0);
        }
    }

    std::copy(ip.cbegin(), ip.cend(), ip_out.data());

    return static_cast<long>(ip.size());
}

void isotropic_repulsion_kernel(TensorX<float> r,
                                TensorX<float> output,
                                TensorX<float> d_r,
                                const float d0,
                                const float eps)
{
    const auto num = r.shape(0);

    CHECK_TENSOR_SIZE(output, 0, num);
    CHECK_TENSOR_SIZE(d_r, 0, num);

    using namespace Eigen;
    std::vector<Vector3f> p, q, d_r_vec3;
    p.resize(num, Vector3f::Zero());
    q.resize(num, Vector3f::Zero());
    d_r_vec3.resize(num, Vector3f::Zero());

    for (int i = 0; i < num; ++i)
        q[i][0] = r(i);

    wg::cpu::parallel_for(num, [&] (int idx, int tid) {
        output(idx) = wg::isotropic_repulsion_kernel(p[idx], q[idx], d_r_vec3[idx], d0, eps);
        d_r(idx) += d_r_vec3[idx][0];
    });
}

void render_triangles(TensorNN<float, 4, 4> mvp,
                      TensorXN<float, 3> vertices,
                      TensorXN<int, 3> triangles,
                      TensorXN<float, 3> colors,
                      TensorN<float, 3> background,
                      TensorXXN<float, 3> image,
                      int num_samples,
                      int num_cpu_threads)
{
    const auto num_vertices = vertices.shape(0);
    const auto num_triangles = triangles.shape(0);
    const auto num_colors = colors.shape(0);

    if (num_colors != 1)
        CHECK_TENSOR_SIZE(colors, 0, num_vertices);

    const auto height = static_cast<int>(image.shape(0));
    const auto width = static_cast<int>(image.shape(1));

    wg::render_triangles(mvp.data(),
                         num_vertices, vertices.data(),
                         num_triangles, triangles.data(),
                         num_colors, colors.data(),
                         background.data(),
                         width, height, image.data(),
                         num_samples,
                         num_cpu_threads);
}

} // namespace debug

} // namespace


NB_MODULE(_m, m)
{
    m.def("cuda_is_available", [] { return wg::cuda::is_available(); });

    m.def("create_line_hierarchy3d", &::create_line_hierarchy3d);
    m.def("create_line_hierarchy2d", &::create_line_hierarchy2d);

    m.def("repulsion", &::repulsion);

    m.def("cubic_basis_spline", &::cubic_basis_spline);
    m.def("d_cubic_basis_spline", &::d_cubic_basis_spline);

    m.def("checkerboard_pattern", &::checkerboard_pattern);
    m.def("render_filled_polygons", &::render_filled_polygons);
    m.def("d_render_filled_polygons", &::d_render_filled_polygons);
    m.def("render_polylines", &::render_polylines);
    m.def("d_render_polylines", &::d_render_polylines);

    m.def("sample_edges", &::sample_edges);
    m.def("sample_edges_and_corners", &::sample_edges_and_corners);

    m.def("debug_intersect_plane_polyline", &::debug::intersect_plane_polyline);
    m.def("debug_isotropic_repulsion_kernel", &::debug::isotropic_repulsion_kernel);
    m.def("debug_render_triangles", &::debug::render_triangles);
}



