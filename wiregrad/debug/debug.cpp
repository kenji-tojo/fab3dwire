#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "wiregrad.h"


int main()
{
    using namespace Eigen;
    namespace wg = wiregrad;

    Matrix<float, Dynamic, 3, RowMajor> nodes;
    nodes.resize(30000, 3);

    for (int i = 0; i < nodes.rows(); ++i) {
        auto phi = 2.0f * wg::Pi<float> * float(i) / float(nodes.rows());
        nodes(i,0) = std::cos(phi) + 2.0f * std::cos(2.0f * phi);
        nodes(i,1) = std::sin(phi) - 2.0f * std::sin(2.0f * phi);
        nodes(i,1) = 2.0f * std::sin(3.0f * phi);
    }

    VectorXi splits;
    splits.setZero(1 + wg::helper::max<int>(0, int(std::log2(nodes.rows() / 16)) - 1));
    splits[0] = 16;
    std::fill(splits.begin() + 1, splits.end(), 2);
    assert(splits.prod() > 0);
    assert(splits.prod() < nodes.rows());

    VectorXi boxes = splits;
    for (int i = 1; i < splits.size(); ++i)
        boxes[i] *= boxes[i - 1];

    std::cout << "tree splits = " << splits.transpose() << std::endl;
    std::cout << "# of boxes  = " << boxes.transpose() << std::endl;
    std::cout << "# of prims  = " << nodes.rows() << std::endl;

    const auto total_boxes = boxes.sum();
    std::vector<int> prims(total_boxes * 2);
    std::vector<float> box_points(total_boxes * 24);

    wg::Timer timer;

    timer.reset();
    wg::create_line_hierarchy3d(nodes.rows(), nodes.data(), boxes.size(), boxes.data(),
                                total_boxes, prims.data(), box_points.data(), /*cyclic=*/true);

    auto time_construction = timer.elapsed_microseconds();
    printf("hierarchy construction took %.3f ms\n", time_construction / 1000.0);

    {
        std::vector<int> ip_brute_force;
        std::vector<int> ip_accelerated;

        double time_brute_force = 0.0f;
        double time_accelerated = 0.0f;

        Vector3f center, normal;
        center.setZero();

        auto n = 50;
        auto m = 50;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                auto u = (float(i) + 0.5f) / float(n);
                auto v = (float(j) + 0.5f) / float(m);
                auto z = -1.0f + 2.0f * u;
                auto phi = 2.0f * wg::Pi<float> * v;
                auto tmp = std::sqrt(wg::helper::relu(1.0f - z * z));
                auto x = tmp * std::cos(phi);
                auto y = tmp * std::sin(phi);
                normal << x, y, z;
                ASSERT(std::abs(normal.norm() - 1.0f) < 1e-5f);

                timer.reset();
                for (int ip0 = 0; ip0 < nodes.rows(); ++ip0) {
                    auto ip1 = ip0 < nodes.rows() - 1 ? ip0 + 1 : 0;
                    const float *p0 = nodes.data() + ip0 * 3;
                    const float *p1 = nodes.data() + ip1 * 3;
                    if (wg::intersect_plane_line(center.data(), normal.data(), p0, p1))
                        ip_brute_force.push_back(ip0);
                }
                time_brute_force += timer.elapsed_microseconds();

                timer.reset();
                wg::intersect_plane_polyline(center.data(), normal.data(),
                                              static_cast<int>(nodes.rows()), nodes.data(),
                                              static_cast<int>(splits.size()), splits.data(),
                                              prims.data(), box_points.data(),
                                              [&] (int ip0) { ip_accelerated.push_back(ip0); });
                time_accelerated += timer.elapsed_microseconds();
            }
        }

        printf("query (brute-force) took %.3f ms\n", time_brute_force / 1000.0);
        printf("query (accelerated) took %.3f ms\n", time_accelerated / 1000.0);
        printf("total (accelerated) took %.3f ms\n", (time_construction + time_accelerated) / 1000.0);
        printf("compute time was reduced to %.2f%%\n", 100.0 * (time_construction + time_accelerated) / time_brute_force);

        auto num_isects = (int) ip_brute_force.size();
        auto num_mismatch = 0;

        assert(num_isects > 0);
        assert(ip_brute_force.size() == ip_accelerated.size());

        for (int i = 0; i < ip_brute_force.size(); ++i)
            num_mismatch += int(ip_brute_force[i] != ip_accelerated[i]);

        if (num_mismatch > 0)
            fprintf(stderr, "brute_force vs. accelerated mismatch count = %d/%d\n", num_mismatch, num_isects);
        else
            printf("brute_force vs. accelerated mismatch count = %d/%d\n", num_mismatch, num_isects);
    }
}

