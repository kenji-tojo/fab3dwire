#pragma once

#include "common.h"


namespace wiregrad {

template<typename T> HOST_DEVICE inline
auto gaussian_filter_weight(const T p[2], int iw, int ih) -> T
{
    constexpr auto sigma = 0.5f;
    constexpr auto radius = 4.0f * sigma;
    constexpr auto c = 1.0f / (sigma * sigma * TwoPi<T>);
    constexpr auto alpha = 0.5f / sigma / sigma;
    const auto x = static_cast<T>(iw) + 0.5f - p[0];
    const auto y = static_cast<T>(ih) + 0.5f - p[1];
    const auto r_sq = x * x + y * y;
    return r_sq < radius * radius ? c * std::exp(-alpha * r_sq) : 0.0f;
}

struct gaussian_filter_weight_op {
public:
    static constexpr int filter_radius = 2;
    float p[2]{};
    HOST_DEVICE inline float operator()(int iw, int ih) const { return gaussian_filter_weight(p, iw, ih); }
};

template<typename T>  inline
void checkerboard_pattern(int &num, T *xy /* xy[num * 2] */)
{
    const auto maximum = num;
    ASSERT(maximum > 0);

    auto resolution = static_cast<int>(std::floor(std::sqrt(maximum * 2)));
    resolution = 2 * (resolution / 2);

    if (resolution < 2) {
        xy[0] = xy[1] = 0.5f;
        num = 1;
        return;
    }

    num = 0;

    for (int iy = 0; iy < resolution; ++iy) {
        const auto y = (static_cast<T>(iy) + 0.5f) / static_cast<T>(resolution);
        for (int ix = 0; ix < resolution / 2; ++ix) {
            const auto x = (static_cast<T>(ix * 2 + iy % 2) + 0.5f) / static_cast<T>(resolution);
            ASSERT(num < maximum);
            xy[num * 2 + 0] = x;
            xy[num * 2 + 1] = y;
            num += 1;
        }
    }
    ASSERT(num <= maximum);
}

} // namespace wiregrad


