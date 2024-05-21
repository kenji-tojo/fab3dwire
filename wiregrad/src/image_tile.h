#pragma once

#include "common.h"


namespace wiregrad {

class ImageTile {
public:
    const int width;
    const int height;
    const int tile_id;
    const int tile_height;
    const int tile_margin;
    const int tile_offset;
    float *const tile_weight_sum;
    float *const tile_contrib_sum;

    HOST_DEVICE [[nodiscard]] inline
    int tile_vertical_size() const { return tile_height + 2 * tile_margin; }

    ImageTile(int width, int height,
              int tile_id, int tile_height, int tile_margin,
              float *tile_weight_sum,
              float *tile_contrib_sum)
            : width{ width }, height{ height }
            , tile_id{ tile_id }, tile_height{ tile_height }, tile_margin{ tile_margin }
            , tile_offset{ tile_height * tile_id - tile_margin }
            , tile_weight_sum{ tile_weight_sum }, tile_contrib_sum{ tile_contrib_sum } { }

    template<typename Func> HOST_DEVICE inline
    void accumulate(int iw, int ih,
                    const float contrib[3],
                    const Func &filter_weight_func) const
    {
        const auto filter_radius = Func::filter_radius;
        const auto h_lo = helper::max(ih - filter_radius, 0);
        const auto h_hi = helper::min(ih + filter_radius, height - 1);
        const auto w_lo = helper::max(iw - filter_radius, 0);
        const auto w_hi = helper::min(iw + filter_radius, width - 1);

        for (int jh = h_lo; jh <= h_hi; ++jh)
            for (int jw = w_lo; jw <= w_hi; ++jw) {
                const auto weight = filter_weight_func(jw, jh);

                if (weight > 0.0f) {
                    ASSERT(jw >= 0 && jw < width);
                    ASSERT(jh >= 0 && jh < height);
                    ASSERT(jh >= tile_offset && jh < tile_offset + tile_vertical_size());

                    const auto pixel_id = width * (jh - tile_offset) + jw;
                    ATOMIC_ADD(tile_weight_sum, pixel_id, weight);
                    ATOMIC_ADD(tile_contrib_sum, pixel_id * 3 + 0, weight * contrib[0]);
                    ATOMIC_ADD(tile_contrib_sum, pixel_id * 3 + 1, weight * contrib[1]);
                    ATOMIC_ADD(tile_contrib_sum, pixel_id * 3 + 2, weight * contrib[2]);
                }
            }
    }

    HOST_DEVICE
    void merge(float *weight_sum,
               float *contrib_sum) const
    {
        const auto sz = tile_vertical_size();

        for (int tile_ih = 0; tile_ih < sz; ++tile_ih) {
            const auto ih = tile_ih + tile_offset;

            if (ih < 0 || ih >= height)
                continue;

            for (int iw = 0; iw < width; ++iw) {
                weight_sum[width * ih + iw] += tile_weight_sum[width * tile_ih + iw];
                contrib_sum[(width * ih + iw) * 3 + 0] += tile_contrib_sum[(width * tile_ih + iw) * 3 + 0];
                contrib_sum[(width * ih + iw) * 3 + 1] += tile_contrib_sum[(width * tile_ih + iw) * 3 + 1];
                contrib_sum[(width * ih + iw) * 3 + 2] += tile_contrib_sum[(width * tile_ih + iw) * 3 + 2];
            }
        }
    }

};

} // namespace wiregrad


