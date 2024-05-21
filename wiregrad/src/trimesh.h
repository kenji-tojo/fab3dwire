#pragma once

#include <memory>

#include "common.h"


namespace wiregrad {

class TriMesh {
public:
    TriMesh(unsigned long num_vertices,
            const float *vertices,
            unsigned long num_triangles,
            const int *triangles);

    ~TriMesh();

    void intersect(const float mvp[16],
                   const float ndc[2],
                   int &prim_id,
                   float &t,
                   float &u,
                   float &v) const;

private:
    class Bvh_;
    std::unique_ptr<Bvh_> bvh;

};

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
                      int num_cpu_threads);

} // namespace wiregrad


