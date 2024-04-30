fn main() {
    let fname = String::from("test4_0");
    let (tri2vtx, vtx2xyz)
        = del_msh::io_off::load_as_tri_mesh(&(fname.clone() + ".off"));
    dbg!(tri2vtx.len());
    let tripairs = {
        let bvhnodes = del_msh::bvh3_topology_topdown::from_triangle_mesh(
            &tri2vtx, &vtx2xyz);
        let mut aabbs = vec!(0f64; bvhnodes.len()/3*6);
        del_msh::bvh3::build_geometry_aabb_for_uniform_mesh(
            &mut aabbs,
            0, &bvhnodes,
            &tri2vtx, 3, &vtx2xyz, &[]);
        let mut tripairs = Vec::<del_msh::trimesh3_intersection::IntersectingPair<f64>>::new();
        del_msh::trimesh3_intersection::search_with_bvh_inside_branch(
            &mut tripairs,
            &tri2vtx, &vtx2xyz,
            0, &bvhnodes, &aabbs);
        tripairs
    };
    let mut out_tri2vtx = vec!(0usize;0);
    let mut out_vtx2xyz = vec!(0f64;0);
    for tripair in tripairs {
        let (tri2vtx0, vtx2xyz0) = del_msh::trimesh3_primitive::from_capsule_connecting_two_point(
            tripair.p0.as_slice(), tripair.p1.as_slice(),
            0.1, 4,4,4);
        del_msh::trimesh3::merge(
            &mut out_tri2vtx, &mut out_vtx2xyz,
            &tri2vtx0, &vtx2xyz0);
    }
    del_msh::io_obj::save_tri_mesh(
        fname.clone() + "intersection0.obj",
        &out_tri2vtx, &out_vtx2xyz);
}