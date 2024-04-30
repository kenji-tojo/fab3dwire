use rand::Rng;

fn main() {
    let fname = String::from("test4_");
    let loop_vtx2xyz = wirescaffold::read_polyloop(&(fname.clone() + ".obj"));
    let (tri2vtx, vtx2xyz) = del_msh::io_off::load_as_tri_mesh(fname.clone() + "0.off");
    let (face2idx, idx2node) = del_msh::elem2elem::face2node_of_simplex_element(3);
    let tri2tri = del_msh::elem2elem::from_uniform_mesh(
        &tri2vtx, 3, &face2idx, &idx2node,
        vtx2xyz.len() / 3);
    let (num_group, elem2group) = del_msh::elem2group::from_uniform_mesh_with_elem2elem(
        &tri2vtx, 3, &tri2tri);
    let mut out_tri2vtx = Vec::<usize>::new();
    let mut out_vtx2xyz = Vec::<f64>::new();
    let mut rng = rand::thread_rng();
    for i_group in 0..num_group {
        let ptri2vtx = del_msh::extract::from_uniform_mesh_lambda(
            &tri2vtx, 3, |i_tri| elem2group[i_tri] == i_group);
        let cumsum_area = del_msh::sampling::cumulative_area_sum(&ptri2vtx, &vtx2xyz);
        let mut pos0 = [f64::MAX; 3];
        let mut dist_max = 0_f64;
        for _itr in 0..3000 {
            let smpl = del_msh::sampling::sample_uniformly_trimesh(
                &cumsum_area, rng.gen(), rng.gen());
            let pos0_cand = del_msh::sampling::position_on_trimesh3(
                smpl.0, smpl.1, smpl.2,
                &ptri2vtx, &vtx2xyz);
            let pos1_cand = [pos0_cand[0], pos0_cand[1], -2.];
            let dist_cand = del_msh::polyloop3::distance_from_edge3(&vtx2xyz, &pos0_cand, &pos1_cand);
            if dist_cand > dist_max {
                dbg!((i_group, _itr, dist_cand, dist_max));
                pos0 = pos0_cand;
                dist_max = dist_cand;
            }
        }
        let pos1 = [pos0[0], pos0[1], -2.];
        let (pil_tri2vtx, pil_vtx2xyz) = del_msh::trimesh3_primitive::from_capsule_connecting_two_point(
            &pos0, &pos1, 0.3, 32, 8, 2);
        del_msh::trimesh3::merge(&mut out_tri2vtx, &mut out_vtx2xyz, &pil_tri2vtx, &pil_vtx2xyz);
    }
    del_msh::io_off::save_tri_mesh(
        &(fname.clone() + "2.off"),
        &out_tri2vtx, &out_vtx2xyz);
}