fn split_loop_into_two_polylines(
    vtx2xyz: &[f32],
    i0_vtx: usize,
    i1_vtx: usize) -> (Vec<f32>, Vec<f32>)
{
    assert_ne!(i0_vtx, i1_vtx);
    let num_vtx = vtx2xyz.len() / 3;
    assert!(i0_vtx < num_vtx);
    assert!(i1_vtx < num_vtx);
    if i0_vtx < i1_vtx {
        let vtx2xyz_01 = Vec::<f32>::from(&vtx2xyz[i0_vtx * 3..i1_vtx * 3 + 3]);
        let mut vtx2xyz_10 = Vec::<f32>::from(&vtx2xyz[i1_vtx * 3..num_vtx * 3]);
        vtx2xyz_10.extend_from_slice(&vtx2xyz[0..i0_vtx * 3 + 3]);
        return (vtx2xyz_01, vtx2xyz_10);
    } else {
        let vtx2xyz_10 = Vec::<f32>::from(&vtx2xyz[i1_vtx * 3..i0_vtx * 3 + 3]);
        let mut vtx2xyz_01 = Vec::<f32>::from(&vtx2xyz[i0_vtx * 3..num_vtx * 3]);
        vtx2xyz_01.extend_from_slice(&vtx2xyz[0..i1_vtx * 3 + 3]);
        return (vtx2xyz_01, vtx2xyz_10);
    }
}

fn hoge(
    vtx2xyz: &[f32],
    scale: f32,
    path_out: &str)
{

    let mut vtx2xyz = {
        let vtx2xyz = nalgebra::Matrix3xX::<f32>::from_column_slice(vtx2xyz);
        let cov = del_msh::polyline::cov::<f32, 3>(vtx2xyz.as_slice());
        let eig = nalgebra::linalg::SymmetricEigen::new(cov);
        let (_evals, evecs) = {
            let (evals, mut evecs) = del_geo::mat3::sort_eigen(&eig.eigenvalues, &eig.eigenvectors, false);
            if evecs.determinant() < 0.0 {
                let c0 = evecs.column(0).into_owned().scale(-1f32);
                evecs.column_mut(0).copy_from(&c0);
            }
            (evals, evecs)
        };
        dbg!(evecs.determinant());
        evecs.transpose() * vtx2xyz
    };
    {   // center
        let aabb = del_geo::aabb3::from_vtx2xyz(vtx2xyz.as_slice(), 0.);
        let cnt = nalgebra::Vector3::<f32>::from_row_slice(&del_geo::aabb3::center(&aabb));
        vtx2xyz.column_iter_mut().for_each(|mut v| v -= cnt);
        vtx2xyz *= scale;
    }
    {   // cgz should be low
        let cg = del_msh::polyline3::cg(vtx2xyz.as_slice());
        if cg.z > 0. {
            let rot = nalgebra::Matrix3::<f32>::new(
                1.0, 0.0, 0.0,
                0.0, -1.0, 0.0,
                    0.0, 0.0, -1.0);
            vtx2xyz = rot * vtx2xyz;
        }
    }
    {   // center
        let aabb = del_geo::aabb3::from_vtx2xyz(vtx2xyz.as_slice(), 0.);
        let cnt = nalgebra::Vector3::<f32>::new(0., 0., aabb[2]-0.3);
        vtx2xyz.column_iter_mut().for_each(|mut v| v -= cnt);
    }
    del_msh::polyloop3::write_wavefrontobj(std::path::Path::new(path_out), &vtx2xyz);
}

fn load(
    path_in: &str,
    path_out_a: &str,
    path_out_b: &str,
    path_out_a_assembly: &str,
    path_out_b_assembly: &str,
    scale: f32)
{
    let vtx2xyz = wirejig::read_polyloop(path_in);
    let vtx2xyz = del_msh::polyloop::resample::<f32, 3>(vtx2xyz.as_slice(), 300);
    {
        let aabb = del_geo::aabb3::from_vtx2xyz(vtx2xyz.as_slice(), 0.);
        let max_edge_size = del_geo::aabb3::max_edge_size(&aabb);
        dbg!(max_edge_size);
    }
    let num_vtx = vtx2xyz.len() / 3;
    let (vtx2xyz_01, vtx2xyz_10) = {
        let (mut variance_min, mut i0_vtx_min, mut i1_vtx_min) = (f32::MAX, 0, 0);
        for i0_vtx in 0..num_vtx {
            for i1_vtx in i0_vtx + 1..num_vtx {
                let (vtx2xyz_01, vtx2xyz_10)
                    = split_loop_into_two_polylines(&vtx2xyz, i0_vtx, i1_vtx);
                // dbg!(vtx2xyz_01.len()/3, vtx2xyz_10.len()/3);
                let cov_01 = del_msh::polyline::cov::<f32, 3>(vtx2xyz_01.as_slice());
                let cov_10 = del_msh::polyline::cov::<f32, 3>(vtx2xyz_10.as_slice());
                let eig_01 = nalgebra::linalg::SymmetricEigen::new(cov_01);
                let eig_10 = nalgebra::linalg::SymmetricEigen::new(cov_10);
                let (evals_01, _evecs_01)
                    = del_geo::mat3::sort_eigen(&eig_01.eigenvalues, &eig_01.eigenvectors, false);
                let (evals_10, _evecs_10)
                    = del_geo::mat3::sort_eigen(&eig_10.eigenvalues, &eig_10.eigenvectors, false);
                // dbg!(&evals_01, &evals_10);
                let variance = evals_01[2] + evals_10[2];
                if variance < variance_min {
                    variance_min = variance;
                    i0_vtx_min = i0_vtx;
                    i1_vtx_min = i1_vtx;
                    // dbg!(evals_01, evals_10);
                    // dbg!(variance_min, i0_vtx_min, i1_vtx_min);
                }
            }
        }
        dbg!(i0_vtx_min, i1_vtx_min);
        split_loop_into_two_polylines(&vtx2xyz, i0_vtx_min, i1_vtx_min)
    };
    { // output assembly
        let (tri2vtxt01, vtxt2xyz01)
            = del_msh::polyline3::to_trimesh3_capsule(&vtx2xyz_01, 8, 2, 0.03);
        let _ = del_msh::io_obj::save_tri_mesh_( path_out_a_assembly, &tri2vtxt01, &vtxt2xyz01, 3);
        let (tri2vtxt10, vtxt2xyz10)
            = del_msh::polyline3::to_trimesh3_capsule(&vtx2xyz_10, 8, 2, 0.03);
        let _ = del_msh::io_obj::save_tri_mesh_( path_out_b_assembly, &tri2vtxt10, &vtxt2xyz10, 3);
    }
    hoge(&vtx2xyz_01, scale, path_out_a);
    hoge(&vtx2xyz_10, scale, path_out_b);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let proj_name = &args[1];
    println!("project name: {}", proj_name);
    let (path_dir, scale) = ("asset/dog_cat/", 5.);
    load(&(path_dir.to_owned() + "input.obj"),
         &(path_dir.to_owned()+"a.obj"),
         &(path_dir.to_owned() + "b.obj"),
         &(path_dir.to_owned()+"a_assembly.obj"),
         &(path_dir.to_owned()+"b_assembly.obj"),
         scale);
}