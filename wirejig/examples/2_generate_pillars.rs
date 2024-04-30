use std::env;
use std::io::Read;
use num_traits::AsPrimitive;

type Vec3f = nalgebra::Vector3::<f32>;

fn edge_coordinate<T>(hair2vtx: &[usize], ih: usize, rh: T) -> (usize, T)
where T: AsPrimitive<usize> + nalgebra::RealField,
    usize: AsPrimitive<T>
{
    let num_vtx = hair2vtx[ih+1]-hair2vtx[ih];
    if rh >= <usize as AsPrimitive<T>>::as_(num_vtx-1) {
        return (hair2vtx[ih+1]-2,T::one());
    }
    let k0: usize =  hair2vtx[ih] + rh.as_();
    let rk = rh - rh.floor();
    (k0,rk)
}

#[allow(clippy::identity_op)]
fn wdw_proximity<T>(
    eng: &mut T,
    grad_eng: &mut [T],
    prox_idx: &[usize],
    prox_prm: &[T],
    hair2vtx: &[usize],
    vtx2xyz: &[T],
    dist0: T,
    stiff: T)
    where T: nalgebra::RealField + Copy + AsPrimitive<usize>,
        usize: AsPrimitive<T>,
        f64: AsPrimitive<T>
{
    let barrier = |x: T| -stiff * (dist0 - x) * (dist0 - x) * (x / dist0).ln();
    let diff_barrier = |x: T| -stiff * (x - dist0) * (x - dist0) / x - stiff * 2f64.as_() * (x - dist0) * (x / dist0).ln();
    for i_prox in 0..prox_idx.len() / 2 {
        let (j0,rj) =  edge_coordinate(hair2vtx, prox_idx[i_prox * 2 + 0], prox_prm[i_prox * 2 + 0]);
        let (k0, rk) = edge_coordinate(hair2vtx, prox_idx[i_prox * 2 + 1], prox_prm[i_prox * 2 + 1]);
        assert!(j0<hair2vtx[prox_idx[i_prox * 2 + 0]+1]-1, "{} {}", prox_prm[i_prox * 2 + 0], hair2vtx[prox_idx[i_prox * 2 + 0]+1]-hair2vtx[prox_idx[i_prox * 2 + 0]]);
        let pj0 = del_geo::vec3::to_na(vtx2xyz, j0);
        let pj1 = del_geo::vec3::to_na(vtx2xyz, j0+1);
        let pk0 = del_geo::vec3::to_na(vtx2xyz, k0);
        let pk1 = del_geo::vec3::to_na(vtx2xyz, k0+1);
        let pj = pj0 + (pj1-pj0)*rj;
        let pk = pk0 + (pk1-pk0)*rk;
        let dist1 = (pj-pk).norm();
        assert!(dist1<=dist0);
        *eng += barrier(dist1);
        let deng1 = diff_barrier(dist1);
        let unorm = (pj-pk).normalize();
        for i in 0..3 {
            grad_eng[(j0+0) * 3 + i] += unorm[i] * deng1 * (T::one()-rj);
            grad_eng[(j0+1) * 3 + i] += unorm[i] * deng1 * rj;
            grad_eng[(k0+0) * 3 + i] -= unorm[i] * deng1 * (T::one()-rk);
            grad_eng[(k0+1) * 3 + i] -= unorm[i] * deng1 * rk;
        }
    }
}

fn hoge<T>(
    eng: &mut T,
    grad_eng: &mut [T],
    pil2vtxh: &[usize],
    vtxh2xyz: &[T],
    vtxl2xyz: &[T],
    dist0: T,
    stiff: T)
where T: Copy + nalgebra::RealField + num_traits::AsPrimitive<usize>,
      f64: AsPrimitive<T>,
    usize: AsPrimitive<T>
{
    let barrier = |x: T| -stiff * (dist0 - x) * (dist0 - x) * (x / dist0).ln();
    let diff_barrier = |x: T| -stiff * (x - dist0) * (x - dist0) / x - stiff * 2f64.as_() * (x - dist0) * (x / dist0).ln();
    for i_pil in 0..pil2vtxh.len() - 1 {
        for i0_vtx in pil2vtxh[i_pil]..pil2vtxh[i_pil+1]-1 {
            let i1_vtx = i0_vtx+1;
            let p0 = del_geo::vec3::to_na(vtxh2xyz, i0_vtx);
            let p1 = del_geo::vec3::to_na(vtxh2xyz, i1_vtx);
            let res = del_msh::polyloop3::nearest_to_edge3(vtxl2xyz, &p0, &p1);
            if res.0 > dist0 { continue; }
            let qc = del_msh::polyloop3::position_from_barycentric_coordinate(vtxl2xyz, res.1);
            let pc = p0 + (p1-p0)*res.2;
            let dist1 = (pc-qc).norm();
            //dbg!(dist1,dist0);
            //assert!(dist1<=dist0);
            let unorm = pc-qc;
            *eng += barrier(dist1);
            let deng1 = diff_barrier(dist1);
            for i in 0..3 {
                grad_eng[(i0_vtx) * 3 + i] += unorm[i] * deng1 * (T::one()-res.2);
                grad_eng[(i1_vtx) * 3 + i] += unorm[i] * deng1 * res.2;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let proj_name = &args[1];
    let fname = format!("asset/{}", proj_name);
    let vtxl2xyz = wirejig::read_polyloop(fname.clone() + ".obj");
    let (tris2vtxp, vtxs2xyz) = del_msh::io_off::load_as_tri_mesh::<String, f32>(
        fname.clone() + "_ribbon.off");
    let vtxs2flag = {
        let file_path = fname.clone() + "_ribbon_vtx2flag.json";
        let mut file = std::fs::File::open(file_path).expect("file not found.");
        let mut serialized = String::new();
        let _ = file.read_to_string(&mut serialized);
        let deserialized: Vec<i32> = serde_json::from_str(&serialized).unwrap();
        deserialized
    };
    assert_eq!(vtxs2xyz.len() / 3, vtxs2flag.len());
    //
    let (num_group, tris2group) = del_msh::elem2group::from_triangle_mesh(
        &tris2vtxp, vtxs2xyz.len() / 3);
    assert_eq!(tris2group.len(), tris2vtxp.len() / 3);
    let group2vtxs_socket = {
        let mut group2vtx = vec!(std::collections::BTreeSet::<usize>::new(); num_group);
        for i_tri in 0..tris2group.len() {
            for i_node in 0..3 {
                let i_vtx = tris2vtxp[i_tri * 3 + i_node];
                if vtxs2flag[i_vtx] != 2 { continue; }
                let i_group = tris2group[i_tri];
                group2vtx[i_group].insert(i_vtx);
            }
        }
        let mut group2vtx_socket = vec!(Vec::<usize>::new(); num_group);
        for i_group in 0..num_group {
            group2vtx_socket[i_group].extend(group2vtx[i_group].iter());
        }
        group2vtx_socket
    };
    println!("initial polyline pillar generation");
    const RAD_AVOID: f32 = 0.35;
    const RAD_PILLAR: f32 = 0.30;
    const PILLAR_INTERVAL: usize = 8;
    let (pil2vtxh, vtxh2xyz_ini) = {
        let mut vtxh2xyz = Vec::<f32>::new();
        let mut pil2vtxh = Vec::<usize>::new();
        pil2vtxh.push(0);
        for i_group in 0..num_group {
            for i_ivtx_socket in 0..(group2vtxs_socket[i_group].len()-1) / PILLAR_INTERVAL +1 {
                let i_vtx = group2vtxs_socket[i_group][i_ivtx_socket * PILLAR_INTERVAL];
                let polyline = {
                    let pos0 = del_geo::vec3::to_na(&vtxs2xyz, i_vtx);
                    let pos1 = nalgebra::Vector3::<f32>::new(pos0[0], pos0[1], 0.);
                    del_msh::polyline::resample_preserve_corner(&vec!(pos0, pos1), RAD_AVOID * 2.)
                };
                for &p in polyline.iter() { vtxh2xyz.extend(&p); }
                pil2vtxh.push(pil2vtxh[pil2vtxh.len() - 1] + polyline.len());
            }
        }
        (pil2vtxh, vtxh2xyz)
    };
    let dof2mask = {
        let mut dof2mask = vec!(1f32; vtxh2xyz_ini.len());
        for i_pil in 0..pil2vtxh.len() - 1 {
            let iv0 = pil2vtxh[i_pil];
            dof2mask[iv0*3+0] = 0f32;
            dof2mask[iv0*3+1] = 0f32;
            dof2mask[iv0*3+2] = 0f32;
            let ivn = pil2vtxh[i_pil+1]-1;
            dof2mask[ivn*3+2] = 0f32;
        }
        dof2mask
    };
    let mut vtxh2xyz = vtxh2xyz_ini.clone();
    for itr in 0..1000{
        let dist0 = RAD_PILLAR * 2.;
        let alpha = 0.00005;
        let (prox_idx, prox_prm)
            = del_msh::polyline3::contacting_pair(&pil2vtxh, &vtxh2xyz, dist0);
        assert_eq!(prox_idx.len() / 2, prox_prm.len() / 2);
        let mut eng = 0f32;
        let mut grad_eng = vec!(0.;vtxh2xyz.len());
        const K_CONTACT: f32 = 1.0e+3;
        const K_DIFF: f32 = 1.0e+2;
        wdw_proximity(
            &mut eng, &mut grad_eng,
            &prox_idx, &prox_prm,
            &pil2vtxh, &vtxh2xyz, dist0, K_CONTACT);
        for i in 0..vtxh2xyz.len() {
            let d = vtxh2xyz[i] - vtxh2xyz_ini[i];
            eng += 0.5 * d * d * K_DIFF;
            grad_eng[i] += d * K_DIFF;
        }
        hoge(
            &mut eng, &mut grad_eng,
            &pil2vtxh, &vtxh2xyz,
            vtxl2xyz.as_slice(),
            dist0,
            K_CONTACT);
        let step: Vec<_> = {
            let mut step = vec!(0f32;vtxh2xyz.len());
            for i in 0..vtxh2xyz.len() {
                step[i] = -grad_eng[i]*alpha*dof2mask[i];
            }
            step
        };
        let vtx2xyz_dist: Vec<f32> = vtxh2xyz.iter().zip(step.iter())
            .map(|(v, r)| v + r).collect();
        vtxh2xyz = vtx2xyz_dist;
        dbg!(eng);
    }
    println!("start generating start & goal");
    let (trip2vtxp, vtxp2xyz_start, vtxp2xyz_goal) = {
        let mut tri2vtx_s = Vec::<usize>::new();
        let mut vtx2xyz_s = Vec::<f32>::new();
        for i_poly in 0..pil2vtxh.len() - 1 { // start
            let polyline = &vtxh2xyz[pil2vtxh[i_poly] * 3..pil2vtxh[i_poly + 1] * 3];
            let (pil_tri2vtx, pil_vtx2xyz) = del_msh::polyline3::to_trimesh3_capsule(
                &polyline, 8, 2, RAD_PILLAR * 0.1);
            del_msh::trimesh3::merge(&mut tri2vtx_s, &mut vtx2xyz_s, &pil_tri2vtx, &pil_vtx2xyz);
        }
        let mut tri2vtx_e = Vec::<usize>::new();
        let mut vtx2xyz_e = Vec::<f32>::new();
        for i_poly in 0..pil2vtxh.len() - 1 { // start
            let polyline = &vtxh2xyz[pil2vtxh[i_poly] * 3..pil2vtxh[i_poly + 1] * 3];
            let (pil_tri2vtx, pil_vtx2xyz) = del_msh::polyline3::to_trimesh3_capsule(
                &polyline, 8, 2, RAD_PILLAR);
            del_msh::trimesh3::merge(&mut tri2vtx_e, &mut vtx2xyz_e, &pil_tri2vtx, &pil_vtx2xyz);
        }
        let vtx2xyz_s: Vec<f64> = vtx2xyz_s.iter().map(|&v| v as f64).collect();
        let vtx2xyz_e: Vec<f64> = vtx2xyz_e.iter().map(|&v| v as f64).collect();
        (tri2vtx_s, vtx2xyz_s, vtx2xyz_e)
    };
    del_msh::io_off::save_tri_mesh(
        fname.clone() + "_pillars_start.off",
        &trip2vtxp, &vtxp2xyz_goal);
    del_msh::io_off::save_tri_mesh(
        fname.clone() + "_pillars_goal.off",
        &trip2vtxp, &vtxp2xyz_goal);
    println!("inflating pillar mesh");

    const DIST0: f64 = 0.01;
    const ALPHA: f64 = 1.0e-2;
    const K_CONTACT: f64 = 1.0e+3;
    const K_DIFF: f64 = 1.0e+2;
    let vtxp2xyz = del_msh::trimesh3_move_avoid_intersection::match_vtx2xyz_while_avoid_collision(
        &trip2vtxp, &vtxp2xyz_start, &vtxp2xyz_goal,
        K_DIFF, K_CONTACT, DIST0, ALPHA, 20);
    del_msh::io_off::save_tri_mesh(
        fname.clone() + "_pillars.off",
        &trip2vtxp, &vtxp2xyz);
}