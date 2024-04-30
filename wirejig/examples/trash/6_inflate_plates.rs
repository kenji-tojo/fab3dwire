use std::io::Read;

// use num_traits::AsPrimitive;

/*
fn hoge<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    bvhnodes: &[usize],
    aabbs: &mut [T]) -> Vec<del_msh::trimesh3_intersection::IntersectingPair<T>>
where T: num_traits::Float + nalgebra::RealField
{
    del_msh::bvh3::build_geometry_aabb_for_uniform_mesh(
        aabbs,
        0, &bvhnodes,
        &tri2vtx, 3, &vtx2xyz, &[]);
    let mut tripairs = Vec::<del_msh::trimesh3_intersection::IntersectingPair<T>>::new();
    /*
    del_msh::trimesh3_intersection::search_with_bvh_inside_branch(
        &mut tripairs,
        &tri2vtx, &vtx2xyz,
        0, &bvhnodes, &aabbs);
     */
    let mut tripairs = Vec::<del_msh::trimesh3_intersection::IntersectingPair<T>>::new();
    del_msh::trimesh3_intersection::search_brute_force(
        &mut tripairs,
        &tri2vtx, &vtx2xyz);
    tripairs
}
 */


fn main() {
    //test();
    // let fname = String::from("test4_");
    //let fname = String::from("test5_");
    // let fname = String::from("test6_"); // iron
    let fname = String::from("test7_"); // einstein
    let (tri2vtx, vtx2xyz_start) = del_msh::io_off::load_as_tri_mesh(
        "target/".to_owned() + &fname.clone() + "start.off");
    let (_, vtx2xyz_goal) = del_msh::io_off::load_as_tri_mesh::<_, f64>(
        "target/".to_owned() + &fname.clone() + "goal.off");

    let vtx2flag = {
        let file_path = "target/".to_owned() + &fname.clone() + "vtx2flag.json";
        let mut file = std::fs::File::open(file_path).expect("file not found.");
        let mut serialized = String::new();
        let _ = file.read_to_string(&mut serialized);
        let deserialized: Vec<i32> = serde_json::from_str(&serialized).unwrap();
        deserialized
    };
    assert_eq!(vtx2xyz_start.len() / 3, vtx2flag.len());

    /*
    let bvhnodes = del_msh::bvh3_topology_topdown::from_triangle_mesh(
        &tri2vtx, &vtx2xyz_goal);
    let mut aabbs = vec!(0f64; bvhnodes.len()/3*6);
    let tripairs = hoge(&tri2vtx, &vtx2xyz_goal, &bvhnodes, &mut aabbs);
    println!("goal intersection {:}",tripairs.len());

    dbg!(time0);
    let mut vtx2xyz: Vec<_> = vtx2xyz_start.iter().zip(vtx2xyz_goal.iter())
        .map(|(&v0,&v1)| v0+(v1-v0)*time0*0.95).collect();
    //let tripairs = hoge(&tri2vtx, &vtx2xyz, &bvhnodes, &mut aabbs);
    //dbg!(tripairs.len());
    del_msh::io_off::save_tri_mesh("target/".to_owned() + &fname+"ini.off", &tri2vtx, &vtx2xyz);
     */
    const DIST0: f64 = 0.01;
    const ALPHA: f64 = 1.0e-2;
    const K_CONTACT: f64 = 1.0e+3;
    const K_DIFF: f64 = 1.0e+2;
    let vtx2xyz = del_msh::trimesh3_move_avoid_intersection::match_vtx2xyz_while_avoid_collision(
        &tri2vtx, &vtx2xyz_start, &vtx2xyz_goal,
        K_DIFF, K_CONTACT, DIST0, ALPHA, 10);
    //

    del_msh::io_off::save_tri_mesh("target/".to_owned() + &fname + "ini.off", &tri2vtx, &vtx2xyz);
}