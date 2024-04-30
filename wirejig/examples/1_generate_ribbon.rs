use std::io::Write;



fn main() {
    let args: Vec<String> = std::env::args().collect();
    let proj_name = &args[1];
    let width = 0.5_f32;
    let thickness = 0.3_f32;
    let wire_rad = 0.24 * 0.5_f32;
    // let fname = String::from("bunny_wire");
    // let fname = String::from("test2_");
    // let fname = String::from("test1_");
    // let fname = String::from("test4_");
    // let fname = String::from("test5_");
    // let fname = String::from("test6_"); // iron (jan12)
    // let fname = String::from("asset/dog_cat/b");
    // let fname = String::from("asset/bunny_star/b");
    // let fname = String::from("asset/astro_einstein_v3/b");
    let fname = String::from(format!("asset/{}", proj_name) );
    // let fname = String::from("asset/shark_octopus/a");
    //let fname = String::from("asset/einstein_newton/b");
    /* ---------------- */
    let vtxl2xyz = wirejig::read_polyloop(
        &(fname.clone() + ".obj"));
    {   // output tube
        // let vtx2bin0 = del_msh::polyline::parallel_transport_polyline((&vtx2xyz).into());
        //let vtxl2framex1 = del_msh::polyline3::vtx2framex(vtxl2xyz.as_slice());
        let (trit2vtxt, vtxt2xyz)
            = del_msh::polyline3::to_trimesh3_capsule(
            &vtxl2xyz.as_slice(), 11, 4, wire_rad);
        del_msh::io_off::save_tri_mesh(
            &(fname.clone() + "_tube.off"),
            &trit2vtxt, &vtxt2xyz);
    }
    // make normal and binormal
    let (vtxl2framey, vtxl2framex) = {
        let vtx2xyz_smooth = del_msh::polyline3::smooth(vtxl2xyz.as_slice(), 0.5, 30);
        let (mut vtxl2framey, mut vtxl2framex)
            = del_msh::polyline3::normal_binormal(&vtx2xyz_smooth);
        for icol in 0..vtxl2xyz.ncols() {
            let z = del_msh::polyline3::framez(vtxl2xyz.as_slice(), icol);
            let x0 = vtxl2framex.column(icol);
            let x1 = (x0 - z.scale(x0.dot(&z))).normalize();
            let y0 = vtxl2framey.column(icol);
            let y1 = (y0 - z.scale(y0.dot(&z)) - x1.scale(y0.dot(&x1))).normalize();
            vtxl2framex.column_mut(icol).copy_from(&x1);
            vtxl2framey.column_mut(icol).copy_from(&y1);
        }
        (vtxl2framey, vtxl2framex)
    };

    // let seg2flag = wirescaffold::seg2flag(&vtx2bin);
    let seg2flag = {
        let mut seg2flag = wirejig::seg2flag_polyline(
            &vtxl2framex);
        let nseg = seg2flag.len();
        seg2flag
    };

    dbg!(&seg2flag);
    // TODO: write a function to smooth binormal based on the seg2flag

    // -----------------
    // below: generate mesh
    let (tri2vtx, vtx2xyz_goal, vtx2xyz_start, vtx2flag) = {
        let (tri2pnt, pnt2flag, pnt2xyz_start, pnt2xyz_goal)
            = wirejig::hoge6(
            true,
            &vtxl2xyz, &vtxl2framex, &vtxl2framey,
            &seg2flag,
            width, thickness, wire_rad);
        // below: mapping
        let (pnt2pnta, num_pnta) = del_msh::map_idx::from_remove_unreferenced_vertices(
            &tri2pnt, pnt2xyz_goal.len() / 3);
        let tri2pnt = del_msh::map_idx::map_elem_index(&tri2pnt, &pnt2pnta);
        let pnt2flag = del_msh::map_idx::map_vertex_attibute(
            &pnt2flag, 1, &pnt2pnta, num_pnta);
        let pnt2xyz_goal = del_msh::map_idx::map_vertex_attibute(
            &pnt2xyz_goal, 3, &pnt2pnta, num_pnta);
        let pnt2xyz_start = del_msh::map_idx::map_vertex_attibute(
            &pnt2xyz_start, 3, &pnt2pnta, num_pnta);
        (tri2pnt, pnt2xyz_goal, pnt2xyz_start, pnt2flag)
    };

    // ----------
    // below: save file
    del_msh::io_off::save_tri_mesh(
        &(fname.clone() + "_ribbon_goal.off"),
        &tri2vtx, &vtx2xyz_goal);
    del_msh::io_off::save_tri_mesh(
        &(fname.clone() + "_ribbon_start.off"),
        &tri2vtx, &vtx2xyz_start);
    {
        let file_path = fname.clone() + "_ribbon_vtx2flag.json";
        let mut file = std::fs::File::create(file_path).expect("file not found.");
        let serialized: String = serde_json::to_string(&vtx2flag).unwrap();
        let _ = file.write(serialized.as_bytes());
    }

    let vtx2xyz_start = del_msh::vtx2xyz::cast::<f64,f32>(&vtx2xyz_start);
    let vtx2xyz_goal = del_msh::vtx2xyz::cast::<f64,f32>(&vtx2xyz_goal);

    const DIST0: f64 = 0.01;
    const ALPHA: f64 = 1.0e-2;
    const K_CONTACT: f64 = 1.0e+3;
    const K_DIFF: f64 = 1.0e+2;
    let vtx2xyz = del_msh::trimesh3_move_avoid_intersection::match_vtx2xyz_while_avoid_collision(
        &tri2vtx, &vtx2xyz_start, &vtx2xyz_goal,
        K_DIFF, K_CONTACT, DIST0, ALPHA, 20);
    //

    del_msh::io_off::save_tri_mesh(&(fname + "_ribbon.off"), &tri2vtx, &vtx2xyz);
}