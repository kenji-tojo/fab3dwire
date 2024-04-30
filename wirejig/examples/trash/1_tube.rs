
fn main() {
    let fname = String::from("test3_");
    let rad = 0.5_f64;
    //
    let vtx2xyz = wirescaffold::read_polyloop(fname.clone() + ".obj");
    let vtx2bin0 = del_msh::polyline::parallel_transport_polyline(vtx2xyz.as_slice());
    let vtx2bin1 = del_msh::polyloop3::smooth_frame(vtx2xyz.as_slice());
    let (tri2pnt, pnt2xyz) = del_msh::polyloop3::tube_mesh(
        &vtx2xyz, &vtx2bin1, rad as f32);
    del_msh::io_obj::save_tri_mesh(
        fname.clone() +"tube.obj",
        &tri2pnt, &pnt2xyz);
    let vtx2xyz_f64 = vtx2xyz.cast::<f64>();
    let vtx2bin1_f64 = vtx2bin1.cast::<f64>();
    let (tri2pnt, pnt2xyz) = del_msh::polyloop3::tube_mesh_avoid_intersection(
        vtx2xyz_f64.as_slice(), &vtx2bin1_f64,rad / 10 as f64, 10);
    del_msh::io_obj::save_tri_mesh(
        fname.clone() + "tube1.obj",
        &tri2pnt, &pnt2xyz);
}