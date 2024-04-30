pub fn read_polyloop<P: AsRef<std::path::Path>>(fpath: P) -> nalgebra::Matrix3xX::<f32> {
    use std::fs::File;
    use std::io::{BufReader, BufRead};
    let mut vtx2xyz = Vec::<f32>::new();
    let f = File::open(fpath).expect("file not found");
    for res_line in BufReader::new(f).lines() {
        let line = res_line.unwrap();
        if line.starts_with("v ") {
            let a: Vec<&str> = line.split(' ').collect();
            let x0: f32 = a[1].parse().unwrap();
            let y0: f32 = a[2].parse().unwrap();
            let z0: f32 = a[3].parse().unwrap();
            vtx2xyz.push(x0);
            vtx2xyz.push(y0);
            vtx2xyz.push(z0);
        }
    }
    nalgebra::Matrix3xX::<f32>::from_vec(vtx2xyz)
}

pub fn seg2flag_loop(
    vtx2bin: &nalgebra::Matrix3xX::<f32>) -> Vec<bool> {
    let num_vtx = vtx2bin.ncols();
    let mut seg2flag = vec!(true; num_vtx);
    for iseg in 0..num_vtx {
        let ip0 = iseg;
        let ip1 = (ip0 + 1) % num_vtx;
        let b0 = vtx2bin.column(ip0);
        let b1 = vtx2bin.column(ip1);
        let dot = b0.dot(&b1);
        if dot < 0.9 {
            seg2flag[iseg] = false;
        }
    }
    seg2flag
}

pub fn seg2flag_polyline(
    vtx2bin: &nalgebra::Matrix3xX::<f32>) -> Vec<bool> {
    let num_vtx = vtx2bin.ncols();
    let mut seg2flag = vec!(true; num_vtx-1);
    for iseg in 0..num_vtx-1 {
        let ip0 = iseg;
        let ip1 = ip0 + 1;
        let b0 = vtx2bin.column(ip0);
        let b1 = vtx2bin.column(ip1);
        let dot = b0.dot(&b1);
        if dot < 0.8 {
            seg2flag[iseg] = false;
        }
    }
    seg2flag
}


pub fn tri2pnt_segmented_loop(
    vtx2bin: &nalgebra::Matrix3xX::<f32>,
    seg2flag: &[bool],
    m: usize) -> Vec<usize>
{
    let mut tri2pnt = Vec::<usize>::new();
    let num_vtx = vtx2bin.ncols();
    assert_eq!(seg2flag.len(), num_vtx);
    for iseg1 in 0..num_vtx {
        let ip0 = iseg1;
        let ip1 = (ip0 + 1) % num_vtx;
        let iseg0 = (iseg1 + num_vtx - 1) % num_vtx;
        let iseg2 = (iseg1 + 1) % num_vtx;
        if seg2flag[iseg1] {
            for i in 0..m {
                tri2pnt.push(ip1 * m + (0 + i)%m);
                tri2pnt.push(ip0 * m + (0 + i)%m);
                tri2pnt.push(ip0 * m + (1 + i)%m);
                //
                tri2pnt.push(ip1 * m + (0 + i)%m);
                tri2pnt.push(ip0 * m + (1 + i)%m);
                tri2pnt.push(ip1 * m + (1 + i)%m);
            }
        } else {
            if seg2flag[iseg0] {
                for i in 0..m-2 {
                    tri2pnt.push(ip0 * m);
                    tri2pnt.push(ip0 * m + (i+1)%m);
                    tri2pnt.push(ip0 * m + (i+2)%m);
                }
            }
            if seg2flag[iseg2] {
                for i in 0..m-2 {
                    tri2pnt.push(ip1 * m);
                    tri2pnt.push(ip1 * m + (i+2)%m);
                    tri2pnt.push(ip1 * m + (i+1)%m);
                }
            }

        }
    }
    tri2pnt
}


pub fn tri2pnt_segmented_polyline(
    vtx2bin: &nalgebra::Matrix3xX::<f32>,
    seg2flag: &[bool],
    m: usize) -> Vec<usize>
{
    let mut tri2pnt = Vec::<usize>::new();
    let num_vtx = vtx2bin.ncols();
    assert_eq!(seg2flag.len(), num_vtx - 1);
    {
        let ip1 = 0;
        for i in 0..m - 2 {
            tri2pnt.push(ip1 * m);
            tri2pnt.push(ip1 * m + (i + 2) % m);
            tri2pnt.push(ip1 * m + (i + 1) % m);
        }
    }
    for iseg1 in 0..num_vtx - 1 {
        let ip0 = iseg1;
        let ip1 = ip0 + 1;
        if seg2flag[iseg1] {
            for i in 0..m {
                tri2pnt.push(ip1 * m + (0 + i) % m);
                tri2pnt.push(ip0 * m + (0 + i) % m);
                tri2pnt.push(ip0 * m + (1 + i) % m);
                //
                tri2pnt.push(ip1 * m + (0 + i) % m);
                tri2pnt.push(ip0 * m + (1 + i) % m);
                tri2pnt.push(ip1 * m + (1 + i) % m);
            }
        } else {
            if seg2flag[iseg1 - 1] {
                for i in 0..m - 2 {
                    tri2pnt.push(ip0 * m);
                    tri2pnt.push(ip0 * m + (i + 1) % m);
                    tri2pnt.push(ip0 * m + (i + 2) % m);
                }
            }
            if seg2flag[iseg1 + 1] {
                for i in 0..m - 2 {
                    tri2pnt.push(ip1 * m);
                    tri2pnt.push(ip1 * m + (i + 2) % m);
                    tri2pnt.push(ip1 * m + (i + 1) % m);
                }
            }
        }
    }
    {
        let ip0 = num_vtx - 1;
        for i in 0..m - 2 {
            tri2pnt.push(ip0 * m);
            tri2pnt.push(ip0 * m + (i + 1) % m);
            tri2pnt.push(ip0 * m + (i + 2) % m);
        }
    }
    tri2pnt
}


/*
fn hoge5(
    is_polyline: bool,
    vtx2xyz: &nalgebra::Matrix3xX::<f32>,
    vtx2bin: &nalgebra::Matrix3xX::<f32>,
    vtx2nrm: &nalgebra::Matrix3xX::<f32>,
    seg2flag: &[bool],
    width: f32,
    thickness: f32,
    wire_rad: f32) -> (Vec<usize>, Vec<usize>, Vec<f32>, Vec<f32>)
{
    let tri2pnt = tri2pnt_segmented_tube(&vtx2bin, seg2flag, 5);
    let pnt2flag = {
        let mut pnt2flag = vec!(0; vtx2xyz.ncols() * 5);
        for i in 0..vtx2xyz.ncols() {
            pnt2flag[i * 5 + 0] = 1;
        }
        pnt2flag
    };
    let pnt2xyz_goal = {
        let num_vtx = vtx2xyz.ncols();
        let mut pnt2xyz = Vec::<f32>::new();
        for ipnt in 0..num_vtx {
            let p0: nalgebra::Vector3::<f32> = vtx2xyz.column(ipnt).into_owned();
            let p1 = (p0 + width * vtx2bin.column(ipnt)).into_owned();
            let p2 = (p0 - width * vtx2bin.column(ipnt)).into_owned();
            let p3 = p1 + vtx2nrm.column(ipnt).scale(thickness);
            let p4 = p2 + vtx2nrm.column(ipnt).scale(thickness);
            p0.iter().for_each(|&v| pnt2xyz.push(v));
            p1.iter().for_each(|&v| pnt2xyz.push(v));
            p3.iter().for_each(|&v| pnt2xyz.push(v));
            p4.iter().for_each(|&v| pnt2xyz.push(v));
            p2.iter().for_each(|&v| pnt2xyz.push(v));
        }
        pnt2xyz
    };
    let pnt2xyz_start = {
        let num_vtx = vtx2xyz.ncols();
        let mut pnt2xyz = Vec::<f32>::new();
        for ipnt in 0..num_vtx {
            let p0: nalgebra::Vector3::<f32> = vtx2xyz.column(ipnt).into_owned();
            let p1 = (p0 + wire_rad * vtx2bin.column(ipnt)).into_owned();
            let p2 = (p0 - wire_rad * vtx2bin.column(ipnt)).into_owned();
            let p3 = p1 + vtx2nrm.column(ipnt).scale(wire_rad);
            let p4 = p2 + vtx2nrm.column(ipnt).scale(wire_rad);
            p0.iter().for_each(|&v| pnt2xyz.push(v));
            p1.iter().for_each(|&v| pnt2xyz.push(v));
            p3.iter().for_each(|&v| pnt2xyz.push(v));
            p4.iter().for_each(|&v| pnt2xyz.push(v));
            p2.iter().for_each(|&v| pnt2xyz.push(v));
        }
        pnt2xyz
    };
    (tri2pnt, pnt2flag, pnt2xyz_start, pnt2xyz_goal)
}
 */

pub fn hoge6(
    is_polyline: bool,
    vtx2xyz: &nalgebra::Matrix3xX::<f32>,
    vtx2bin: &nalgebra::Matrix3xX::<f32>,
    vtx2nrm: &nalgebra::Matrix3xX::<f32>,
    seg2flag: &[bool],
    width: f32,
    thickness: f32,
    wire_rad: f32) -> (Vec<usize>, Vec<usize>, Vec<f32>, Vec<f32>)
{
    let tri2pnt =
    if is_polyline {
        tri2pnt_segmented_polyline(&vtx2bin, seg2flag, 6)
    } else {
        tri2pnt_segmented_loop(&vtx2bin, seg2flag, 6)
    };
    let pnt2flag = {
        let mut pnt2flag = vec!(0; vtx2xyz.ncols() * 6);
        for i in 0..vtx2xyz.ncols() {
            pnt2flag[i * 6 + 0] = 1;
            pnt2flag[i * 6 + 3] = 2;
        }
        pnt2flag
    };
    let pnt2xyz_goal = {
        let num_vtx = vtx2xyz.ncols();
        let mut pnt2xyz = Vec::<f32>::new();
        for ipnt in 0..num_vtx {
            let p0: nalgebra::Vector3::<f32> = vtx2xyz.column(ipnt).into_owned();
            let p1 = (p0 + width * vtx2bin.column(ipnt)).into_owned();
            let p2 = (p0 - width * vtx2bin.column(ipnt)).into_owned();
            let p3 = p1 + vtx2nrm.column(ipnt).scale(thickness);
            let p4 = p2 + vtx2nrm.column(ipnt).scale(thickness);
            let p5 = p0 + vtx2nrm.column(ipnt).scale(thickness);
            p0.iter().for_each(|&v| pnt2xyz.push(v));
            p1.iter().for_each(|&v| pnt2xyz.push(v));
            p3.iter().for_each(|&v| pnt2xyz.push(v));
            p5.iter().for_each(|&v| pnt2xyz.push(v));
            p4.iter().for_each(|&v| pnt2xyz.push(v));
            p2.iter().for_each(|&v| pnt2xyz.push(v));
        }
        pnt2xyz
    };
    let pnt2xyz_start = {
        let num_vtx = vtx2xyz.ncols();
        let mut pnt2xyz = Vec::<f32>::new();
        for ipnt in 0..num_vtx {
            let p0: nalgebra::Vector3::<f32> = vtx2xyz.column(ipnt).into_owned();
            let p1 = (p0 + wire_rad * vtx2bin.column(ipnt)).into_owned();
            let p2 = (p0 - wire_rad * vtx2bin.column(ipnt)).into_owned();
            let p3 = p1 + vtx2nrm.column(ipnt).scale(wire_rad);
            let p4 = p2 + vtx2nrm.column(ipnt).scale(wire_rad);
            let p5 = p0 + vtx2nrm.column(ipnt).scale(wire_rad);
            p0.iter().for_each(|&v| pnt2xyz.push(v));
            p1.iter().for_each(|&v| pnt2xyz.push(v));
            p3.iter().for_each(|&v| pnt2xyz.push(v));
            p5.iter().for_each(|&v| pnt2xyz.push(v));
            p4.iter().for_each(|&v| pnt2xyz.push(v));
            p2.iter().for_each(|&v| pnt2xyz.push(v));
        }
        pnt2xyz
    };
    (tri2pnt, pnt2flag, pnt2xyz_start, pnt2xyz_goal)
}