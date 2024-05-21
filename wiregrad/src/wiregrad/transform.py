from typing import List, Tuple, Union
import math
import torch


def _angle_axis_rotation(
    theta: torch.Tensor,
    axis: int = 0
    ) -> torch.Tensor:

    theta = theta.view(1).squeeze(0)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    m = torch.zeros(3, 3, dtype=torch.float32)

    if axis == 0:
        m[0,0] = 1.0
        m[1,1], m[1,2] = cos_theta, -sin_theta
        m[2,1], m[2,2] = sin_theta,  cos_theta
    elif axis == 1:
        m[1,1] = 1.0
        m[0,0], m[0,2] =  cos_theta, sin_theta
        m[2,0], m[2,2] = -sin_theta, cos_theta
    elif axis == 2:
        m[2,2] = 1.0
        m[0,0], m[0,1] = cos_theta, -sin_theta
        m[1,0], m[1,1] = sin_theta,  cos_theta
    else:
        raise ValueError('axis must be >=0 and <3')

    return m


def rotation(
    angle: torch.Tensor
    ) -> torch.Tensor:

    angle = (math.pi / 180.0) * angle.view(3)

    rot_x = _angle_axis_rotation(angle[0], axis=0)
    rot_y = _angle_axis_rotation(angle[1], axis=1)
    rot_z = _angle_axis_rotation(angle[2], axis=2)

    return torch.matmul(rot_z, torch.matmul(rot_y, rot_x))


def _maybe_float_to_tensor(x: Union[float, torch.Tensor]) -> torch.Tensor:
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    return x.view(1).squeeze(0)


def perspective(
    fovy: Union[float, torch.Tensor],
    aspect: Union[float, torch.Tensor],
    near: Union[float, torch.Tensor],
    far: Union[float, torch.Tensor]
    ) -> torch.Tensor:

    fovy = _maybe_float_to_tensor(fovy)
    aspect = _maybe_float_to_tensor(aspect)
    near = _maybe_float_to_tensor(near)
    far = _maybe_float_to_tensor(far)

    tan_theta = torch.tan(0.5 * math.pi * fovy / 180.0)
    right = aspect * near * tan_theta
    top = near * tan_theta

    m = torch.zeros(4, 4, dtype=torch.float32)
    m[0,0] = near / right
    m[1,1] = near / top
    m[2,2] = -(far + near) / (far - near)
    m[2,3] = -2.0 * far * near / (far - near)
    m[3,2] = -1.0

    return m


def look_at(
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor
    ) -> torch.Tensor:

    eye = eye.view(3)
    center = center.view(3)
    up = up.view(3)

    def _normalized(v: torch.Tensor) -> torch.Tensor:
        return v / torch.linalg.vector_norm(v)

    # right-handed coordinates
    ez = _normalized(eye - center)
    ex = _normalized(torch.linalg.cross(up, ez))
    ey = _normalized(torch.linalg.cross(ez, ex))

    m = torch.zeros(4, 4, dtype=torch.float32)
    m[0,0], m[0,1], m[0,2], m[0,3] = ex[0], ex[1], ex[2], -eye.dot(ex)
    m[1,0], m[1,1], m[1,2], m[1,3] = ey[0], ey[1], ey[2], -eye.dot(ey)
    m[2,0], m[2,1], m[2,2], m[2,3] = ez[0], ez[1], ez[2], -eye.dot(ez)
    m[3,3] = 1.0

    return m


class Camera:
    def __init__(
        self,
        fovy: Union[float, torch.Tensor] = 40.0,
        aspect: Union[float, torch.Tensor] = 1.0,
        near: Union[float, torch.Tensor] = 1e-3,
        far: Union[float, torch.Tensor] = 100.0
        ) -> None:

        self.fovy = fovy
        self.aspect = aspect
        self.near = near
        self.far = far

    def rotate(
        self,
        eye: torch.Tensor,
        angle: torch.Tensor
        ) -> torch.Tensor:

        eye = eye.view(3).cpu()
        angle = angle.view(3).cpu()

        matrix = rotation(angle)
        up = torch.matmul(matrix, torch.tensor([0, 1, 0], dtype=torch.float32))
        to = torch.matmul(matrix, torch.tensor([0, 0, -1], dtype=torch.float32))

        return self.look_at(eye, eye + to, up)

    def look_at(
        self,
        eye: torch.Tensor,
        center: torch.Tensor,
        up = torch.tensor([0, 1, 0], dtype=torch.float32),
        ) -> torch.Tensor:

        eye = eye.view(3).cpu()
        center = center.view(3).cpu()
        up = up.view(3).cpu()
        up = up / torch.linalg.vector_norm(up)

        proj = torch.zeros(4, 4)
        view = torch.zeros(4, 4)

        proj = perspective(self.fovy, self.aspect, self.near, self.far)
        view = look_at(eye, center, up)

        return torch.matmul(proj, view)


def orthographic(
    left: Union[float, torch.Tensor],
    right: Union[float, torch.Tensor],
    bottom: Union[float, torch.Tensor],
    top: Union[float, torch.Tensor],
    near: Union[float, torch.Tensor],
    far: Union[float, torch.Tensor]
    ) -> torch.Tensor:

    left = _maybe_float_to_tensor(left)
    right = _maybe_float_to_tensor(right)
    bottom = _maybe_float_to_tensor(bottom)
    top = _maybe_float_to_tensor(top)
    near = _maybe_float_to_tensor(near)
    far = _maybe_float_to_tensor(far)

    m = torch.zeros(4, 4, dtype=torch.float32)

    m[0,0] =  2.0 / (right - left)
    m[1,1] =  2.0 / (top - bottom)
    m[2,2] = -2.0 / (far - near)

    m[0,3] = -(right + left) / (right - left)
    m[1,3] = -(top + bottom) / (top - bottom)
    m[2,3] = -(far + near)   / (far - near)

    m[3,3] = 1.0

    return m


def homography(
    matrix: torch.Tensor,
    points: torch.Tensor
    ) -> torch.Tensor:

    matrix = matrix.view(4, 4).to(points.device)
    points = points.view(-1, points.size()[-1])

    if points.size()[-1] == 3:
        w = torch.ones(len(points), 1, dtype=torch.float32, device=points.device)
        xyzw = torch.cat((points, w), dim=1)
    elif points.size()[-1] == 4:
        xyzw = points
    else:
        raise ValueError('points.size() must be [>0, 3 or 4]')

    return torch.matmul(xyzw, matrix.t())


def viewport(
    resolution: Union[Tuple[int, ...], List[int]],
    ndcw: torch.Tensor
    ) -> torch.Tensor:

    width, height = resolution
    ndcw = ndcw.view(-1, 4)

    ndc = ndcw[:,:3] / ndcw[:,3].unsqueeze(1)

    out = torch.zeros_like(ndc[:,:2])
    out[:,0] = 0.5 * (ndc[:,0] + 1.0) * width
    out[:,1] = 0.5 * (ndc[:,1] + 1.0) * height

    return out




