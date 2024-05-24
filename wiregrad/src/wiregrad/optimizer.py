from typing import Union, Tuple
import numpy as np
import torch
from scipy.sparse import csr_array
from cholespy import CholeskySolverF, MatrixType


class Optimizer:
    def __init__(
        self,
        points: torch.Tensor,
        step_size = 1e-3
        ) -> None:

        if len(points.size()) != 2:
            raise ValueError('points.size() must be [>0, >0]')

        self.points = points

        self.step_size = step_size

    def zero_grad(self) -> None:
        if self.points.grad is not None:
            self.points.grad.zero_()

    def step(self, step_size: float = None) -> None:
        if self.points.grad is None:
            return

        with torch.no_grad():
            if step_size is None:
                step_size = self.step_size

            self.points -= step_size * self.points.grad

    def reset(self) -> None:
        pass


class VectorAdam(Optimizer):
    def __init__(
        self,
        points: torch.Tensor,
        step_size = 1e-3,
        beta1 = 0.9,
        beta2 = 0.99,
        eps = 1e-12
        ) -> None:

        super().__init__(points, step_size)

        if beta1 <= 0.0 or beta1 >= 1.0:
            ValueError(f'beta1 must be >0 and <1 but is {beta1}')

        if beta2 <= 0.0 or beta2 >= 1.0:
            ValueError(f'beta2 must be >0 and <1 but is {beta2}')

        if eps <= 0.0:
            ValueError(f'eps must be >0 but is {eps}')

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.beta1_acc = 1.0
        self.beta2_acc = 1.0
        self.m1: torch.Tensor = None
        self.m2: torch.Tensor = None

    def reset(self):
        self.beta1_acc = 1.0
        self.beta2_acc = 1.0
        self.m1 = None
        self.m2 = None

    def step(self, step_size: float = None) -> None:
        if self.points.grad is None:
            return

        if self.m1 is None or self.m2 is None:
            self.m1 = torch.zeros_like(self.points)
            self.m2 = torch.zeros_like(self.points)

        self.beta1_acc *= self.beta1
        self.beta2_acc *= self.beta2

        with torch.no_grad():
            self.m1 = self.beta1 * self.m1 + (1.0 - self.beta1) * self.points.grad
            self.m2 = self.beta2 * self.m2 + (1.0 - self.beta2) * torch.sum(torch.square(self.points.grad), dim=1).unsqueeze(1)

            m1_corr = self.m1 / (1.0 - self.beta1_acc)
            m2_corr = self.m2 / (1.0 - self.beta2_acc)

            grad = m1_corr / (torch.sqrt(m2_corr) + self.eps)

            if step_size is None:
                step_size = self.step_size

            self.points -= step_size * grad


class ReparamVectorAdam(Optimizer):
    def __init__(
        self,
        points: torch.Tensor,
        unique_edges: torch.Tensor,
        step_size = 1e-3,
        beta1 = 0.9,
        beta2 = 0.99,
        eps = 1e-12,
        reparam_lambda = 0.05
        ) -> None:

        super().__init__(points, step_size)

        if beta1 <= 0.0 or beta1 >= 1.0:
            ValueError(f'beta1 must be >0 and <1, but is {beta1}')

        if beta2 <= 0.0 or beta2 >= 1.0:
            ValueError(f'beta2 must be >0 and <1, but is {beta2}')

        if eps <= 0.0:
            ValueError(f'eps must be >0, but is {eps}')

        if reparam_lambda < 0.0 or reparam_lambda > 1.0:
            ValueError(f'reparam_lambda must be >=0 and <=1 but is {reparam_lambda}')

        device = self.points.device

        # assuming that unique_edges do not contain duplicated edges
        n = len(self.points)
        rows, cols, data = reparam_laplacian(n, unique_edges, reparam_lambda)

        u = csr_array((data, (rows, cols)), shape=(n, n)) @ self.points.detach().cpu().numpy()
        rows = torch.tensor(rows, device=device)
        cols = torch.tensor(cols, device=device)
        data = torch.tensor(data, device=device)

        self.u = torch.tensor(u, device=device)
        self.solver = CholeskySolverF(n, rows, cols, data, MatrixType.COO)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.beta1_acc = 1.0
        self.beta2_acc = 1.0
        self.m1: torch.Tensor = None
        self.m2: torch.Tensor = None

    def reset(self):
        self.beta1_acc = 1.0
        self.beta2_acc = 1.0
        self.m1 = None
        self.m2 = None

    def step(self, step_size: float = None) -> None:
        if self.points.grad is None:
            return

        if self.m1 is None or self.m2 is None:
            self.m1 = torch.zeros_like(self.u)
            self.m2 = torch.zeros_like(self.u)

        self.beta1_acc *= self.beta1
        self.beta2_acc *= self.beta2

        grad = torch.zeros_like(self.points.grad)
        self.solver.solve(self.points.grad, grad)

        with torch.no_grad():
            self.m1 = self.beta1 * self.m1 + (1.0 - self.beta1) * grad
            self.m2 = self.beta2 * self.m2 + (1.0 - self.beta2) * torch.sum(torch.square(grad), dim=1).unsqueeze(1)

            m1_corr = self.m1 / (1.0 - self.beta1_acc)
            m2_corr = self.m2 / (1.0 - self.beta2_acc)

            grad = m1_corr / (torch.sqrt(m2_corr) + self.eps)

            if step_size is None:
                step_size = self.step_size

            self.u -= step_size * grad
            self.solver.solve(self.u, self.points)


def unique_edges_from_trimesh(
    triangles: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:

    if isinstance(triangles, torch.Tensor):
        device = triangles.device
        triangles = triangles.detach().cpu().view(-1,3).numpy()
    else:
        device = 'cpu'
        assert isinstance(triangles, np.ndarray)

    edges = set()

    for tri in triangles:
        ip0 = tri[0].item()
        ip1 = tri[1].item()
        ip2 = tri[2].item()

        e0 = (ip0, ip1) if ip0 < ip1 else (ip1, ip0)
        e1 = (ip1, ip2) if ip1 < ip2 else (ip2, ip1)
        e2 = (ip2, ip0) if ip2 < ip0 else (ip0, ip2)

        edges.add(e0)
        edges.add(e1)
        edges.add(e2)

    edges = [ list(e) for e in sorted(edges, key=lambda x: x[0]) ]

    return torch.tensor(edges, dtype=torch.int32, device=device)


def reparam_laplacian(
    num_points: int,
    unique_edges: Union[np.ndarray, torch.Tensor],
    reparam_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:

    if isinstance(unique_edges, torch.Tensor):
        unique_edges = unique_edges.detach().cpu().numpy()

    assert isinstance(unique_edges, np.ndarray)
    assert len(unique_edges.shape) == 2 and unique_edges.shape[1] == 2

    rows = [ np.arange(num_points, dtype=np.int64) ]
    cols = [ np.arange(num_points, dtype=np.int64) ]
    data = [ reparam_lambda * np.ones(num_points, dtype=np.float32) ]

    for (ip0, ip1) in unique_edges:
        assert ip0 >= 0 and ip0 < num_points
        assert ip1 >= 0 and ip1 < num_points

        w = 1.0 - reparam_lambda
        rows.append(np.array([ ip0, ip1, ip0, ip1 ], dtype=np.int64))
        cols.append(np.array([ ip1, ip0, ip0, ip1 ], dtype=np.int64))
        data.append(np.array([  -w,  -w,   w,   w ], dtype=np.float32))

    rows = np.concatenate(rows, axis=0)
    cols = np.concatenate(cols, axis=0)
    data = np.concatenate(data, axis=0)

    return rows, cols, data



