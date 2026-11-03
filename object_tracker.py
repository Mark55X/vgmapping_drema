"""
Rigid object tracking by robust point-to-plane Gauss-Newton against a fixed surface model of each object.

Each object has a model (points + outward normals, optionally colours, in the world frame of the initial scan) and a
grid around it that stores, per cell, the nearest model point, so a correspondence is a lookup instead of a search.
Every frame the object's observed points are averaged per voxel and aligned to the model starting from the previous
pose; a motion prior keeps the directions the observation does not constrain (the yaw of a cylinder, the slide of a flat
face seen from above) where they were. An optional photometric term compares the model colours with the camera images
once the geometric alignment has narrowed down. All objects are solved together, one 6x6 system each per iteration.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree


def estimate_normals(points: np.ndarray, view_points: np.ndarray, k: int = 12) -> np.ndarray:
    """Normals by local PCA, each oriented towards the camera centre (view_points) that observed the point."""
    _, nb = cKDTree(points).query(points, k=min(k, len(points)), workers=-1)
    Q = points[nb] - points[nb].mean(1, keepdims=True)
    _, V = np.linalg.eigh(np.einsum('nki,nkj->nij', Q, Q))
    n = V[:, :, 0]
    return n * np.sign(np.einsum('ij,ij->i', n, view_points - points))[:, None]


def voxel_average(points: np.ndarray, voxel: float, extra: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """One point per occupied voxel, the mean of the points in it (and of extra, row-aligned with points)."""
    _, inv = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)[:, None]
    mean = np.zeros((len(cnt), 3))
    np.add.at(mean, inv, points)
    if extra is None:
        return mean / cnt, None
    ex = np.zeros((len(cnt), extra.shape[1]))
    np.add.at(ex, inv, extra)
    return mean / cnt, ex / cnt


def fuse_scan_and_mesh(scan_points: np.ndarray, scan_normals: np.ndarray, mesh_points: np.ndarray,
                       mesh_normals: np.ndarray, keep: float = 0.003, fill: float = 0.004,
                       scan_colors: Optional[np.ndarray] = None,
                       mesh_colors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Scan points lying on the mesh (within keep: drops mask/depth mismatches) plus mesh samples where the scan has no
    point within fill (surfaces the scan never saw): the scan's fidelity where it looked, the mesh's completeness.
    Colours are carried along when both sets have them.
    """
    colored = scan_colors is not None and mesh_colors is not None
    if len(scan_points) == 0:
        return mesh_points, mesh_normals, mesh_colors
    on_mesh = cKDTree(mesh_points).query(scan_points, workers=-1)[0] < keep
    if not np.any(on_mesh):
        return mesh_points, mesh_normals, mesh_colors
    add = cKDTree(scan_points[on_mesh]).query(mesh_points, workers=-1)[0] > fill
    points = np.vstack([scan_points[on_mesh], mesh_points[add]])
    normals = np.vstack([scan_normals[on_mesh], mesh_normals[add]])
    colors = np.vstack([scan_colors[on_mesh], mesh_colors[add]]) if colored else None
    return points, normals, colors


def _expso3(om: torch.Tensor) -> torch.Tensor:
    th = om.norm(dim=1, keepdim=True).clamp(min=1e-12)
    k = om / th
    Kx = torch.zeros(len(om), 3, 3, device=om.device, dtype=om.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    s, c = torch.sin(th)[:, :, None], torch.cos(th)[:, :, None]
    return torch.eye(3, device=om.device, dtype=om.dtype) + s * Kx + (1 - c) * (Kx @ Kx)


def _rotvec(R: torch.Tensor) -> torch.Tensor:
    th = torch.arccos(((R.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1))
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    f = torch.where(th < 1e-6, torch.full_like(th, 0.5), th / (2 * torch.sin(th).clamp(min=1e-12)))
    return v * f[:, None]


def _bilinear(img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    H, W = img.shape
    u0 = u.floor().long().clamp(0, W - 2)
    v0 = v.floor().long().clamp(0, H - 2)
    du, dv = (u - u0).clamp(0, 1), (v - v0).clamp(0, 1)
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def _huber_weight(a: torch.Tensor, k: float) -> torch.Tensor:
    return torch.where(a <= k, torch.ones_like(a), k / a.clamp(min=1e-12))


class GaussNewtonObjectTracker:
    """
    Batched robust point-to-plane Gauss-Newton tracker for rigid objects with fixed surface models.

    iterations: Gauss-Newton iterations per frame (stops earlier once every object has converged)
    observation_voxel: observed points are averaged per voxel of this size, so the result does not depend on how many
        pixels the object covers and is the same for the same view
    max_distance / initial_max_distance: correspondence gate, shrinking from the initial value by 0.6 per iteration
    huber: robust kernel threshold on the point-to-plane residual
    motion_prior: penalise the motion from the previous pose with the sigmas below, in units of measurement_sigma
    motion_retry: an object whose centre moved more than this in its last update is also solved from the
        constant-velocity prediction and keeps the better fit (0 disables); static objects never use the
        prediction, which would amplify their noise
    photometric: add, once the gate has reached max_distance, the intensity difference between each coloured model
        point and the image pixel it projects to, for the model points the camera sees on the object (mask) and not
        occluded (depth within photometric_depth_tolerance); weighed against the geometric term by
        photometric_weight * (measurement_sigma / photometric_sigma)^2, at most photometric_max_points per object
    """

    def __init__(self, device: str = "cpu", iterations: int = 8, observation_voxel: float = 0.004,
                 max_distance: float = 0.01, initial_max_distance: float = 0.05, huber: float = 0.003,
                 motion_prior: bool = True, measurement_sigma: float = 0.003, motion_sigma_rot_deg: float = 5.0,
                 motion_sigma_trans: float = 0.02, motion_retry: float = 0.003, grid_cell: float = 0.003,
                 grid_margin: float = 0.06, photometric: bool = False, photometric_weight: float = 1.0,
                 photometric_sigma: float = 0.1, photometric_huber: float = 0.1,
                 photometric_depth_tolerance: float = 0.01, photometric_max_points: int = 2000):
        self.device = device
        self.iterations = iterations
        self.observation_voxel = observation_voxel
        self.max_distance = max_distance
        self.initial_max_distance = initial_max_distance
        self.huber = huber
        self.motion_retry = motion_retry
        self.grid_cell = grid_cell
        self.grid_margin = grid_margin
        self.photometric = photometric
        self.photometric_scale = photometric_weight * (measurement_sigma / photometric_sigma) ** 2
        self.photometric_huber = photometric_huber
        self.photometric_depth_tolerance = photometric_depth_tolerance
        self.photometric_max_points = photometric_max_points
        lam = np.r_[[1.0 / np.radians(motion_sigma_rot_deg) ** 2] * 3, [1.0 / motion_sigma_trans ** 2] * 3] * \
            measurement_sigma ** 2 if motion_prior else np.zeros(6)
        self.lam = torch.tensor(lam, dtype=torch.float32, device=device)
        self.slots: Dict[int, int] = {}
        self.last_motion: Dict[int, torch.Tensor] = {}

    def set_models(self, models: Dict[int, Tuple[np.ndarray, ...]]) -> None:
        """
        models: object id -> (points (N,3), outward normals (N,3)[, colours (N,3) in [0, 1] or None]) in the world
        frame of the scan.
        """
        dt = torch.float32
        self.slots = {oid: k for k, oid in enumerate(models)}
        self.last_motion = {}
        pts, nrm, gray, cells, lo, shape, offsets, ranges = [], [], [], [], [], [], [], []
        m_off, g_off = 0, 0
        for oid, model in models.items():
            p = np.asarray(model[0], dtype=np.float64)
            colors = model[2] if len(model) > 2 else None
            g_lo = p.min(0) - self.grid_margin
            g_shape = np.ceil((p.max(0) + self.grid_margin - g_lo) / self.grid_cell).astype(np.int64) + 1
            centres = g_lo + (np.stack(np.meshgrid(*[np.arange(s) for s in g_shape], indexing='ij'), -1)
                              .reshape(-1, 3) + 0.5) * self.grid_cell
            _, idx = cKDTree(p).query(centres, workers=-1)
            pts.append(p)
            nrm.append(np.asarray(model[1], dtype=np.float64))
            gray.append(np.asarray(colors, dtype=np.float64) @ [0.299, 0.587, 0.114] if colors is not None
                        else np.full(len(p), np.nan))
            cells.append(idx + m_off)
            lo.append(g_lo)
            shape.append(g_shape)
            offsets.append(g_off)
            ranges.append((m_off, m_off + len(p)))
            m_off += len(p)
            g_off += len(idx)
        self.m = torch.tensor(np.vstack(pts), dtype=dt, device=self.device)
        self.n = torch.tensor(np.vstack(nrm), dtype=dt, device=self.device)
        self.gray = torch.tensor(np.concatenate(gray), dtype=dt, device=self.device)
        self.ranges = ranges
        self.cells = torch.tensor(np.concatenate(cells), dtype=torch.long, device=self.device)
        self.lo = torch.tensor(np.stack(lo), dtype=dt, device=self.device)
        self.shape = torch.tensor(np.stack(shape), dtype=torch.long, device=self.device)
        self.g_off = torch.tensor(offsets, dtype=torch.long, device=self.device)
        self.centre = torch.stack([torch.tensor(p.mean(0), dtype=dt) for p in pts]).to(self.device)

    def _nearest(self, P: torch.Tensor, ob: torch.Tensor, T: torch.Tensor, grid: Tuple[torch.Tensor, ...]):
        """Nearest model point of each observed point (through the grid of its object) and its distance."""
        lo, shape, offset = grid
        R, t = T[ob, :3, :3], T[ob, :3, 3]
        Q = torch.einsum('nji,nj->ni', R, P - t)
        ijk = torch.floor((Q - lo[ob]) / self.grid_cell).long()
        shp = shape[ob]
        inside = ((ijk >= 0) & (ijk < shp)).all(1)
        ijk = torch.minimum(ijk.clamp(min=0), shp - 1)
        idx = self.cells[(ijk[:, 0] * shp[:, 1] + ijk[:, 1]) * shp[:, 2] + ijk[:, 2] + offset[ob]]
        mw = torch.einsum('nij,nj->ni', R, self.m[idx]) + t
        nw = torch.einsum('nij,nj->ni', R, self.n[idx])
        return mw, nw, (mw - P).norm(dim=1), inside

    def _photometric_system(self, photo: Dict[str, Any], T: torch.Tensor, c: torch.Tensor):
        """Normal equations of the photometric term, accumulated over the cameras."""
        K = len(T)
        H = torch.zeros(K, 6, 6, device=self.device)
        g = torch.zeros(K, 6, device=self.device)
        ob = photo['ob']
        pw = torch.einsum('nij,nj->ni', T[ob, :3, :3], self.m[photo['idx']]) + T[ob, :3, 3]
        for v in photo['views']:
            pc = (pw - v['t']) @ v['R']
            z = pc[:, 2].clamp(min=0.05)
            u = v['fx'] * pc[:, 0] / z + v['cx']
            w_px = v['fy'] * pc[:, 1] / z + v['cy']
            Hi, Wi = v['gray'].shape
            ok = (pc[:, 2] > 0.05) & (u >= 0) & (u <= Wi - 2) & (w_px >= 0) & (w_px <= Hi - 2)
            ui, vi = u.round().long().clamp(0, Wi - 1), w_px.round().long().clamp(0, Hi - 1)
            ok &= (v['mask'][vi, ui] == photo['oid']) & ((v['depth'][vi, ui] - pc[:, 2]).abs() < self.photometric_depth_tolerance)
            if not torch.any(ok):
                continue
            r = _bilinear(v['gray'], u, w_px) - photo['gray']
            # exposure differs between the scan and this camera: remove each object's median brightness offset
            for k in range(K):
                seen = ok & (ob == k)
                if torch.any(seen):
                    r = torch.where(ob == k, r - r[seen].median(), r)
            gx, gy = _bilinear(v['gx'], u, w_px), _bilinear(v['gy'], u, w_px)
            gc = torch.stack([gx * v['fx'] / z, gy * v['fy'] / z,
                              -(gx * v['fx'] * pc[:, 0] + gy * v['fy'] * pc[:, 1]) / z ** 2], 1)
            gw = gc @ v['R'].T
            J = torch.cat([torch.cross(pw - c[ob], gw, dim=1), gw], 1)
            w = self.photometric_scale * _huber_weight(r.abs(), self.photometric_huber) * ok
            H.index_add_(0, ob, J[:, :, None] * J[:, None, :] * w[:, None, None])
            g.index_add_(0, ob, J * (w * r)[:, None])
        return H, g

    def _solve(self, P: torch.Tensor, ob: torch.Tensor, T_ref: torch.Tensor, grid: Tuple[torch.Tensor, ...],
               centre: torch.Tensor, photo: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        K = len(T_ref)
        # T = S T_ref with S rotating about the model centre c (fixed for the frame): S p = Rs (p - c) + c + vs
        c = torch.einsum('kij,kj->ki', T_ref[:, :3, :3], centre) + T_ref[:, :3, 3]
        Rs = torch.eye(3, device=self.device).repeat(K, 1, 1)
        vs = torch.zeros(K, 3, device=self.device)
        T = T_ref.clone()
        prior = torch.diag_embed(self.lam.expand(K, 6))
        for it in range(self.iterations):
            md = max(self.max_distance, self.initial_max_distance * 0.6 ** it)
            mw, nw, d, inside = self._nearest(P, ob, T, grid)
            r = ((mw - P) * nw).sum(1)
            J = torch.cat([torch.cross(mw - c[ob], nw, dim=1), nw], 1)
            w = _huber_weight(r.abs(), self.huber) * (inside & (d < md))
            H = torch.zeros(K, 6, 6, device=self.device).index_add_(0, ob, J[:, :, None] * J[:, None, :] * w[:, None, None])
            g = torch.zeros(K, 6, device=self.device).index_add_(0, ob, J * (w * r)[:, None])
            if photo is not None and md <= self.max_distance:
                # the image gradient only holds within a pixel or two: photometry refines, geometry gets it there
                Hp, gp = self._photometric_system(photo, T, c)
                H, g = H + Hp, g + gp
            delta = -torch.linalg.solve(H + prior, g + self.lam * torch.cat([_rotvec(Rs), vs], 1))
            Rd = _expso3(delta[:, :3])
            Rs = Rd @ Rs
            vs = torch.einsum('kij,kj->ki', Rd, vs) + delta[:, 3:]
            S = torch.eye(4, device=self.device).repeat(K, 1, 1)
            S[:, :3, :3] = Rs
            S[:, :3, 3] = c - torch.einsum('kij,kj->ki', Rs, c) + vs
            T = S @ T_ref
            if md <= self.max_distance and float(delta.abs().max()) < 1e-5:
                break
        return T

    def _fit(self, P: torch.Tensor, ob: torch.Tensor, T: torch.Tensor, grid: Tuple[torch.Tensor, ...]):
        """Per object: fraction of observed points within max_distance of the model, and their mean Huber cost."""
        K = len(T)
        mw, nw, d, inside = self._nearest(P, ob, T, grid)
        ok = (inside & (d < self.max_distance)).float()
        a = ((mw - P) * nw).sum(1).abs()
        rho = torch.where(a <= self.huber, 0.5 * a ** 2, self.huber * (a - 0.5 * self.huber)) * ok
        n_obs = torch.zeros(K, device=self.device).index_add_(0, ob, torch.ones_like(ok))
        n_ok = torch.zeros(K, device=self.device).index_add_(0, ob, ok)
        cost = torch.zeros(K, device=self.device).index_add_(0, ob, rho)
        return n_ok / n_obs.clamp(min=1), cost / n_ok.clamp(min=1)

    def _photometric_data(self, oids: List[int], views: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Coloured model points of the tracked objects (evenly thinned) and the grey images with their gradients."""
        idx, ob, oid = [], [], []
        for k, o in enumerate(oids):
            a, b = self.ranges[self.slots[o]]
            sel = torch.arange(a, b, device=self.device)
            sel = sel[~torch.isnan(self.gray[sel])]
            if len(sel) > self.photometric_max_points:
                sel = sel[torch.linspace(0, len(sel) - 1, self.photometric_max_points, device=self.device).long()]
            idx.append(sel)
            ob.append(torch.full((len(sel),), k, dtype=torch.long, device=self.device))
            oid.append(torch.full((len(sel),), int(o), dtype=torch.long, device=self.device))
        idx = torch.cat(idx)
        if len(idx) == 0 or not views:
            return None
        prepared = []
        for v in views:
            gray = v['gray'].to(self.device, torch.float32)
            gx, gy = torch.zeros_like(gray), torch.zeros_like(gray)
            gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
            gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
            prepared.append({'gray': gray, 'gx': gx, 'gy': gy, 'depth': v['depth'].to(self.device, torch.float32),
                             'mask': v['mask'].to(self.device).long(), 'fx': float(v['fx']), 'fy': float(v['fy']),
                             'cx': float(v['cx']), 'cy': float(v['cy']),
                             'R': v['R_c2w'].to(self.device, torch.float32), 't': v['t_c2w'].to(self.device, torch.float32)})
        return {'idx': idx, 'ob': torch.cat(ob), 'oid': torch.cat(oid), 'gray': self.gray[idx], 'views': prepared}

    @torch.no_grad()
    def track(self, previous: Dict[int, torch.Tensor], observations: Dict[int, torch.Tensor],
              views: Optional[List[Dict[str, Any]]] = None) -> Dict[int, torch.Tensor]:
        """
        previous: object id -> its last pose (4x4, scan frame -> world); observations: object id -> its observed world
        points (N,3) this frame; views (photometric term only): per camera 'gray' (H,W) in [0, 1], 'depth', 'mask',
        'fx', 'fy', 'cx', 'cy', 'R_c2w', 't_c2w'. Returns the new pose of every observed object that has a model.
        """
        oids = [o for o in observations if o in self.slots and len(observations[o]) >= 4]
        if not oids:
            return {}
        sel = torch.tensor([self.slots[o] for o in oids], device=self.device)
        grid = (self.lo[sel], self.shape[sel], self.g_off[sel])
        centre = self.centre[sel]
        pts, obj = [], []
        for k, o in enumerate(oids):
            p, _ = voxel_average(observations[o].detach().cpu().double().numpy(), self.observation_voxel)
            pts.append(torch.tensor(p, dtype=torch.float32, device=self.device))
            obj.append(torch.full((len(p),), k, dtype=torch.long, device=self.device))
        P, ob = torch.cat(pts), torch.cat(obj)
        T_prev = torch.stack([previous[o].to(self.device, torch.float32) for o in oids])
        photo = self._photometric_data(oids, views) if self.photometric and views else None

        T = self._solve(P, ob, T_prev, grid, centre, photo)
        moving = []
        if self.motion_retry > 0:
            for k, o in enumerate(oids):
                if o in self.last_motion:
                    c = T_prev[k, :3, :3] @ centre[k] + T_prev[k, :3, 3]
                    M = self.last_motion[o]
                    if float((M[:3, :3] @ c + M[:3, 3] - c).norm()) > self.motion_retry:
                        moving.append(k)
        if moving:
            T_cv = T_prev.clone()
            for k in moving:
                T_cv[k] = self.last_motion[oids[k]] @ T_prev[k]
            T2 = self._solve(P, ob, T_cv, grid, centre, photo)
            f1, c1 = self._fit(P, ob, T, grid)
            f2, c2 = self._fit(P, ob, T2, grid)
            for k in moving:
                if (float(f2[k]), -float(c2[k])) > (float(f1[k]), -float(c1[k])):
                    T[k] = T2[k]
        for k, o in enumerate(oids):
            self.last_motion[o] = T[k] @ torch.linalg.inv(T_prev[k])
        return {o: T[k] for k, o in enumerate(oids)}
