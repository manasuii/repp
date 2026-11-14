"""Step 3: parametric 2D L-bracket generator (Gmsh).

Geometry (all lengths in metres, corner of the L at the origin):

        y
        ^
      H |  +--+            vertical leg  : x in [0, t],  y in [0, H]   (clamped at y = H)
        |  |  |            horizontal leg: x in [0, W],  y in [0, t]   (loaded at x = W)
        |  |  |
      t |  |  +--(r)--------+   <- inner fillet of radius r at (t, t)
        |  |      o hole     |      circular hole, diameter d = hole_ratio * t,
        +--+-----------------+--> x   centre at y = t/2
        0                    W

Six shape parameters (PARAM_NAMES):
    W          horizontal leg length
    H          vertical leg height
    t          arm thickness (both arms)
    r          inner fillet radius
    hole_ratio hole diameter / t          (0 -> no hole)
    hole_pos   hole position in [0, 1] along the free length of the horizontal leg

Boundary sets returned with every mesh:
    fixed_nodes  nodes on the top edge of the vertical leg (y = H)  -> clamp
    load_nodes   nodes on the end face of the horizontal leg (x = W) -> traction
    load_weights consistent nodal weights (sum = 1) for a uniform traction on that face

Mesh: linear (3-node) triangles, matching the CST solver in fem2d.py, with local
refinement near the fillet and the hole where stress concentrates.

Usage as a script (writes a gallery image so you can eyeball the designs):
    python bracket_generator.py --n 12 --family id --out bracket_gallery.png
"""
from __future__ import annotations

import argparse
import atexit
import time
import warnings
from dataclasses import dataclass

import numpy as np

PARAM_NAMES = ("W", "H", "t", "r", "hole_ratio", "hole_pos")

# Sampling bounds per geometry family.  "id" is the in-distribution family.
# The "ood_*" families are meant to be held out entirely (never used for training).
FAMILIES = {
    "id": dict(W=(0.20, 0.30), H=(0.14, 0.22), t=(0.020, 0.035),
               r=(0.004, 0.012), hole_ratio=(0.25, 0.55), hole_pos=(0.25, 0.75)),
    # larger, longer brackets (extrapolation in size / stiffness)
    "ood_scale": dict(W=(0.32, 0.40), H=(0.24, 0.30), t=(0.020, 0.035),
                      r=(0.004, 0.012), hole_ratio=(0.25, 0.55), hole_pos=(0.25, 0.75)),
    # thin arms with big fillets and big holes (extrapolation in stress-concentration features)
    "ood_features": dict(W=(0.20, 0.30), H=(0.14, 0.22), t=(0.012, 0.018),
                         r=(0.014, 0.020), hole_ratio=(0.55, 0.65), hole_pos=(0.25, 0.75)),
}


class InvalidGeometry(ValueError):
    """Raised when a parameter set cannot produce a valid bracket."""


# ----------------------------------------------------------------------------
# parameters, validation, sampling
# ----------------------------------------------------------------------------
def hole_geometry(p):
    """Returns (xc, yc, d). d = 0 means no hole."""
    t, W, r = p["t"], p["W"], p["r"]
    d = p["hole_ratio"] * t
    if d <= 0:
        return None, None, 0.0
    margin = 0.3 * d
    x_min = t + r + 0.5 * d + margin
    x_max = W - 0.5 * d - margin
    xc = x_min + p["hole_pos"] * (x_max - x_min)
    return xc, 0.5 * t, d


def validate_geometry(p):
    """Cheap geometric feasibility check. Returns (ok, reason)."""
    W, H, t, r = p["W"], p["H"], p["t"], p["r"]
    if min(W, H, t) <= 0 or r < 0:
        return False, "non-positive dimension"
    if W < 3 * t or H < 3 * t:
        return False, "legs too short relative to thickness"
    if r > 0.5 * min(W - t, H - t):
        return False, "fillet too large for the legs"
    if not (0.0 <= p["hole_pos"] <= 1.0):
        return False, "hole_pos outside [0, 1]"
    if p["hole_ratio"] != 0 and not (0.1 <= p["hole_ratio"] <= 0.7):
        return False, "hole_ratio outside [0.1, 0.7]"
    if p["hole_ratio"] > 0:
        d = p["hole_ratio"] * t
        margin = 0.3 * d
        x_min = t + r + 0.5 * d + margin
        x_max = W - 0.5 * d - margin
        if x_max - x_min < 0.2 * d:
            return False, "no room for the hole between fillet and tip"
    return True, ""


def analytic_area(p):
    W, H, t, r = p["W"], p["H"], p["t"], p["r"]
    area = W * t + t * (H - t) + r * r * (1.0 - np.pi / 4.0)
    area -= np.pi * (p["hole_ratio"] * t) ** 2 / 4.0
    return area


def scale_unit(u, bounds):
    """Map unit-cube samples (n, 6) to physical parameter dicts."""
    lo = np.array([bounds[k][0] for k in PARAM_NAMES])
    hi = np.array([bounds[k][1] for k in PARAM_NAMES])
    x = lo + np.asarray(u) * (hi - lo)
    return [dict(zip(PARAM_NAMES, map(float, row))) for row in x]


def sample_geometry(n, family="id", seed=0, max_rounds=100):
    """n valid parameter dicts from a scrambled Sobol sequence (rejection of invalid ones)."""
    from scipy.stats import qmc
    bounds = FAMILIES[family]
    sob = qmc.Sobol(d=len(PARAM_NAMES), scramble=True, seed=seed)
    out = []
    for _ in range(max_rounds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = sob.random(max(16, n))
        for p in scale_unit(u, bounds):
            if validate_geometry(p)[0]:
                out.append(p)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"Only {len(out)} valid samples for family {family!r}; check bounds")
    return out[:n]


# ----------------------------------------------------------------------------
# meshing
# ----------------------------------------------------------------------------
_GMSH_READY = False


def _gmsh():
    """Initialise Gmsh once per process and return the module."""
    global _GMSH_READY
    import gmsh
    if not _GMSH_READY:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        atexit.register(_finalize_gmsh)
        _GMSH_READY = True
    return gmsh


def _finalize_gmsh():
    global _GMSH_READY
    if _GMSH_READY:
        import gmsh
        gmsh.finalize()
        _GMSH_READY = False


@dataclass
class BracketMesh:
    params: dict
    nodes: np.ndarray          # (nn, 2) float64
    tris: np.ndarray           # (ne, 3) int, counter-clockwise
    fixed_nodes: np.ndarray    # clamp (top of vertical leg)
    load_nodes: np.ndarray     # tip face of horizontal leg, sorted by y
    load_weights: np.ndarray   # consistent nodal weights, sum = 1
    area: float
    min_angle_deg: float
    hole: tuple                # (xc, yc, d)  d = 0 -> no hole


def triangle_min_angle_deg(nodes, tris):
    p = nodes[tris]
    a = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    cosA = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1, 1)
    cosB = np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1, 1)
    cosC = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1, 1)
    ang = np.degrees(np.arccos(np.stack([cosA, cosB, cosC], axis=1)))
    return float(ang.min())


def build_mesh(p, n_thick=6, refine=2.0, min_angle_deg=12.0, check_area=True):
    """Generate the bracket mesh for parameter dict p.

    n_thick : target number of elements through the arm thickness (global size = t / n_thick)
    refine  : size reduction factor near the fillet and the hole
    """
    ok, why = validate_geometry(p)
    if not ok:
        raise InvalidGeometry(why)

    gmsh = _gmsh()
    gmsh.clear()
    gmsh.model.add("bracket")
    geo = gmsh.model.geo
    W, H, t, r = p["W"], p["H"], p["t"], p["r"]
    h = t / n_thick
    h_fine = h / refine

    # ---- outer outline, counter-clockwise ------------------------------------
    if r > 0:
        corner = [(0, 0), (W, 0), (W, t), (t + r, t), (t, t + r), (t, H), (0, H)]
    else:
        corner = [(0, 0), (W, 0), (W, t), (t, t), (t, H), (0, H)]
    pid = [geo.addPoint(x, y, 0.0, h) for x, y in corner]
    lines, fine_curves = [], []
    n = len(pid)
    for i in range(n):
        a, b = pid[i], pid[(i + 1) % n]
        if r > 0 and i == 3:                       # fillet arc (t+r,t) -> (t,t+r)
            c = geo.addPoint(t + r, t + r, 0.0, h)
            arc = geo.addCircleArc(a, c, b)
            lines.append(arc)
            fine_curves.append(arc)
        else:
            lines.append(geo.addLine(a, b))
    loops = [geo.addCurveLoop(lines)]

    # ---- hole ------------------------------------------------------------------
    xc, yc, d = hole_geometry(p)
    if d > 0:
        a = 0.5 * d
        c = geo.addPoint(xc, yc, 0.0, h_fine)
        q = [geo.addPoint(xc + a, yc, 0.0, h_fine), geo.addPoint(xc, yc + a, 0.0, h_fine),
             geo.addPoint(xc - a, yc, 0.0, h_fine), geo.addPoint(xc, yc - a, 0.0, h_fine)]
        arcs = [geo.addCircleArc(q[i], c, q[(i + 1) % 4]) for i in range(4)]
        fine_curves += arcs
        loops.append(geo.addCurveLoop(arcs))

    geo.addPlaneSurface(loops)
    geo.synchronize()

    # ---- size field: fine near fillet / hole, coarse elsewhere -------------------
    fld = gmsh.model.mesh.field
    if fine_curves:
        fld.add("Distance", 1)
        fld.setNumbers(1, "CurvesList", fine_curves)
        fld.setNumber(1, "Sampling", 60)
        fld.add("Threshold", 2)
        fld.setNumber(2, "InField", 1)
        fld.setNumber(2, "SizeMin", h_fine)
        fld.setNumber(2, "SizeMax", h)
        fld.setNumber(2, "DistMin", 1.0 * h)
        fld.setNumber(2, "DistMax", 3.0 * h)
        fld.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0 if fine_curves else 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0 if fine_curves else 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.Algorithm", 6)            # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(2)

    # ---- extract ---------------------------------------------------------------
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(xyz).reshape(-1, 3)[:, :2]
    _, tri_tags = gmsh.model.mesh.getElementsByType(2)     # 3-node triangles
    tri_tags = np.asarray(tri_tags, dtype=np.int64).reshape(-1, 3)
    lut = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    lut[np.asarray(tags, dtype=np.int64)] = np.arange(len(tags))
    tris = lut[tri_tags]
    # drop nodes not used by any triangle (arc centres, hole centre)
    used = np.unique(tris)
    remap = np.full(len(xyz), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes = xyz[used].astype(float)
    tris = remap[tris]

    # enforce counter-clockwise orientation
    sa = 0.5 * ((nodes[tris[:, 1], 0] - nodes[tris[:, 0], 0]) * (nodes[tris[:, 2], 1] - nodes[tris[:, 0], 1])
                - (nodes[tris[:, 2], 0] - nodes[tris[:, 0], 0]) * (nodes[tris[:, 1], 1] - nodes[tris[:, 0], 1]))
    flip = sa < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    area = float(np.abs(sa).sum())

    if check_area and abs(area - analytic_area(p)) / analytic_area(p) > 0.01:
        raise InvalidGeometry(f"mesh area {area:.6e} differs from analytic {analytic_area(p):.6e}")

    min_ang = triangle_min_angle_deg(nodes, tris)
    if min_ang < min_angle_deg:
        raise InvalidGeometry(f"poor mesh quality: min angle {min_ang:.1f} deg")

    # ---- boundary sets -----------------------------------------------------------
    tol = 1e-9 * max(W, H)
    fixed = np.where(nodes[:, 1] >= H - tol)[0]
    load = np.where(nodes[:, 0] >= W - tol)[0]
    load = load[np.argsort(nodes[load, 1])]
    seg = np.diff(nodes[load, 1])
    w = np.zeros(len(load))
    w[:-1] += 0.5 * seg
    w[1:] += 0.5 * seg
    w /= w.sum()
    if len(fixed) < 2 or len(load) < 2:
        raise InvalidGeometry("boundary sets not found")

    return BracketMesh(params=dict(p), nodes=nodes, tris=tris, fixed_nodes=fixed,
                       load_nodes=load, load_weights=w, area=area,
                       min_angle_deg=min_ang, hole=(xc, yc, d))


# ----------------------------------------------------------------------------
# preview gallery
# ----------------------------------------------------------------------------
def plot_gallery(meshes, path, ncols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    n = len(meshes)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, m in zip(axes.ravel(), meshes):
        ax.axis("on")
        tp = mtri.Triangulation(m.nodes[:, 0], m.nodes[:, 1], m.tris)
        ax.triplot(tp, lw=0.3, color="0.35")
        ax.plot(m.nodes[m.fixed_nodes, 0], m.nodes[m.fixed_nodes, 1], "r.", ms=3)
        ax.plot(m.nodes[m.load_nodes, 0], m.nodes[m.load_nodes, 1], "b.", ms=3)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        p = m.params
        ax.set_title(f"W={p['W']*1e3:.0f} H={p['H']*1e3:.0f} t={p['t']*1e3:.0f} r={p['r']*1e3:.0f}mm\n"
                     f"{len(m.tris)} tris, min angle {m.min_angle_deg:.0f} deg", fontsize=8)
    fig.suptitle("red = clamped edge, blue = loaded edge", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(description="Preview random parametric brackets")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--family", default="id", choices=sorted(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-thick", type=int, default=6)
    ap.add_argument("--out", default="bracket_gallery.png")
    a = ap.parse_args()

    meshes, t0 = [], time.time()
    for p in sample_geometry(a.n, a.family, a.seed):
        try:
            meshes.append(build_mesh(p, n_thick=a.n_thick))
        except InvalidGeometry as e:
            print("skipped:", e)
    dt = time.time() - t0
    ne = [len(m.tris) for m in meshes]
    print(f"{len(meshes)} meshes in {dt:.1f}s | triangles: min {min(ne)}, mean {np.mean(ne):.0f}, max {max(ne)}")
    print(f"worst min-angle: {min(m.min_angle_deg for m in meshes):.1f} deg")
    plot_gallery(meshes, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    _main()
