"""Step 4: dataset pipeline for the parametric bracket elastodynamics surrogate.

Workflow (each stage is a sub-command and can be re-run safely):

    python dataset_pipeline.py plan        --root data/v1 --n-id 2000 --n-ood 300 --seed 0
    python dataset_pipeline.py generate    --root data/v1 --workers 4
    python dataset_pipeline.py consolidate --root data/v1
    python dataset_pipeline.py inspect     --root data/v1 --id 5
    python dataset_pipeline.py smoke       --root /tmp/bracket_smoke      # 12 cases end-to-end

What each stage does
--------------------
plan         Samples geometry (Sobol), material, and load for every case, assigns splits,
             and freezes everything in <root>/manifest.json.  Splits are fixed here, before
             any model sees data.  The held-out geometry family becomes split "test_ood".
generate     Runs the FE solver for every case in parallel (one process per worker), writing
             one HDF5 file per case to <root>/cases/.  Resumable: finished cases are skipped.
             Failures are logged to <root>/failures.jsonl and never stop the run.
consolidate  Merges the case files into <root>/dataset.h5 with an index, splits, and
             normalisation statistics computed on the TRAIN split only.
inspect      Plots one case (von Mises map at the peak, load, tip response).

Physics set-up (all SI units): linear plane-stress elastodynamics, CST triangles, clamped top
of the vertical leg, vertical tip traction on the end of the horizontal leg, Rayleigh damping
tuned to the requested damping ratio at two modes, Newmark average acceleration.
All cases share ONE time grid (T_total, nt), which a fixed-size time input needs.

Linear elasticity is only meaningful below yield: `consolidate` reports how many cases exceed
250 MPa peak von Mises so you can tighten the load range if needed.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
import traceback
import warnings

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.stats import qmc

import bracket_generator as bg
from fem2d import ElastoDynamics2D

# ----------------------------------------------------------------------------
# configuration (frozen into the manifest by `plan`)
# ----------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    b_out_of_plane=0.02,             # m, plane-stress thickness
    T_total=0.03,                    # s, simulated duration (same for every case)
    nt=600,                          # number of time steps (dt = T_total / nt)
    store_every=6,                   # keep every k-th frame of the fields (QoIs use all steps)
    store_vm=True,                   # also store nodal von Mises history
    n_thick=6,                       # elements through the arm thickness
    min_angle_deg=12.0,
    n_modes=6,
    damping_modes=[0, 2],            # Rayleigh calibrated at these (0-based) modes
    material=dict(E=[70e9, 210e9], rho=[2700.0, 7850.0], nu=[0.28, 0.34], zeta=[0.005, 0.05]),
    load=dict(amp=[100.0, 500.0],    # N, total tip force
              f_range=[100.0, 1000.0],   # Hz
              n_sines=5, ramp_frac=0.05,
              kinds=["multisine", "pulse", "sweep"]),
    split_fractions=dict(train=0.8, val=0.1, test_id=0.1),
    ood_family="ood_scale",
    yield_check_mpa=250.0,
)

MAT_NAMES = ("E", "rho", "nu", "zeta")
QOI_NAMES = ("peak_vm_elem", "peak_vm_node", "max_disp", "max_tip_uy", "freq1_hz", "freq2_hz",
             "freq3_hz", "freq4_hz", "freq5_hz")


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def case_path(root, cid):
    return os.path.join(root, "cases", f"case_{cid:06d}.h5")


def load_manifest(root):
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def make_load_signal(kind, seed, t, load_cfg):
    """Peak-normalised load shape s(t) in [-1, 1] with s(0) = 0 (system starts at rest)."""
    rng = np.random.default_rng(seed)
    T = t[-1]
    fmin, fmax = load_cfg["f_range"]
    ramp = np.clip(t / (load_cfg["ramp_frac"] * T), 0.0, 1.0)
    ramp = 0.5 * (1.0 - np.cos(np.pi * ramp))
    if kind == "multisine":
        k = load_cfg["n_sines"]
        f = rng.uniform(fmin, fmax, k)
        a = rng.uniform(0.3, 1.0, k)
        ph = rng.uniform(0.0, 2 * np.pi, k)
        s = (a[:, None] * np.sin(2 * np.pi * f[:, None] * t[None, :] + ph[:, None])).sum(0) * ramp
    elif kind == "pulse":
        dur = rng.uniform(0.10, 0.40) * T
        onset = rng.uniform(0.0, 0.20) * T
        sign = rng.choice([-1.0, 1.0])
        s = sign * np.where((t >= onset) & (t < onset + dur), np.sin(np.pi * (t - onset) / dur), 0.0)
    elif kind == "sweep":
        f0 = rng.uniform(fmin, 0.5 * (fmin + fmax))
        f1 = rng.uniform(f0 + 0.2 * (fmax - fmin), fmax)
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t**2)
        s = np.sin(phase) * ramp
    else:
        raise ValueError(f"unknown load kind {kind!r}")
    peak = np.abs(s).max()
    return s / peak if peak > 0 else s


def nodal_average_operator(m):
    """Sparse (nn, ne) operator: area-weighted average of element values onto nodes."""
    rows = m.tris.ravel()
    cols = np.repeat(np.arange(m.ne), 3)
    A = sp.coo_matrix((np.repeat(m.area, 3), (rows, cols)), shape=(m.nn, m.ne)).tocsr()
    return sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A


def _h5_write(f, name, data, compress=True):
    if compress and np.asarray(data).size > 64:
        f.create_dataset(name, data=data, compression="gzip", compression_opts=4)
    else:
        f.create_dataset(name, data=data)


# ----------------------------------------------------------------------------
# stage 1: plan
# ----------------------------------------------------------------------------
def plan(root, n_id, n_ood, seed, force=False, cfg_overrides=None):
    os.makedirs(os.path.join(root, "cases"), exist_ok=True)
    mpath = os.path.join(root, "manifest.json")
    if os.path.exists(mpath) and not force:
        sys.exit(f"{mpath} exists. Use --force to overwrite (this orphans existing case files).")

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    geo_id = bg.sample_geometry(n_id, "id", seed)
    geo_ood = bg.sample_geometry(n_ood, cfg["ood_family"], seed + 1) if n_ood > 0 else []
    geoms = geo_id + geo_ood
    families = ["id"] * len(geo_id) + [cfg["ood_family"]] * len(geo_ood)
    n = len(geoms)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = qmc.Sobol(d=len(MAT_NAMES), scramble=True, seed=seed + 2).random(n)
    mats = []
    for row in u:
        mats.append({k: float(lo + x * (hi - lo))
                     for x, k in zip(row, MAT_NAMES)
                     for lo, hi in [cfg["material"][k]]})

    rng = np.random.default_rng(seed + 3)
    lc = cfg["load"]
    kinds = rng.choice(lc["kinds"], size=n)
    amps = rng.uniform(lc["amp"][0], lc["amp"][1], size=n)
    load_seeds = rng.integers(0, 2**31 - 1, size=n)

    # splits: random within the in-distribution family, the whole OOD family held out
    split = np.empty(n, dtype=object)
    perm = rng.permutation(len(geo_id))
    fr = cfg["split_fractions"]
    n_tr = int(round(fr["train"] * len(geo_id)))
    n_va = int(round(fr["val"] * len(geo_id)))
    for rank, i in enumerate(perm):
        split[i] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test_id")
    split[len(geo_id):] = "test_ood"

    cases = [dict(id=i, split=str(split[i]), family=families[i], geometry=geoms[i], material=mats[i],
                  load=dict(kind=str(kinds[i]), amp=float(amps[i]), seed=int(load_seeds[i])))
             for i in range(n)]

    with open(mpath, "w") as f:
        json.dump(dict(config=cfg, seed=seed, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                       cases=cases), f, indent=1)

    counts = {s: int(sum(c["split"] == s for c in cases)) for s in ("train", "val", "test_id", "test_ood")}
    est = estimate_case_mb(cfg)
    print(f"planned {n} cases -> {mpath}")
    print("splits:", counts)
    print(f"estimated disk per case ~{est:.1f} MB (uncompressed fields); "
          f"total ~{est * n / 1024:.1f} GB before gzip")


def estimate_case_mb(cfg, n_nodes=1700):
    frames = len(range(0, cfg["nt"] + 1, cfg["store_every"]))
    per = frames * n_nodes * (2 + (1 if cfg["store_vm"] else 0)) * 4
    return per / 1e6


# ----------------------------------------------------------------------------
# stage 2: generate
# ----------------------------------------------------------------------------
def run_case(spec, cfg):
    """Solve one case and return a dict of arrays/scalars ready to be written."""
    t_start = time.time()
    p, mat, ld = spec["geometry"], spec["material"], spec["load"]

    mesh = bg.build_mesh(p, n_thick=cfg["n_thick"], min_angle_deg=cfg["min_angle_deg"])
    m = ElastoDynamics2D(mesh.nodes, mesh.tris, mat["E"], mat["nu"], mat["rho"],
                         cfg["b_out_of_plane"], plane="stress")
    m.fix_nodes(mesh.fixed_nodes)

    w, _ = m.modal(cfg["n_modes"])
    i, j = cfg["damping_modes"]
    m.set_rayleigh_from_modes(w[i], w[j], mat["zeta"])

    nt, T = cfg["nt"], cfg["T_total"]
    dt = T / nt
    t = np.arange(nt + 1) * dt
    s = make_load_signal(ld["kind"], ld["seed"], t, cfg["load"])
    P = np.zeros(m.ndof)
    P[2 * mesh.load_nodes + 1] = -ld["amp"] * mesh.load_weights       # downward tip force
    U, _, _ = m.transient(P, s, dt)
    nn = m.nn
    U3 = U.reshape(nt + 1, nn, 2)
    if not np.all(np.isfinite(U)):
        raise FloatingPointError("non-finite displacements")

    # stress post-processing in time chunks (bounded memory), QoIs from EVERY step
    S = nodal_average_operator(m)
    keep_idx = np.arange(0, nt + 1, cfg["store_every"])
    keep_set = set(keep_idx.tolist())
    peak_el, peak_node, peak_loc, peak_t = 0.0, 0.0, (0.0, 0.0), 0.0
    vm_frames = []
    cent = mesh.nodes[mesh.tris].mean(axis=1)
    chunk = 100
    for a in range(0, nt + 1, chunk):
        b = min(a + chunk, nt + 1)
        vm_el = m.von_mises(m.element_stress(U[a:b]))                  # (c, ne)
        vm_nd = (S @ vm_el.T).T                                        # (c, nn)
        k = np.unravel_index(np.argmax(vm_el), vm_el.shape)
        if vm_el[k] > peak_el:
            peak_el, peak_loc, peak_t = float(vm_el[k]), tuple(cent[k[1]]), float((a + k[0]) * dt)
        peak_node = max(peak_node, float(vm_nd.max()))
        if cfg["store_vm"]:
            rows = [r - a for r in range(a, b) if r in keep_set]
            if rows:
                vm_frames.append(vm_nd[rows].astype(np.float32))

    tip_uy = U3[:, mesh.load_nodes, 1] @ mesh.load_weights
    qoi = dict(peak_vm_elem=peak_el, peak_vm_node=peak_node,
               max_disp=float(np.linalg.norm(U3, axis=2).max()),
               max_tip_uy=float(np.abs(tip_uy).max()))
    for q in range(5):
        qoi[f"freq{q + 1}_hz"] = float(w[q] / (2 * np.pi))
    if not all(np.isfinite(list(qoi.values()))):
        raise FloatingPointError("non-finite QoI")

    out = dict(
        nodes=mesh.nodes, tris=mesh.tris.astype(np.int32),
        fixed_nodes=mesh.fixed_nodes.astype(np.int32), load_nodes=mesh.load_nodes.astype(np.int32),
        load_weights=mesh.load_weights, t_store=t[keep_idx].astype(np.float32),
        load_shape=s.astype(np.float32), load_force=(ld["amp"] * s).astype(np.float32),
        disp=U3[keep_idx].astype(np.float32),
        tip_uy=tip_uy.astype(np.float32),
        params=np.array([p[k] for k in bg.PARAM_NAMES] + [mat[k] for k in MAT_NAMES], dtype=np.float64),
        attrs=dict(id=spec["id"], split=spec["split"], family=spec["family"],
                   load_kind=ld["kind"], load_amp=ld["amp"], dt=dt, nt=nt,
                   store_every=cfg["store_every"], min_angle_deg=mesh.min_angle_deg,
                   n_nodes=nn, n_tris=m.ne, area=mesh.area, peak_vm_time=peak_t,
                   peak_vm_x=peak_loc[0], peak_vm_y=peak_loc[1],
                   hole_xc=-1.0 if mesh.hole[0] is None else mesh.hole[0], hole_d=mesh.hole[2],
                   runtime_s=time.time() - t_start),
        qoi=qoi)
    if cfg["store_vm"]:
        out["vm"] = np.concatenate(vm_frames, axis=0)
    return out


def write_case(path, out):
    tmp = path + ".part"
    with h5py.File(tmp, "w") as f:
        for k, v in out.items():
            if k in ("attrs", "qoi"):
                continue
            _h5_write(f, k, v)
        for k, v in out["attrs"].items():
            f.attrs[k] = v
        g = f.create_group("qoi")
        for k, v in out["qoi"].items():
            g.attrs[k] = v
        f.attrs["param_names"] = np.array(list(bg.PARAM_NAMES) + list(MAT_NAMES), dtype="S16")
    os.replace(tmp, path)


def _worker(task):
    spec, root, cfg = task
    t0 = time.time()
    try:
        out = run_case(spec, cfg)
        write_case(case_path(root, spec["id"]), out)
        return dict(id=spec["id"], ok=True, seconds=time.time() - t0,
                    n_tris=out["attrs"]["n_tris"])
    except Exception as e:                                  # noqa: BLE001 - we log everything
        return dict(id=spec["id"], ok=False, seconds=time.time() - t0,
                    error=f"{type(e).__name__}: {e}",
                    trace="".join(traceback.format_exc().splitlines(True)[-4:]))


def generate(root, workers, limit=None):
    man = load_manifest(root)
    cfg, cases = man["config"], man["cases"]
    os.makedirs(os.path.join(root, "cases"), exist_ok=True)
    todo = [c for c in cases if not os.path.exists(case_path(root, c["id"]))]
    done_before = len(cases) - len(todo)
    if limit:
        todo = todo[:limit]
    print(f"{len(cases)} planned | {done_before} already done | running {len(todo)} with {workers} worker(s)")
    if not todo:
        return

    fail_path = os.path.join(root, "failures.jsonl")
    tasks = [(c, root, cfg) for c in todo]
    n_ok = n_fail = 0
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=100) as pool:
        for res in pool.imap_unordered(_worker, tasks, chunksize=1):
            if res["ok"]:
                n_ok += 1
            else:
                n_fail += 1
                with open(fail_path, "a") as f:
                    f.write(json.dumps(res) + "\n")
            n = n_ok + n_fail
            if n % max(1, len(tasks) // 50) == 0 or n == len(tasks):
                el = time.time() - t0
                eta = el / n * (len(tasks) - n)
                print(f"  {n}/{len(tasks)} done  ok={n_ok} failed={n_fail}  "
                      f"elapsed {el / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)
    print(f"finished: {n_ok} ok, {n_fail} failed" + (f" (see {fail_path})" if n_fail else ""))


# ----------------------------------------------------------------------------
# stage 3: consolidate
# ----------------------------------------------------------------------------
class _Acc:
    """Running mean / std over many arrays (float64 accumulators)."""
    def __init__(self, dim=1):
        self.n = 0
        self.s = np.zeros(dim)
        self.ss = np.zeros(dim)

    def add(self, x):                       # x: (..., dim)
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.s.size)
        self.n += len(x)
        self.s += x.sum(0)
        self.ss += (x**2).sum(0)

    def mean_std(self):
        mean = self.s / max(self.n, 1)
        var = np.maximum(self.ss / max(self.n, 1) - mean**2, 0.0)
        return mean, np.maximum(np.sqrt(var), 1e-12)


def consolidate(root, out_name="dataset.h5"):
    man = load_manifest(root)
    cases = [c for c in man["cases"] if os.path.exists(case_path(root, c["id"]))]
    if not cases:
        sys.exit("no finished case files found")
    out_path = os.path.join(root, out_name)

    acc_disp, acc_vm, acc_force = _Acc(2), _Acc(1), _Acc(1)
    ids, splits, families, params, qois = [], [], [], [], []
    with h5py.File(out_path, "w") as dst:
        grp = dst.create_group("samples")
        for c in cases:
            with h5py.File(case_path(root, c["id"]), "r") as src:
                g = grp.create_group(str(c["id"]))
                for k in src:
                    if k == "qoi":
                        continue
                    src.copy(k, g)
                for k, v in src.attrs.items():
                    g.attrs[k] = v
                q = {k: float(v) for k, v in src["qoi"].attrs.items()}
                for k, v in q.items():
                    g.attrs["qoi_" + k] = v
                ids.append(c["id"]); splits.append(c["split"]); families.append(c["family"])
                params.append(src["params"][()]); qois.append([q[k] for k in QOI_NAMES])
                if c["split"] == "train":
                    acc_disp.add(src["disp"][()])
                    if "vm" in src:
                        acc_vm.add(src["vm"][()])
                    acc_force.add(src["load_force"][()])

        ids = np.array(ids); params = np.array(params); qois = np.array(qois)
        idx = dst.create_group("index")
        idx["ids"] = ids
        idx.create_dataset("split", data=np.array(splits, dtype="S10"))
        idx.create_dataset("family", data=np.array(families, dtype="S16"))
        idx["params"] = params
        idx.attrs["param_names"] = np.array(list(bg.PARAM_NAMES) + list(MAT_NAMES), dtype="S16")
        idx["qoi"] = qois
        idx.attrs["qoi_names"] = np.array(QOI_NAMES, dtype="S16")

        tr = np.array(splits) == "train"
        nrm = dst.create_group("norm")                      # statistics from TRAIN only
        nrm["disp_mean"], nrm["disp_std"] = acc_disp.mean_std()
        if acc_vm.n:
            nrm["vm_mean"], nrm["vm_std"] = acc_vm.mean_std()
        nrm["force_std"] = acc_force.mean_std()[1]
        nrm["param_mean"], nrm["param_std"] = params[tr].mean(0), np.maximum(params[tr].std(0), 1e-12)
        nrm["qoi_mean"], nrm["qoi_std"] = qois[tr].mean(0), np.maximum(qois[tr].std(0), 1e-12)
        dst.attrs["config"] = json.dumps(man["config"])

    split_ids = {s: [int(i) for i in ids[np.array(splits) == s]]
                 for s in ("train", "val", "test_id", "test_ood")}
    with open(os.path.join(root, "splits.json"), "w") as f:
        json.dump(split_ids, f)

    y = man["config"]["yield_check_mpa"] * 1e6
    over = float((qois[:, QOI_NAMES.index("peak_vm_elem")] > y).mean())
    print(f"wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB, {len(ids)} cases)")
    print("split sizes:", {s: len(v) for s, v in split_ids.items()})
    print(f"{100 * over:.1f}% of cases exceed {man['config']['yield_check_mpa']:.0f} MPa peak von Mises"
          + ("  -> consider lowering the load amplitude range" if over > 0.05 else ""))
    f1 = qois[:, QOI_NAMES.index("freq1_hz")]
    print(f"first natural frequency range: {f1.min():.0f} - {f1.max():.0f} Hz")


# ----------------------------------------------------------------------------
# stage 4: inspect
# ----------------------------------------------------------------------------
def inspect_case(root, cid, out="inspect_case.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    with h5py.File(case_path(root, cid), "r") as f:
        nodes, tris = f["nodes"][()], f["tris"][()]
        vm, t_store = f["vm"][()], f["t_store"][()]
        force, tip = f["load_force"][()], f["tip_uy"][()]
        a = dict(f.attrs)
        q = dict(f["qoi"].attrs)
    nt, dt = int(a["nt"]), float(a["dt"])
    tt = np.arange(nt + 1) * dt
    k = int(np.argmax(vm.max(axis=1)))

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].plot(tt * 1e3, force); ax[0].set(xlabel="time [ms]", ylabel="tip force [N]",
                                           title=f"load ({a['load_kind']})")
    ax[1].plot(tt * 1e3, tip * 1e3); ax[1].set(xlabel="time [ms]", ylabel="tip uy [mm]",
                                               title="tip response")
    tp = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    im = ax[2].tricontourf(tp, vm[k] / 1e6, levels=20)
    ax[2].set(aspect="equal", title=f"von Mises [MPa], t = {t_store[k] * 1e3:.1f} ms")
    fig.colorbar(im, ax=ax[2])
    fig.suptitle(f"case {cid} | split {a['split'].decode() if isinstance(a['split'], bytes) else a['split']} | "
                 f"{int(a['n_tris'])} tris | f1 = {q['freq1_hz']:.0f} Hz | "
                 f"peak vM = {q['peak_vm_elem'] / 1e6:.1f} MPa", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan"); p.add_argument("--root", required=True)
    p.add_argument("--n-id", type=int, default=2000); p.add_argument("--n-ood", type=int, default=300)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--force", action="store_true")

    p = sub.add_parser("generate"); p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("consolidate"); p.add_argument("--root", required=True)

    p = sub.add_parser("inspect"); p.add_argument("--root", required=True)
    p.add_argument("--id", type=int, default=0); p.add_argument("--out", default="inspect_case.png")

    p = sub.add_parser("smoke"); p.add_argument("--root", default="smoke_data")
    p.add_argument("--workers", type=int, default=2)

    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.root, a.n_id, a.n_ood, a.seed, a.force)
    elif a.cmd == "generate":
        generate(a.root, a.workers, a.limit)
    elif a.cmd == "consolidate":
        consolidate(a.root)
    elif a.cmd == "inspect":
        inspect_case(a.root, a.id, a.out)
    elif a.cmd == "smoke":
        plan(a.root, n_id=10, n_ood=4, seed=0, force=True)
        generate(a.root, a.workers)
        consolidate(a.root)
        inspect_case(a.root, 0, os.path.join(a.root, "inspect_case.png"))


if __name__ == "__main__":
    _cli()
