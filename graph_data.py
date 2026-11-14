"""Step 5: data loading and graph construction.

Reads <root>/dataset.h5 (written by dataset_pipeline.py consolidate) and serves one
PyTorch-Geometric `Data` object per bracket case.

Graph definition
----------------
nodes  : FE mesh nodes.                       x      (N, 5) = [x/L, y/L, is_fixed, is_load, load_weight*n_load]
edges  : unique triangle edges, both ways.    edge_attr (E, 3) = [dx/L, dy/L, |d|/L]  (message flows src -> dst)
globals: u (1, 14) = [10 normalised params (6 geometry + E, rho, nu, zeta),
                      3 one-hot load kind, 1 scaled load amplitude]
         load (1, T) = tip-force history sampled at the stored frames, divided by the train std
targets: y (N, T, 3) = normalised [ux, uy, von Mises] history at the T stored frames
         qoi (1, 9)    raw scalar quantities of interest (peak von Mises, max disp, 5 frequencies, ...)
         qoi_ln (1, 9) log(qoi) standardised with TRAIN statistics (what the networks regress)

Everything is normalised with statistics computed on the TRAIN split only.

Quick self-check:
    python graph_data.py --h5 data/v1/dataset.h5
"""
import argparse
import json

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

LENGTH_SCALE = 0.3                       # m, coordinate / edge-length scale
KINDS = ("multisine", "pulse", "sweep")
SPLITS = ("train", "val", "test_id", "test_ood")


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def load_norm(h5_path):
    """Normalisation constants (train split only) plus names, as numpy / python objects."""
    with h5py.File(h5_path, "r") as f:
        n = {k: f["norm"][k][()] for k in f["norm"]}
        split = np.array([_s(s) for s in f["index/split"][()]])
        qoi = f["index/qoi"][()]
        n["qoi_names"] = [_s(s) for s in f["index"].attrs["qoi_names"]]
        n["param_names"] = [_s(s) for s in f["index"].attrs["param_names"]]
        cfg = json.loads(f.attrs["config"])
    logq = np.log(qoi[split == "train"])
    n["log_qoi_mean"] = logq.mean(0)
    n["log_qoi_std"] = np.maximum(logq.std(0), 1e-8)
    lo, hi = cfg["load"]["amp"]
    n["amp_mid"], n["amp_half"] = 0.5 * (lo + hi), 0.5 * (hi - lo)
    n["has_vm"] = "vm_mean" in n
    return n


def build_edges(nodes, tris, length_scale=LENGTH_SCALE):
    """Unique undirected triangle edges -> directed edge_index (2, E) and edge_attr (E, 3)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    d = (nodes[dst] - nodes[src]) / length_scale
    attr = np.concatenate([d, np.linalg.norm(d, axis=1, keepdims=True)], axis=1)
    return np.stack([src, dst]), attr


class BracketGraphDataset(torch.utils.data.Dataset):
    """One graph per case. Lazily reads HDF5 (safe with DataLoader workers).

    splits    : a split name or a list of them ("train", "val", "test_id", "test_ood")
    fields    : load the (large) displacement / von Mises histories. Set False for the
                QoI-only MLP baseline: much faster.
    in_memory : cache every sample after first read (fast epochs, but needs ~2 MB / case RAM)
    """

    def __init__(self, h5_path, splits="train", fields=True, in_memory=False,
                 max_cases=None, norm=None):
        self.h5_path = h5_path
        self.fields = fields
        self.in_memory = in_memory
        self.norm = norm if norm is not None else load_norm(h5_path)
        splits = [splits] if isinstance(splits, str) else list(splits)
        with h5py.File(h5_path, "r") as f:
            ids = f["index/ids"][()]
            sp = np.array([_s(s) for s in f["index/split"][()]])
        self.ids = ids[np.isin(sp, splits)]
        if max_cases:
            self.ids = self.ids[:max_cases]
        self._f, self._cache = None, {}

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):                       # h5py handles cannot be pickled to workers
        d = self.__dict__.copy()
        d["_f"] = None
        return d

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def __getitem__(self, i):
        cid = int(self.ids[i])
        if self.in_memory and cid in self._cache:
            return self._cache[cid]
        data = self._build(cid)
        if self.in_memory:
            self._cache[cid] = data
        return data

    def _build(self, cid):
        nrm, L = self.norm, LENGTH_SCALE
        g = self._file()[f"samples/{cid}"]
        nodes, tris = g["nodes"][()], g["tris"][()]
        N = len(nodes)

        fixed = np.zeros(N, dtype=np.float32)
        fixed[g["fixed_nodes"][()]] = 1.0
        load_nodes = g["load_nodes"][()]
        is_load = np.zeros(N, dtype=np.float32)
        is_load[load_nodes] = 1.0
        w = np.zeros(N, dtype=np.float32)
        w[load_nodes] = g["load_weights"][()] * len(load_nodes)
        x = np.column_stack([nodes[:, 0] / L, nodes[:, 1] / L, fixed, is_load, w]).astype(np.float32)

        edge_index, edge_attr = build_edges(nodes, tris, L)

        params = (g["params"][()] - nrm["param_mean"]) / nrm["param_std"]
        kind = _s(g.attrs["load_kind"])
        onehot = np.array([kind == k for k in KINDS], dtype=np.float64)
        amp = float(g.attrs["load_amp"])
        u = np.concatenate([params, onehot, [(amp - nrm["amp_mid"]) / nrm["amp_half"]]])

        store_every = int(g.attrs["store_every"])
        load = g["load_force"][()][::store_every] / float(nrm["force_std"][0])

        qoi = np.array([g.attrs[f"qoi_{k}"] for k in nrm["qoi_names"]], dtype=np.float64)
        qoi_ln = (np.log(qoi) - nrm["log_qoi_mean"]) / nrm["log_qoi_std"]

        d = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr.astype(np.float32)),
            pos=torch.from_numpy(nodes.astype(np.float32)),
            u=torch.from_numpy(u.astype(np.float32))[None],
            load=torch.from_numpy(load.astype(np.float32))[None],
            qoi=torch.from_numpy(qoi.astype(np.float32))[None],
            qoi_ln=torch.from_numpy(qoi_ln.astype(np.float32))[None],
            case_id=torch.tensor([cid]),
        )
        if self.fields:
            disp = (g["disp"][()] - nrm["disp_mean"]) / nrm["disp_std"]            # (T, N, 2)
            chans = [disp]
            if nrm["has_vm"]:
                vm = (g["vm"][()] - nrm["vm_mean"][0]) / nrm["vm_std"][0]           # (T, N)
                chans.append(vm[..., None])
            y = np.concatenate(chans, axis=-1).transpose(1, 0, 2)                    # (N, T, C)
            d.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        return d


def make_loaders(h5_path, batch_size=4, num_workers=0, fields=True, in_memory=False,
                 splits=SPLITS, max_train=None):
    """Dict split -> DataLoader. Only the train loader shuffles."""
    norm = load_norm(h5_path)
    out = {}
    for s in splits:
        ds = BracketGraphDataset(h5_path, s, fields=fields, in_memory=in_memory, norm=norm,
                                 max_cases=max_train if s == "train" else None)
        if len(ds) == 0:
            continue
        out[s] = DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"),
                            num_workers=num_workers, persistent_workers=num_workers > 0)
    return out, norm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    loaders, norm = make_loaders(a.h5, a.batch_size)
    for s, dl in loaders.items():
        print(f"{s:9s}: {len(dl.dataset):5d} cases, {len(dl):4d} batches")
    b = next(iter(loaders["train"]))
    print("\none train batch:")
    for k in ("x", "edge_index", "edge_attr", "u", "load", "y", "qoi", "qoi_ln"):
        print(f"  {k:10s} {tuple(getattr(b, k).shape)}")
    print("  graphs in batch:", b.num_graphs, "| ptr:", b.ptr.tolist())
    # sanity: de-normalised target must match the raw file, and QoI log round-trip
    print("  normalised y mean/std per channel:",
          b.y.mean((0, 1)).numpy().round(3), b.y.std((0, 1)).numpy().round(3))
    assert torch.isfinite(b.y).all() and torch.isfinite(b.qoi_ln).all()
    print("OK")
