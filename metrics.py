"""Step 6: evaluation metrics (all computed in PHYSICAL units, after de-normalisation).

Field metrics, per case
    rel_l2_disp     ||u_pred - u_true|| / ||u_true||   over all nodes, stored frames, ux and uy
    rel_l2_vm       same for the nodal von Mises history
    peak_vm_err     |max vM_pred - max vM_true| / max vM_true, both over the STORED frames
    max_disp_err    same for the peak displacement magnitude

QoI metrics (scalar quantities, from the QoI head or the QoI-MLP)
    per QoI: median / mean / p95 relative error and R^2 (raw and log space)

Decimation caveat: the dataset's peak_vm_node QoI is taken over every solver step, while a
predicted field only exists at the stored frames. `peak_vm_gt_decimation_gap` reports how much the
GROUND TRUTH peak shrinks when restricted to stored frames, which is a floor on any field-based
peak error.

`MetricAccumulator.arrays()` returns per-case predictions and truths; save them, because the
conformal-prediction stage needs exactly these residuals.
"""
import numpy as np
import torch


def _summ(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(np.max(x)))


def denorm_field(y, norm):
    """Normalised (..., C) -> physical disp (..., 2) and von Mises (...) tensors."""
    dm = torch.as_tensor(norm["disp_mean"], dtype=y.dtype, device=y.device)
    ds = torch.as_tensor(norm["disp_std"], dtype=y.dtype, device=y.device)
    disp = y[..., :2] * ds + dm
    vm = None
    if y.shape[-1] > 2 and norm["has_vm"]:
        vm = y[..., 2] * float(norm["vm_std"][0]) + float(norm["vm_mean"][0])
    return disp, vm


def denorm_qoi(q_ln, norm):
    """Standardised log-QoI -> raw QoI (numpy)."""
    q = np.asarray(q_ln, dtype=np.float64) * norm["log_qoi_std"] + norm["log_qoi_mean"]
    return np.exp(q)


class MetricAccumulator:
    def __init__(self, norm):
        self.norm = norm
        self.rows = []                    # one dict per case

    @torch.no_grad()
    def add_batch(self, out, batch):
        """out: model output dict; batch: PyG batch (needs batch.qoi, optionally batch.y)."""
        nrm = self.norm
        qn = nrm["qoi_names"]
        q_true = batch.qoi.cpu().numpy().astype(np.float64)
        q_pred = denorm_qoi(out["qoi"].detach().cpu().numpy(), nrm) if "qoi" in out else None
        ptr = batch.ptr.tolist()
        ids = batch.case_id.cpu().numpy()

        for i in range(batch.num_graphs):
            row = dict(case_id=int(ids[i]), qoi_true=q_true[i])
            if q_pred is not None:
                row["qoi_pred"] = q_pred[i]
            if "field" in out and hasattr(batch, "y"):
                a, b = ptr[i], ptr[i + 1]
                pd, pv = denorm_field(out["field"][a:b].detach().float(), nrm)
                td, tv = denorm_field(batch.y[a:b].float(), nrm)
                row["rel_l2_disp"] = float(torch.linalg.norm(pd - td) / torch.linalg.norm(td).clamp(min=1e-30))
                pm, tm = pd.norm(dim=-1).max().item(), td.norm(dim=-1).max().item()
                row["max_disp_pred_field"], row["max_disp_true_stored"] = pm, tm
                row["max_disp_err"] = abs(pm - tm) / max(tm, 1e-30)
                if pv is not None:
                    row["rel_l2_vm"] = float(torch.linalg.norm(pv - tv) / torch.linalg.norm(tv).clamp(min=1e-30))
                    pp, tp = pv.max().item(), tv.max().item()
                    full = float(q_true[i][qn.index("peak_vm_node")])
                    row["peak_vm_pred_field"], row["peak_vm_true_stored"] = pp, tp
                    row["peak_vm_err"] = abs(pp - tp) / max(tp, 1e-30)
                    row["peak_vm_gt_decimation_gap"] = (full - tp) / full
            self.rows.append(row)

    # -- aggregation -----------------------------------------------------------
    def summary(self):
        s, qn = {}, self.norm["qoi_names"]
        for k in ("rel_l2_disp", "rel_l2_vm", "peak_vm_err", "max_disp_err", "peak_vm_gt_decimation_gap"):
            vals = [r[k] for r in self.rows if k in r]
            if vals:
                s[k] = _summ(vals)
        if self.rows and "qoi_pred" in self.rows[0]:
            qt = np.stack([r["qoi_true"] for r in self.rows])
            qp = np.stack([r["qoi_pred"] for r in self.rows])
            rel = np.abs(qp - qt) / np.abs(qt)
            r2 = lambda t, p: float(1 - ((t - p) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-300))
            s["qoi"] = {name: dict(rel_err=_summ(rel[:, j]), r2=r2(qt[:, j], qp[:, j]),
                                   r2_log=r2(np.log(qt[:, j]), np.log(qp[:, j])))
                        for j, name in enumerate(qn)}
        s["n_cases"] = len(self.rows)
        return s

    def arrays(self):
        """Per-case arrays for saving (used later for conformal calibration)."""
        out = dict(case_id=np.array([r["case_id"] for r in self.rows]),
                   qoi_true=np.stack([r["qoi_true"] for r in self.rows]),
                   qoi_names=np.array(self.norm["qoi_names"]))
        for k in ("qoi_pred", "peak_vm_pred_field", "peak_vm_true_stored", "max_disp_pred_field",
                  "max_disp_true_stored", "rel_l2_disp", "rel_l2_vm"):
            if k in self.rows[0]:
                out[k] = np.stack([r[k] for r in self.rows])
        return out


def format_summary(s, title=""):
    lines = [f"--- {title} ({s['n_cases']} cases)"]
    for k in ("rel_l2_disp", "rel_l2_vm", "peak_vm_err", "max_disp_err", "peak_vm_gt_decimation_gap"):
        if k in s:
            v = s[k]
            lines.append(f"  {k:28s} median {100 * v['median']:6.2f}%  mean {100 * v['mean']:6.2f}%  p95 {100 * v['p95']:6.2f}%")
    if "qoi" in s:
        lines.append("  QoI (head / MLP):          median rel.err    R2     R2(log)")
        for k, v in s["qoi"].items():
            lines.append(f"    {k:22s} {100 * v['rel_err']['median']:8.2f}%   {v['r2']:7.3f}  {v['r2_log']:7.3f}")
    return "\n".join(lines)
