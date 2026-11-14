"""Step 6: train and evaluate the baselines.

Typical sequence
----------------
  # 0) can the model overfit a handful of cases?  (loss should drop by orders of magnitude)
  python train.py --h5 data/v1/dataset.h5 --model mgn --overfit 4 --epochs 300 --name overfit_mgn

  # 1) sanity floor: scalar-QoI regressor (no graphs, fast)
  python train.py --h5 data/v1/dataset.h5 --model mlp --epochs 300 --name mlp_qoi

  # 2) graph model, modal decoder, then the direct-decoder ablation
  python train.py --h5 data/v1/dataset.h5 --model mgn --decoder modal  --epochs 200 --name mgn_modal
  python train.py --h5 data/v1/dataset.h5 --model mgn --decoder direct --epochs 200 --name mgn_direct

Each run writes runs/<name>/ with: config.json, best.pt, history.csv, metrics.json and
preds_<split>.npz (per-case predictions and truths, needed by the conformal stage).

Evaluation splits: val, test_id (same geometry family as training) and test_ood (whole held-out
family). Always compare architectures on identical splits (they are fixed by `plan`).
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from graph_data import make_loaders
from metrics import MetricAccumulator, format_summary
from models import build_model, count_params


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(out, batch, a):
    """Returns (total loss, dict of float components). All terms are in normalised units."""
    l_qoi = F.mse_loss(out["qoi"], batch.qoi_ln)
    if a.model == "mlp":
        return l_qoi, dict(qoi=l_qoi.item())
    y = batch.y
    l_disp = F.mse_loss(out["field"][..., :2], y[..., :2])
    total = a.w_disp * l_disp + a.lambda_qoi * l_qoi
    parts = dict(disp=l_disp.item(), qoi=l_qoi.item())
    if y.shape[-1] > 2:
        l_vm = F.mse_loss(out["field"][..., 2], y[..., 2])
        total = total + a.w_vm * l_vm
        parts["vm"] = l_vm.item()
    return total, parts


@torch.no_grad()
def evaluate(model, loader, device, norm, a):
    model.eval()
    acc, tot, n = MetricAccumulator(norm), 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss, _ = compute_loss(out, batch, a)
        tot += loss.item() * batch.num_graphs
        n += batch.num_graphs
        acc.add_batch(out, batch)
    return tot / max(n, 1), acc


def headline(summary, model_name):
    if model_name == "mlp":
        j = summary["qoi"]["peak_vm_node"]["rel_err"]["median"]
        return f"peak-vM median err {100 * j:5.1f}%"
    return (f"relL2 disp {100 * summary['rel_l2_disp']['median']:5.1f}%  "
            f"vM {100 * summary['rel_l2_vm']['median']:5.1f}%  "
            f"peak-vM err {100 * summary['peak_vm_err']['median']:5.1f}%"
            if "rel_l2_vm" in summary else f"relL2 disp {100 * summary['rel_l2_disp']['median']:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--model", choices=["mlp", "mgn"], default="mgn")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=200, help="warm-up optimiser steps")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=0, help="early-stop patience in epochs (0 = off)")
    # model
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--decoder", choices=["modal", "direct"], default="modal")
    # loss
    ap.add_argument("--w-disp", type=float, default=1.0)
    ap.add_argument("--w-vm", type=float, default=1.0)
    ap.add_argument("--lambda-qoi", type=float, default=0.1)
    # misc
    ap.add_argument("--overfit", type=int, default=0, help="train on the first N train cases only")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--in-memory", action="store_true", help="cache samples in RAM (~2 MB / case)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    set_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    name = a.name or f"{a.model}{'_' + a.decoder if a.model == 'mgn' else ''}_s{a.seed}"
    run_dir = os.path.join(a.out_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    fields = a.model == "mgn"
    max_train = a.overfit or a.max_train
    bs = a.overfit if a.overfit else a.batch_size
    loaders, norm = make_loaders(a.h5, bs, a.workers, fields=fields, in_memory=a.in_memory or bool(a.overfit),
                                 splits=("train",) if a.overfit else ("train", "val", "test_id", "test_ood"),
                                 max_train=max_train)
    print({s: len(dl.dataset) for s, dl in loaders.items()}, "| device:", device)

    # model dimensions come from one batch
    b0 = next(iter(loaders["train"]))
    kw = dict(u_dim=b0.u.shape[1], load_len=b0.load.shape[1], n_qoi=b0.qoi.shape[1])
    if a.model == "mlp":
        kw.update(hidden=a.hidden if a.hidden != 128 else 256, depth=3)
    else:
        kw.update(node_in=b0.x.shape[1], edge_in=b0.edge_attr.shape[1], n_frames=b0.y.shape[1],
                  n_chan=b0.y.shape[2], hidden=a.hidden, n_layers=a.layers,
                  n_modes=a.modes, decoder=a.decoder)
    model = build_model(a.model, **kw).to(device)
    print(f"{a.model}: {count_params(model):,} parameters")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(args=vars(a), model_kwargs=kw, n_params=count_params(model)), f, indent=1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    train_loader = loaders["train"]
    total_steps = a.epochs * len(train_loader)
    warm = max(1, min(a.warmup, total_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    amp = a.amp and device == "cuda"

    eval_loader = (DataLoader(train_loader.dataset, batch_size=bs, shuffle=False)
                   if a.overfit else loaders.get("val"))
    best, best_epoch, bad = float("inf"), 0, 0
    hist_path = os.path.join(run_dir, "history.csv")
    t0 = time.time()
    with open(hist_path, "w", newline="") as hf:
        wr = csv.writer(hf)
        wr.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])
        for ep in range(1, a.epochs + 1):
            model.train()
            tot, n = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    out = model(batch)
                    loss, _ = compute_loss(out, batch, a)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if a.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                sched.step()
                tot += loss.item() * batch.num_graphs
                n += batch.num_graphs
            tr = tot / n

            do_eval = (ep % max(1, a.epochs // 40) == 0) or ep == a.epochs or ep <= 3
            if do_eval and eval_loader is not None:
                vl, acc = evaluate(model, eval_loader, device, norm, a)
                wr.writerow([ep, tr, vl, opt.param_groups[0]["lr"], time.time() - t0]); hf.flush()
                print(f"ep {ep:4d} | train {tr:.4e} | {'train-eval' if a.overfit else 'val'} {vl:.4e} | "
                      f"{headline(acc.summary(), a.model)} | {time.time() - t0:5.0f}s", flush=True)
                if vl < best:
                    best, best_epoch, bad = vl, ep, 0
                    torch.save(dict(model=model.state_dict(), model_kwargs=kw, args=vars(a)),
                               os.path.join(run_dir, "best.pt"))
                else:
                    bad += max(1, a.epochs // 40)
                if a.patience and bad >= a.patience:
                    print("early stopping"); break

    if a.overfit:
        print(f"\noverfit check: final train loss {tr:.3e} (started higher; should be orders of magnitude lower)")
        return

    # ---- final evaluation with the best checkpoint --------------------------------
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    print(f"\nbest val loss {best:.4e} at epoch {best_epoch}")
    results = {}
    for split in ("val", "test_id", "test_ood"):
        if split not in loaders:
            continue
        dl = DataLoader(loaders[split].dataset, batch_size=bs, shuffle=False)
        _, acc = evaluate(model, dl, device, norm, a)
        results[split] = acc.summary()
        print(format_summary(results[split], split))
        np.savez_compressed(os.path.join(run_dir, f"preds_{split}.npz"), **acc.arrays())
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nsaved to", run_dir)


if __name__ == "__main__":
    main()
