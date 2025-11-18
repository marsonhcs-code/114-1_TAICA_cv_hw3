#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, csv
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# project modules
from tools.utils import (
    set_seed, is_main, default_device, ensure_dir,
    capture_env, cleanup_ddp
)
from tools.io import write_json, append_csv_row, save_ckpt, load_ckpt, load_cfg
from tools.kfold import make_kfold_splits
from hooks import build_model, build_dataloaders, evaluate

# ======================= 直接在這裡改參數 =======================
CONFIG = dict(
    # 基本
    cfg="configs/exp.yaml",
    out="runs/ddpm_mnist",
    model_name="ddpm",
    epochs=50,
    seed=42,
    note="DDPM for MNIST - baseline training",

    # 訓練
    amp=True,               # 使用混合精度
    compile=False,          # 需 PyTorch 2.x
    accum=1,                # 梯度累積步數
    grad_clip=1.0,          # 梯度裁切

    # DDP
    dist=False,             # 用 torchrun 時設 True
    find_unused_parameters=False,

    # 選擇最佳模型的指標名稱（對生成任務用 val_loss，越小越好）
    best_metric="val_loss",
    metric_mode="min",      # "min" for loss, "max" for mAP/accuracy

    # 每個 epoch 都做 test（可選）
    eval_test_each_epoch=False,

    # 斷點續訓（可選）
    resume=None,            # 例: "runs/ddpm_mnist/weights/best.pt"

    # Early Stopping（可選）
    early_stop_patience=10,

    # Scheduler（可選）
    scheduler="cos",        # None / "cos" / "step"
    step_size=30,
    gamma=0.1,

    # K-fold（可選；>0 啟動）
    kfold=0,
    save_splits=False,
)
# ===============================================================


def train_one_epoch(model, loader, optimizer, device, scaler=None,
                    accum_steps: int = 1, grad_clip: Optional[float] = None) -> Dict[str, float]:
    """
    通用單 epoch 訓練 - 支援 DDPM 和其他任務
    """
    model.train()
    total_loss = 0.0
    n = 0

    for step, batch in enumerate(loader, start=1):
        # 處理不同格式的 batch
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, targets = batch
        else:
            images = batch
            targets = None

        def to_dev(x):
            if torch.is_tensor(x): return x.to(device, non_blocking=True)
            if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)): return type(x)(to_dev(v) for v in x)
            return x

        images = to_dev(images)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast(True):
                # DDPM forward 只需要 images
                loss = model(images)
            scaler.scale(loss / accum_steps).backward()
            if step % accum_steps == 0:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters() if not isinstance(model, DDP) else model.module.parameters(),
                        max_norm=grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
        else:
            loss = model(images)
            (loss / accum_steps).backward()
            if step % accum_steps == 0:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters() if not isinstance(model, DDP) else model.module.parameters(),
                        max_norm=grad_clip
                    )
                optimizer.step()

        total_loss += float(loss.detach().item()); n += 1

    return {"train_loss": total_loss / max(1, n)}


def run_once(args, cfg, device, dist_enabled=False, fold_split=None) -> Dict[str, Any]:
    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir / "weights")
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "preds")
    ensure_dir(out_dir / "splits")

    # ----- 建模 / 優化器 / AMP / Scheduler / Resume -----
    model = build_model(cfg).to(device)
    
    # Move DDPM schedule tensors to device
    if hasattr(model, 'betas'):
        model.betas = model.betas.to(device)
        model.alphas = model.alphas.to(device)
        model.alphas_cumprod = model.alphas_cumprod.to(device)
        model.alphas_cumprod_prev = model.alphas_cumprod_prev.to(device)
        model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
        model.sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod.to(device)
        model.posterior_variance = model.posterior_variance.to(device)
    
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 2e-4), 
        weight_decay=cfg.get("weight_decay", 0.0)
    )

    # scheduler
    sched = None
    if args.scheduler == "cos":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # 資料載入
    dls = build_dataloaders(cfg, fold_split=fold_split, dist=dist_enabled)
    train_loader, val_loader, test_loader = dls.get("train"), dls.get("val"), dls.get("test")

    # DDP 包裝
    if dist_enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.find_unused_parameters)

    # env & hparams
    start_epoch = 0
    if is_main():
        write_json(out_dir / "hparams.json",
                   {"CONFIG": {**vars(args), "start_epoch": start_epoch}, "cfg": cfg, "note": args.note})
        capture_env(out_dir)

    # resume
    if args.resume:
        try:
            ckpt_path = Path(args.resume)
            if ckpt_path.is_dir():
                ckpt_path = ckpt_path / "weights" / "best.pt"
            start_epoch = load_ckpt(ckpt_path, model, optimizer, scaler, map_location="cpu")
            if is_main(): print(f"[resume] loaded from {ckpt_path}, next epoch = {start_epoch+1}")
        except Exception as e:
            if is_main(): print(f"[resume] skip ({e})")

    # ----- 主訓練迴圈 -----
    best_metric_name = args.best_metric
    metric_mode = args.metric_mode  # "min" or "max"
    
    if metric_mode == "min":
        best_metric_val = float('inf')
    else:
        best_metric_val = -float('inf')
    
    header = ["epoch", "train_loss", "val_loss", "lr"]
    patience_left = args.early_stop_patience if args.early_stop_patience else None

    for epoch in range(start_epoch, args.epochs):
        if dist_enabled and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_stats = {}
        if train_loader is not None:
            train_stats = train_one_epoch(
                model, train_loader, optimizer, device, scaler,
                accum_steps=max(1, cfg.get("accum", args.accum)),
                grad_clip=args.grad_clip
            )

        val_metrics, val_per_class, val_per_dets = {}, [], []
        if val_loader is not None:
            val_metrics, val_per_class, val_per_dets = evaluate(
                model.module if isinstance(model, DDP) else model, val_loader, device
            )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_stats.get("train_loss"),
            "val_loss": val_metrics.get("val_loss"),
            "lr": optimizer.param_groups[0]["lr"],
        }
        if is_main():
            append_csv_row(out_dir / "results.csv", header, row)
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {row['train_loss']:.4f}, Val Loss: {row['val_loss']:.4f}, LR: {row['lr']:.6f}")

        # checkpoint：每 epoch + best.pt
        if is_main():
            save_ckpt(out_dir / "weights" / f"epoch_{epoch+1:03d}.pt", model, optimizer, epoch, scaler)
            
            metric_for_best = val_metrics.get(best_metric_name)
            is_better = False
            
            if metric_for_best is not None:
                if metric_mode == "min":
                    if metric_for_best < best_metric_val:
                        best_metric_val = metric_for_best
                        is_better = True
                else:  # max
                    if metric_for_best > best_metric_val:
                        best_metric_val = metric_for_best
                        is_better = True
            
            if is_better:
                save_ckpt(out_dir / "weights" / "best.pt", model, optimizer, epoch, scaler)
                print(f"New best {best_metric_name}: {best_metric_val:.4f}")
                if args.early_stop_patience:
                    patience_left = args.early_stop_patience
            elif args.early_stop_patience:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[early-stop] no improvement on {best_metric_name} for {args.early_stop_patience} epochs.")
                    break

        # scheduler 每 epoch step
        if sched is not None:
            sched.step()

    return {"best_metric": best_metric_val}


def run_kfold(args, cfg):
    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    if "all_ids" in cfg:
        ids = list(cfg["all_ids"])
    else:
        tmp = build_dataloaders(cfg, fold_split=None, dist=False)
        meta = tmp.get("meta") or {}
        ids = meta.get("all_ids") or meta.get("train_ids")
        if not ids:
            raise RuntimeError("K-fold 需要 IDs；請在 cfg['all_ids'] 提供，或讓 build_dataloaders(meta['all_ids']) 回傳。")

    splits = make_kfold_splits(ids, args.kfold, args.seed)
    if args.save_splits:
        for sp in splits:
            write_json(out_root / "splits" / f"fold_{sp['fold']}.json", sp)

    import csv as _csv
    summary_rows = []
    for sp in splits:
        fold_k = sp["fold"]
        fold_out = out_root / f"fold_{fold_k}"
        args_single = SimpleNamespace(**vars(args))
        args_single.out = str(fold_out)

        device, _ = default_device(dist_enabled=args.dist)
        set_seed(args.seed + fold_k)
        result = run_once(args_single, cfg, device, dist_enabled=args.dist, fold_split=sp)
        summary_rows.append({"fold": fold_k, args.best_metric: result["best_metric"]})

    kf_path = out_root / f'kfold_{args.model_name}_val.csv'
    with kf_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["fold", args.best_metric]
        w = _csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summary_rows: w.writerow(r)


def main():
    args = SimpleNamespace(**CONFIG)
    cfg = load_cfg(args.cfg)
    set_seed(args.seed)

    if args.kfold and args.kfold > 0:
        run_kfold(args, cfg)
    else:
        device, _ = default_device(dist_enabled=args.dist)
        run_once(args, cfg, device, dist_enabled=args.dist)

    cleanup_ddp()


if __name__ == "__main__":
    main()