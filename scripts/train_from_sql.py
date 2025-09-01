#!/usr/bin/env python3
import os, csv, random, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset_sql import Im2LatexSQL, pad_batch
from src.model_im2latex import Im2Latex, cross_entropy_smoothed

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    cfg = {
        "db":   "data/im2latex.db",
        "spm":  "data/spm/latex_sp.model",
        "H":    196,
        "Tmax": 256,
        "model": {"d_model": 512, "nhead": 8, "nlayers": 6},
        "optim": {"lr": 3e-4, "wd": 1e-2, "batch": 32, "epochs": 30, "clip": 1.0},
        # training control
        "sched": {"warmup_frac": 0.05},   # 5% warmup, cosine decay
        "early_stop": {"patience": 5}
    }

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    ds_tr = Im2LatexSQL(cfg["db"], "train", cfg["spm"], cfg["H"], cfg["Tmax"], train=True)
    ds_va = Im2LatexSQL(cfg["db"], "val",   cfg["spm"], cfg["H"], cfg["Tmax"], train=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["optim"]["batch"], shuffle=True,
                       num_workers=4, collate_fn=pad_batch)
    dl_va = DataLoader(ds_va, batch_size=cfg["optim"]["batch"], shuffle=False,
                       num_workers=4, collate_fn=pad_batch)

    # model/optim
    model = Im2Latex(ds_tr.tk.vocab_size, **cfg["model"]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["wd"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # cosine schedule with warmup
    steps_per_epoch = max(1, len(dl_tr))
    total_steps     = max(1, cfg["optim"]["epochs"] * steps_per_epoch)
    warmup_steps    = max(1, int(cfg["sched"]["warmup_frac"] * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * t))  # cosine decay

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # io
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,lr\n")

    best = float("inf")
    bad  = 0
    patience = cfg["early_stop"]["patience"]
    global_step = 0

    for ep in range(cfg["optim"]["epochs"]):
        # ---------------- train ----------------
        model.train()
        tr = 0.0
        pbar = tqdm(dl_tr, desc=f"train {ep}", dynamic_ncols=True)
        for X, Y, *_ in pbar:
            X, Y = X.to(device), Y.to(device)
            y_in, y_out = Y[:, :-1], Y[:, 1:]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(X, y_in)
                loss   = cross_entropy_smoothed(logits, y_out, pad_id=3, eps=0.1)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()         # step LR scheduler *after* optimizer step

            tr += loss.item()
            global_step += 1
            cur_lr = opt.param_groups[0]["lr"]
            pbar.set_postfix(loss=tr / (pbar.n + 1), lr=f"{cur_lr:.2e}")

        # ---------------- val ----------------
        model.eval()
        va = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            for X, Y, *_ in dl_va:
                X, Y = X.to(device), Y.to(device)
                logits = model(X, Y[:, :-1])
                va += cross_entropy_smoothed(logits, Y[:, 1:], pad_id=3, eps=0.1).item()
        va /= max(1, len(dl_va))
        print(f"val={va:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            f.write(f"{ep},{tr/len(dl_tr):.6f},{va:.6f},{opt.param_groups[0]['lr']:.8f}\n")

        # ---------------- checkpoint + early stop ----------------
        if va < best - 1e-4:
            best, bad = va, 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "cfg": cfg,
            }, "checkpoints/best_sql.pt")
            print("saved checkpoints/best_sql.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (no val improvement for {patience} epochs).")
                break

if __name__ == "__main__":
    main()
