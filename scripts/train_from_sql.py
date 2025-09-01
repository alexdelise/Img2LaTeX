#!/usr/bin/env python3
import os, csv, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset_sql import Im2LatexSQL,pad_batch
from src.model_im2latex import Im2Latex,cross_entropy_smoothed

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    cfg = {
        "db":"data/im2latex.db",
        "spm":"data/spm/latex_sp.model",
        "H":128, "Tmax":256,
        "model":{"d_model":512,"nhead":8,"nlayers":6},
        "optim":{"lr":3e-4,"wd":1e-2,"batch":32,"epochs":30,"clip":1.0}
    }
    set_seed(42); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr=Im2LatexSQL(cfg["db"],"train",cfg["spm"],cfg["H"],cfg["Tmax"],train=True)
    ds_va=Im2LatexSQL(cfg["db"],"val",cfg["spm"],cfg["H"],cfg["Tmax"],train=False)
    dl_tr=DataLoader(ds_tr,batch_size=cfg["optim"]["batch"],shuffle=True,num_workers=4,collate_fn=pad_batch)
    dl_va=DataLoader(ds_va,batch_size=cfg["optim"]["batch"],shuffle=False,num_workers=4,collate_fn=pad_batch)

    model=Im2Latex(ds_tr.tk.vocab_size,**cfg["model"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["optim"]["lr"],weight_decay=cfg["optim"]["wd"])
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    os.makedirs("checkpoints",exist_ok=True); os.makedirs("logs",exist_ok=True)
    log_path="logs/train_log.csv"; open(log_path,"w").write("epoch,train_loss,val_loss\n")

    best=float("inf")
    for ep in range(cfg["optim"]["epochs"]):
        model.train(); tr=0.0; pbar=tqdm(dl_tr,desc=f"train {ep}")
        for X,Y,*_ in pbar:
            X,Y=X.to(device),Y.to(device); y_in,y_out=Y[:,:-1],Y[:,1:]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits=model(X,y_in); loss=cross_entropy_smoothed(logits,y_out,pad_id=0,eps=0.1)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg["optim"]["clip"])
            scaler.step(opt); scaler.update()
            tr+=loss.item(); pbar.set_postfix(loss=tr/(pbar.n+1))

        # validation
        model.eval(); va=0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for X,Y,*_ in dl_va:
                X,Y=X.to(device),Y.to(device)
                logits=model(X,Y[:,:-1]); va+=cross_entropy_smoothed(logits,Y[:,1:],pad_id=0,eps=0.1).item()
        va/=len(dl_va)
        print(f"val={va:.4f}")
        with open(log_path,"a") as f: f.write(f"{ep},{tr/len(dl_tr)},{va}\n")
        if va<best:
            best=va; torch.save({"model":model.state_dict(),"cfg":cfg},"checkpoints/best_sql.pt")
            print("saved checkpoints/best_sql.pt")

if __name__=="__main__": main()
