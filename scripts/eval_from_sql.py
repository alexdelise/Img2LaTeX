#!/usr/bin/env python3
import csv, torch
from src.dataset_sql import Im2LatexSQL
from src.model_im2latex import Im2Latex

def beam_search(model, x, bos_id=1, eos_id=2, beam=5, max_len=256, alpha=0.8):
    device = next(model.parameters()).device
    mem = model.enc(x.to(device))
    beams = [([bos_id], 0.0)]
    for _ in range(max_len):
        cand = []
        for seq, lp in beams:
            if seq[-1] == eos_id:
                cand.append((seq, lp))
                continue
            y = torch.tensor([seq], device=device)
            logits = model.dec(y, mem)[:, -1]
            logp = torch.log_softmax(logits, -1).squeeze(0)
            vals, idx = torch.topk(logp, beam)
            for v, i in zip(vals.tolist(), idx.tolist()):
                cand.append((seq + [i], lp + v))
        cand.sort(key=lambda p: p[1] / (len(p[0]) ** alpha), reverse=True)
        beams = cand[:beam]
        if all(s[-1] == eos_id for s, _ in beams):
            break
    return max(beams, key=lambda p: p[1] / (len(p[0]) ** alpha))[0]

def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]

def canonicalize(s):
    return " ".join(
        s.replace("\\dfrac", "\\frac")
         .replace("\\tfrac", "\\frac")
         .replace("\\,", " ")
         .split()
    )

if __name__ == "__main__":
    db = "data/im2latex.db"
    spm = "data/spm/latex_sp.model"
    ckpt = "checkpoints/best_sql.pt"
    split = "val"
    N = 10  # evaluate first 10 samples for speed

    ds = Im2LatexSQL(db, split, spm, 128, 256, train=False)
    tk = ds.tk
    model = Im2Latex(tk.vocab_size).eval()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])

    from tqdm.auto import tqdm

    total, exact, dsum = 0, 0, 0
    rows = []
    for i in tqdm(range(min(N, len(ds))), desc=f"Evaluating {split}", unit="sample"):
        x, y_true, path, gt = ds[i]
        ids = beam_search(model, x.unsqueeze(0), bos_id=1, eos_id=2)
        pred = tk.decode(ids)
        if canonicalize(pred) == canonicalize(gt):
            exact += 1
        dsum += levenshtein(
            tk.encode(pred, True, True),
            tk.encode(gt, True, True)
        )
        total += 1
        if i < 50:
            rows.append([path, gt, pred])

    print(f"Exact-match: {exact/total:.4f} (N={total})")
    print(f"Avg token edit distance: {dsum/total:.2f}")

    with open("logs/samples_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path","gt","pred"]); w.writerows(rows)
    print("Wrote logs/samples_val.csv")
