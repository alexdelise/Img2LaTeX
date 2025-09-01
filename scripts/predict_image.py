#!/usr/bin/env python3
import sys, torch, numpy as np
from PIL import Image
from src.dataset_sql import LatexTokenizer, make_transforms
from src.model_im2latex import Im2Latex

def beam_search(model, x, bos_id=1, eos_id=2, beam=5, max_len=256, alpha=0.8):
    device = next(model.parameters()).device
    mem = model.enc(x.to(device))
    beams = [([bos_id], 0.0)]
    for _ in range(max_len):
        cand = []
        for seq, lp in beams:
            if seq[-1] == eos_id:
                cand.append((seq, lp)); continue
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

def render_preview(tex, out_path="outputs/pred_preview.png"):
    import matplotlib.pyplot as plt, os
    os.makedirs("outputs", exist_ok=True)
    plt.figure(); plt.axis("off")
    plt.text(0.05,0.5,f"${tex}$",fontsize=28,va="center")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1); plt.close()
    print("Saved preview:", out_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_image.py path/to/image.png")
        sys.exit(1)

    img_path = sys.argv[1]
    H = 128
    spm = "data/spm/latex_sp.model"
    ckpt = "checkpoints/best_sql.pt"

    tk = LatexTokenizer(spm)
    model = Im2Latex(tk.vocab_size).eval()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])

    tfms = make_transforms(H, train=False)
    arr = np.array(Image.open(img_path).convert("L"))
    x = tfms(image=arr)["image"].unsqueeze(0)
    ids = beam_search(model, x, bos_id=1, eos_id=2, beam=5)
    pred = tk.decode(ids)

    print("Predicted LaTeX:\n", pred)
    try:
        render_preview(pred)
    except Exception as e:
        print("Preview failed:", e)
