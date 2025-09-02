import sqlite3, numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sentencepiece as spm

# ----------------------------
# Data transforms
# ----------------------------
def make_transforms(H: int, train: bool=False):
    aug = []
    if train:
        aug += [
            A.GaussNoise(var_limit=(0.0, 10.0), p=0.15),
            A.RandomBrightnessContrast(0.05, 0.05, p=0.15),
        ]
    return A.Compose([
        A.ToGray(p=1.0),  # images are already gray; p=1 avoids warnings
        A.LongestMaxSize(max_size=H, interpolation=1),
        A.PadIfNeeded(min_height=H, min_width=H, border_mode=0, value=255),
        *aug,
        ToTensorV2()
    ])

# ----------------------------
# Tokenizer wrapper
# ----------------------------
class LatexTokenizer:
    def __init__(self, spm_model, bos_id=1, eos_id=2, pad_id=3):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model)
        self.bos_id, self.eos_id, self.pad_id = bos_id, eos_id, pad_id
        pid = self.sp.pad_id()
        if pid >= 0:
            assert pid == self.pad_id, f"SPM pad_id={pid} but code expects pad_id={self.pad_id}"

    def encode(self, s, add_bos=True, add_eos=True):
        ids = self.sp.encode(s, out_type=int)
        if add_bos: ids = [self.bos_id] + ids
        if add_eos: ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.decode(ids)

    @property
    def vocab_size(self):
        # SPM vocab already includes PAD
        return self.sp.get_piece_size()

# ----------------------------
# Dataset from SQLite
# ----------------------------
class Im2LatexSQL(Dataset):
    def __init__(self, db_path, split, spm_model, H=128, max_len=256, train=False):
        self.tk = LatexTokenizer(spm_model)
        self.tfms = make_transforms(H, train)
        self.max_len = max_len

        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT path, latex FROM samples WHERE split=?", (split,))
        self.rows = cur.fetchall()
        con.close()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, tex = self.rows[i]
        arr = np.array(Image.open(path).convert("L"))
        x = self.tfms(image=arr)["image"]     # [1,H,W]
        # Ensure float in [0,1] to match AMP half/float math
        if x.dtype != torch.float32:
            x = x.float().div(255.0)
        y = torch.tensor(self.tk.encode(tex)[:self.max_len], dtype=torch.long)
        return x, y, path, tex

# ----------------------------
# Batch collator
# ----------------------------
def pad_batch(batch, pad_id=3):
    xs, ys, paths, texs = zip(*batch)
    X = torch.stack(xs)
    L = max(len(y) for y in ys)
    Y = torch.full((len(ys), L), pad_id, dtype=torch.long)
    for i, y in enumerate(ys):
        Y[i, :len(y)] = y
    return X, Y, paths, texs
