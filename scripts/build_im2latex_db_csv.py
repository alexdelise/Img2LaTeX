#!/usr/bin/env python3
import argparse, csv, sqlite3
from pathlib import Path

DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

DROP TABLE IF EXISTS formulas;
DROP TABLE IF EXISTS images;
DROP TABLE IF EXISTS splits;

CREATE TABLE formulas(
  id    INTEGER PRIMARY KEY,
  latex TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE images(
  path       TEXT PRIMARY KEY,
  formula_id INTEGER NOT NULL REFERENCES formulas(id)
);

CREATE TABLE splits(
  path  TEXT PRIMARY KEY REFERENCES images(path) ON DELETE CASCADE,
  split TEXT NOT NULL CHECK(split IN ('train','val','test'))
);

CREATE VIEW IF NOT EXISTS view_samples AS
SELECT s.split, i.path, f.latex
FROM images i
JOIN splits s USING(path)
JOIN formulas f ON i.formula_id = f.id;

CREATE INDEX IF NOT EXISTS idx_images_formula ON images(formula_id);
CREATE INDEX IF NOT EXISTS idx_splits_split   ON splits(split);
"""

def sniff_csv(path: Path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        sample = f.read(4096); f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
        has_header = csv.Sniffer().has_header(sample)
    return dialect, has_header

def load_formulas_txt(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        for i, line in enumerate(f):
            rows.append((i, line.rstrip("\n")))
    return rows

def load_split_csv(path: Path):
    pairs = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if not parts or len(parts) < 2:
                continue
            # Skip header
            if i == 0 and (not parts[0].isdigit() and not parts[1].isdigit()):
                continue
            # Case 1: img first, index second
            if parts[1].isdigit():
                img, idx = parts[0].strip(), int(parts[1])
            # Case 2: index first, img second
            elif parts[0].isdigit():
                idx, img = int(parts[0]), parts[1].strip()
            else:
                continue  # malformed row
            pairs.append((img, idx))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", help="data/im2latex-100k")
    ap.add_argument("out_db",       help="data/im2latex.db")
    args = ap.parse_args()

    root   = Path(args.dataset_root)
    dbpath = Path(args.out_db)
    dbpath.parent.mkdir(parents=True, exist_ok=True)

    images_root = root / "formula_images_processed"
    if not images_root.exists():
        # fallback to common names
        for nm in ["images","imgs","img","formula_images"]:
            if (root/nm).exists():
                images_root = root/nm; break

    formulas_csv = root / "im2latex_formulas.norm.csv"
    train_csv    = root / "im2latex_train.csv"
    val_csv      = root / "im2latex_validate.csv"
    test_csv     = root / "im2latex_test.csv"

    assert formulas_csv.exists(), "im2latex_formulas.norm.csv not found"
    assert train_csv.exists(),    "im2latex_train.csv not found"
    assert val_csv.exists(),      "im2latex_validate.csv not found"

    formulas = load_formulas_txt(formulas_csv)
    tr_pairs = load_split_csv(root / "im2latex_train.csv")
    va_pairs = load_split_csv(root / "im2latex_validate.csv")
    te_pairs = load_split_csv(root / "im2latex_test.csv")

    con = sqlite3.connect(str(dbpath)); cur = con.cursor()
    cur.executescript(DDL)

    cur.executemany("INSERT INTO formulas(id, latex) VALUES (?,?)", formulas)

    def resolve(img_rel: str) -> str:
        p = images_root/Path(img_rel).name
        return str((p if p.exists() else (root/img_rel)).resolve())

    def ingest(pairs, split):
        rows_img, rows_split = [], []
        for rel, idx in pairs:
            path = resolve(rel)
            rows_img.append((path, idx))
            rows_split.append((path, split))
        cur.executemany("INSERT OR IGNORE INTO images(path, formula_id) VALUES (?,?)", rows_img)
        cur.executemany("INSERT OR REPLACE INTO splits(path, split) VALUES (?,?)", rows_split)

    ingest(tr_pairs, "train")
    ingest(va_pairs, "val")
    if te_pairs: ingest(te_pairs, "test")

    con.commit()
    for sp in ("train","val","test"):
        n = cur.execute("SELECT COUNT(*) FROM splits WHERE split=?", (sp,)).fetchone()[0]
        print(f"{sp}: {n}")
    con.close()
    print(f"[ok] built {dbpath}")

if __name__ == "__main__":
    main()
