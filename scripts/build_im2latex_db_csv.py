#!/usr/bin/env python3
import argparse, sqlite3, csv, sys
from pathlib import Path

DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

DROP TABLE IF EXISTS samples;

CREATE TABLE samples(
  path   TEXT PRIMARY KEY,
  latex  TEXT NOT NULL,
  split  TEXT NOT NULL CHECK(split IN ('train','val','test'))
);

CREATE INDEX IF NOT EXISTS idx_split ON samples(split);
"""

def build_png_index(roots):
    """
    Recursively index all *.png files under the given root directories.
    Returns dict: {basename.png -> absolute_posix_path}
    """
    idx = {}
    dups = 0
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for p in r.rglob("*.png"):
            key = p.name
            if key in idx:
                dups += 1  # keep first occurrence
                continue
            idx[key] = p.as_posix()
    if not idx:
        print("[error] No PNGs found under provided roots.", file=sys.stderr)
    if dups:
        print(f"[warn] {dups} duplicate basenames encountered; kept first occurrence.", file=sys.stderr)
    return idx

def load_split_csv(path: Path, split: str, png_index: dict):
    """
    CSV with header containing columns:
      - 'image'  (e.g., 66667cee5b.png) or 'path'
      - 'formula' (or 'latex')
    Resolves image basenames via png_index and returns rows as (abs_image_path, latex, split).
    """
    rows = []
    missing = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            img = (row.get("image") or row.get("path") or "").strip()
            latex = (row.get("formula") or row.get("latex") or "").strip()
            if not img or not latex:
                continue
            img_path = png_index.get(Path(img).name)
            if not img_path:
                missing += 1
                continue
            rows.append((img_path, latex, split))
    if missing:
        print(f"[warn] {missing} rows in {path.name} skipped (image not found by basename).", file=sys.stderr)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", help="e.g. data/im2latex-100k")
    ap.add_argument("out_db", help="e.g. data/im2latex.db")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    dbpath = Path(args.out_db)
    dbpath.parent.mkdir(parents=True, exist_ok=True)

    # Build a robust index of all PNGs under typical roots (and any subfolders)
    png_index = build_png_index([
        root,
        root / "formula_images_processed",
        root / "formula_images",
        root / "images",
    ])

    con = sqlite3.connect(str(dbpath))
    cur = con.cursor()
    cur.executescript(DDL)

    # Load train/val/test
    all_rows = []
    all_rows += load_split_csv(root / "im2latex_train.csv", "train", png_index)
    all_rows += load_split_csv(root / "im2latex_validate.csv", "val", png_index)
    if (root / "im2latex_test.csv").exists():
        all_rows += load_split_csv(root / "im2latex_test.csv", "test", png_index)

    cur.executemany("INSERT OR IGNORE INTO samples(path, latex, split) VALUES (?,?,?)", all_rows)
    con.commit()

    for sp in ("train", "val", "test"):
        n = cur.execute("SELECT COUNT(*) FROM samples WHERE split=?", (sp,)).fetchone()[0]
        print(f"{sp}: {n}")

    con.close()
    print(f"[ok] built {dbpath}")

if __name__ == "__main__":
    main()
