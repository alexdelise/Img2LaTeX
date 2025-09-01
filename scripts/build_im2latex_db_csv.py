#!/usr/bin/env python3
import argparse, sqlite3, csv
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

def load_split_csv(path: Path, split: str, images_root: Path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        for i, line in enumerate(f):
            rows.append((i, line.rstrip("\n")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", help="e.g. data/im2latex-100k")
    ap.add_argument("out_db", help="e.g. data/im2latex.db")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    dbpath = Path(args.out_db)
    dbpath.parent.mkdir(parents=True, exist_ok=True)

    images_root = root / "formula_images_processed"
    if not images_root.exists():
        raise FileNotFoundError("Could not find formula_images_processed/ folder")

    con = sqlite3.connect(str(dbpath))
    cur = con.cursor()
    cur.executescript(DDL)

    # Load train/val/test
    all_rows = []
    all_rows += load_split_csv(root / "im2latex_train.csv", "train", images_root)
    all_rows += load_split_csv(root / "im2latex_validate.csv", "val", images_root)
    if (root / "im2latex_test.csv").exists():
        all_rows += load_split_csv(root / "im2latex_test.csv", "test", images_root)

    cur.executemany("INSERT OR IGNORE INTO samples(path, latex, split) VALUES (?,?,?)", all_rows)
    con.commit()

    for sp in ("train","val","test"):
        n = cur.execute("SELECT COUNT(*) FROM samples WHERE split=?", (sp,)).fetchone()[0]
        print(f"{sp}: {n}")

    con.close()
    print(f"[ok] built {dbpath}")

if __name__ == "__main__":
    main()
