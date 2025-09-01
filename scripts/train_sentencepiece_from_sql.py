#!/usr/bin/env python3
import argparse, sqlite3
from pathlib import Path
import sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_path", help="e.g. data/im2latex.db")
    ap.add_argument("--out_dir", default="data/spm")
    ap.add_argument("--vocab_size", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    corpus = out/"corpus.txt"

    con = sqlite3.connect(args.db_path)
    with open(corpus, "w", encoding="utf-8") as f:
        for (latex,) in con.execute("SELECT latex FROM samples"):
            f.write(latex + "\n")
    con.close()

    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(out/"latex_sp"),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        user_defined_symbols=["{","}","^","_","\\left","\\right","\\frac","\\sqrt"],
        bos_id=1, eos_id=2, pad_id=-1,  # disable auto pad
        unk_id=0
    )
    print("[ok] tokenizer:", out/"latex_sp.model")

if __name__=="__main__":
    main()
