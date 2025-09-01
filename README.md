# Img2LaTeX

## Overview

Img2LaTeX is a deep learning system that converts mathematical expressions from images into LaTeX markup. It is built using **PyTorch**, **Albumentations**, and **SentencePiece**, and trained on the **Im2Latex-100k** dataset. The pipeline involves preprocessing images, tokenizing LaTeX code, training a CNN–Transformer architecture, and decoding predictions via beam search.

This repository provides:
- Scripts to download and preprocess datasets into a SQLite database.
- SentencePiece tokenization for LaTeX formulae.
- A CNN encoder and Transformer decoder for sequence modeling.
- Training, evaluation, and prediction pipelines.

## Mathematical Model Description

### Encoder: CNN Feature Extractor

The encoder processes grayscale formula images into a sequence of embeddings.

1. **Convolutions**:  
   $
      f = \text{ReLU}(\text{Conv2D}(x))
   $ 
   stacked layers increase channels from 1 → 64 → 128 → 256 → $d_{model}$.

2. **Positional Encoding**:  
   A 2D sinusoidal encoding injects spatial information:  
   $
      PE_{x,y}(i) =
      \begin{cases}
         \sin(x / 10000^{2i/d}), & i \text{ even} \\
         \cos(x / 10000^{2i/d}), & i \text{ odd}
      \end{cases}
   $
   Similarly for $y$, then concatenated.

3. **Flattening**: The feature map is reshaped into a sequence suitable for the decoder.

### Decoder: Transformer

The decoder is a Transformer with $N$ stacked layers:
- Multi-head self-attention
- Cross-attention with encoder memory
- Position-wise feed-forward layers

The input is token embeddings plus positional encodings:  
$
   h = \text{TransformerDecoder}(y_{in}, \text{mem})
$
The final logits are projected into the vocabulary space.

### Loss Function: Label Smoothed Cross-Entropy

To prevent overfitting and encourage generalization, we use label smoothing:  
$
   \mathcal{L} = (1-\epsilon) \cdot \text{CE}(y, \hat{y}) + \epsilon \cdot U
$
where $U$ is the uniform distribution loss and $\epsilon = 0.1$.

### Decoding: Beam Search

Instead of greedy decoding, predictions are generated using **beam search**. At each step, we keep the top-$k$ candidate sequences:  
$
`   \text{score}(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log p(y_t | y_{<t}, x)
$ 
with length normalization parameter $\alpha = 0.8$.

This balances fluency and sequence length.

## Installation & Requirements

### Dependencies
- Python 3.9+
- PyTorch
- Albumentations
- SentencePiece
- tqdm
- PIL
- matplotlib
- kagglehub

Install all packages with:
```bash
pip install torch torchvision albumentations sentencepiece tqdm pillow matplotlib kagglehub
```

## Usage

### Step 1: Download the Dataset
```bash
python scripts/download_datasets.py
```

### Step 2: Build SQLite Database
```bash
python scripts/build_im2latex_db_csv.py data/im2latex-100k data/im2latex.db
```

### Step 3: Train SentencePiece Tokenizer
```bash
python scripts/train_sentencepiece_from_sql.py data/im2latex.db --out_dir data/spm --vocab_size 2000
```

### Step 4: Train the Model
```bash
python scripts/train_from_sql.py
```

Checkpoints are saved under `checkpoints/`.

### Step 5: Evaluate the Model
```bash
python scripts/eval_from_sql.py
```

### Step 6: Predict from Image
```bash
python scripts/predict_image.py test_images/ap1lessbp1.png
```

This outputs predicted LaTeX and saves a rendered preview.

## File Structure

```
├── .gitignore
├── README.md
├── scripts/                # Training, evaluation, preprocessing scripts
│   ├── build_im2latex_db_csv.py
│   ├── download_datasets.py
│   ├── eval_from_sql.py
│   ├── predict_image.py
│   ├── train_from_sql.py
│   └── train_sentencepiece_from_sql.py
├── src/                    # Core dataset & model code
│   ├── dataset_sql.py
│   └── model_im2latex.py
├── test_images/            # Sample test images
│   └── ap1lessbp1.png
├── checkpoints/            # Saved models
├── logs/                   # Training & evaluation logs

```

## Citation

If you use this repo, please cite the Im2Latex-100k dataset:

```
@inproceedings{deng2017latex,
   author = {Deng, Yuntian and Kanervisto, Anssi and Ling, Jeffrey and Rush, Alexander M.},
   title = {Image-to-markup generation with coarse-to-fine attention},
   year = {2017},
   publisher = {JMLR.org},
   booktitle = {Proceedings of the 34th International Conference on Machine Learning - Volume 70},
   pages = {980–989},
   numpages = {10},
   location = {Sydney, NSW, Australia},
   series = {ICML'17}
}
```

## License

This repository is released under the MIT License.
