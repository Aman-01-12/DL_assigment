<!--
File: README.md
Purpose: Project overview, setup, and usage instructions.
Author: Aman Verma (23bcs012), Ankit Kashyap (23bcs017), Abhishek Chauhan (23bcs003)
Affiliation: School of Computer science and Engineering, Shri Mata Vaishno Devi University
Contact: info@smvdu.ac.in
-->

# VGGNet19 Blood Cell Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

## Overview
This project trains and evaluates a VGG-19 model on the Blood Cell Images (BCCD) dataset to classify four blood cell types.

## Dataset Download (Kaggle)
1. Install the Kaggle API and configure credentials in `~/.kaggle/kaggle.json`:

```bash
pip install kaggle
```

2. Download and unzip the dataset into `dataset/`:

```bash
kaggle datasets download -d paultimothymooney/blood-cells -p dataset --unzip
```

Kaggle unzips into the following layout (used by the default config):
```
dataset/
└── dataset2-master/
    └── dataset2-master/
        └── images/
            ├── TRAIN/
            │   ├── EOSINOPHIL/
            │   ├── LYMPHOCYTE/
            │   ├── MONOCYTE/
            │   └── NEUTROPHIL/
            └── TEST/
                ├── EOSINOPHIL/
                ├── LYMPHOCYTE/
                ├── MONOCYTE/
                └── NEUTROPHIL/
```

If you move the data into a simpler `dataset/train` and `dataset/test` layout, update the paths in `configs/config.yaml`.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python src/train.py --config configs/config.yaml
```

## Evaluate
```bash
python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

## Generate Report Figures
```bash
python src/figures.py --config configs/config.yaml
```

## Results (2-epoch quick run)
| Metric | Value |
| --- | --- |
| Accuracy | 27.91% |
| Macro F1 | 0.1858 |
| Macro AUC-ROC | 0.5875 |

Notes: Metrics above are from a 2-epoch quick run on a small subset for feasibility on a MacBook Air.

## License
© Aman Verma, Ankit Kashyap, Abhishek Chauhan, 2026. All Rights Reserved.
This project is submitted as academic coursework.
No reuse or redistribution permitted without permission.

