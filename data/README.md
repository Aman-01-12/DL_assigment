<!--
File: data/README.md
Purpose: Dataset download and placement instructions.
Author: Aman Verma (23bcs012), Ankit Kashyap (23bcs017), Abhishek Chauhan (23bcs003)
Affiliation: School of Computer science and Engineering, Shri Mata Vaishno Devi University
Contact: info@smvdu.ac.in
-->

# Dataset Instructions

This project uses the Blood Cell Images (BCCD) dataset from Kaggle.

Download and unzip into the project root:
```bash
kaggle datasets download -d paultimothymooney/blood-cells -p dataset --unzip
```

Default config paths point to the Kaggle layout:
```
dataset/dataset2-master/dataset2-master/images/TRAIN
dataset/dataset2-master/dataset2-master/images/TEST
```

If you move the data into a simpler `dataset/train` and `dataset/test` layout, update the paths in `configs/config.yaml`.
