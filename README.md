# Industrial Inspection of 3D Polyurethane Cuts: RANSAC vs. Unsupervised Deep Feature Reconstruction

This repository provides the official PyTorch implementation of **UDFR-Net**
---

## 📑 Table of Contents
- [Abstract](#Abstract)
- [Datasets](#datasets)
- [Checkpoints](#checkpoints)
- [Code](#code)
- [Contacts](#contacts)

## Abstract
Ensuring high-quality industrial products requires reliable surface defect detection for effective quality control. Classical geometric inspection methods can identify deviations on
structured surfaces but often fail to detect subtle, irregular, or sparse anomalies in noisy, complex 3D point clouds. In this work, we evaluate both classical geometric inspec-
tion and deep learning-based approaches on a real industrial dataset of polyurethane cuts. A RANSAC-based geometric method serves as a baseline, while we propose UDFR-
Net, an unsupervised deep feature reconstruction network that leverages features from a frozen autoencoder to model
normal geometric patterns and detect anomalies via deviations in reconstruction. Experimental results demonstrate the limitations of classical methods and show that UDFR- Net achieves
high defect detection and localization perfomance, with accuracy 0.970, I-AUROC 0.986, and P-AUROC 0.910, while maintaining efficiency and robustness suitable
for real-world deployment. Our source code is available at
## Datasets

We evaluate CMDR-IAD on the **[MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)** dataset, which provides paired RGB images and 3D point clouds for industrial anomaly detection.

The raw dataset requires preprocessing to obtain aligned RGB images and organized point clouds. The necessary preprocessing scripts are provided in the `processing` directory.



## 📦 checkpoints

We release the pretrained CMDR-IAD checkpoints used to obtain the results reported in the paper.
The weights are provided per object category and can be directly used for inference.

- The pretrained CMDR-IAD checkpoints used to obtain the results reported in the paper will be released after the paper is accepted.
- Create a folder named `checkpoints` in the project directory;
- Copy the downloaded weights into the `checkpoints`.

## Code
CMDR-IAD provides scripts for **training** and **inference** of cross-modal mapping and dual-branch reconstruction networks for industrial anomaly detection.

To train CMDR-IAD, use the train.py script.
To Test CMDR-IAD. use the inference.py script.


Train and test options

`--dataset_path` : Path to the root directory of the MVTec 3D-AD dataset.

`--checkpoint_savepath` : Directory where trained checkpoints are saved (training) or read from (inference) (default: `./checkpoints/CMDR_IAD_checkpoints`).

`--class_name` : Object category to train on or to test on .

`--epochs_no` : Number of epochs.

`--batch_size` : Batch size.

Each object category is trained independently, and the resulting checkpoints are stored per class for inference.

## Contacts
For questions, please send an email to <radia.daci@isasi.cnr.it>. .

## 📂 Repository Structure

```text
CMDR-IAD/
├── networks/
│   ├── features.py
│   ├── Map.py
│   ├── Dec2d.py
│   ├── Dec3d.py
│   ├── dataset.py
│   └── full_models.py
├── processing/
│   ├── aggregate_results.py
│   ├── preprocess_mvtec.py
├── utils/
│   ├── mvtec3d_utils.py
│   ├── pointnet2_utils.py
│   ├── metrics_utils.py
│   └── general_utils.py
│
├── train.py
├── inference.py
├── README.md
├── requirements.txt
