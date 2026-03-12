# Industrial Inspection of 3D Polyurethane Cuts: RANSAC vs. Unsupervised Deep Feature Reconstruction

This repository provides the official PyTorch implementation of **UDFR-Net**
---

## 📑 Table of Contents
- [Abstract](#Abstract)
- [Datasets](#datasets)
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

The dataset was collected in the MOROSAI project using a dual-sensor profilometer with a 405 nm laser line scanner,
producing dense 3D point clouds (.ply) of polyurethane cutting edges [9]. It contains scans from eight cuts, totaling 26
point cloud files, and serves as a benchmark for 3D anomaly detection.

## Code
To train UDFR-Net, use
python UDFR_Net_Train.py \
    --dataset_path ./dataset/polyurethane_cuts \
    --checkpoint_savepath ./checkpoints \
    --class_name "polyurethane_cuts" \
    --epochs_no 50 \
    --batch_size 4
To test UDFR-Net, use
python UDFR_Net_Inference.py \
    --dataset_path ./dataset/polyurethane_cuts \
    --checkpoint_path ./checkpoints/polyurethane_cuts_50ep_4bs.pth \
    --result_path ./results
## Contacts
For questions, please send an email to <radia.daci@isasi.cnr.it>. .

