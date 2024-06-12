## Intraoperative MR-Ultrasound Image Matching via Texture-invariant Keypoint Descriptors

This repository contains the implementation of our novel method for matching preoperative Magnetic Resonance (MR) images with intraoperative Ultrasound (US) images. The approach is based on texture-invariant keypoint descriptors, specifically designed to address the challenges of multimodal image matching in medical imaging.

### Overview
Our method introduces a matching-by-synthesis strategy, where intraoperative US images are synthesized from MR images, considering multiple MR modalities and the variability of intraoperative US. The keypoint descriptors are trained in a supervised contrastive manner to be invariant to texture changes and robust against speckle noise.

### Key Features
- **Texture-invariant Keypoint Descriptors**: Designed for robust matching of MR and US images.
- **Matching-by-Synthesis**: Synthesizes US images from MR images to create a diverse training set.
- **Supervised Contrastive Learning**: Trains descriptors to focus on structural similarity and ignore texture variations.
- **Patient-specific Training**: Tailors the model to individual patients for improved performance.
- **Siamese Network Architecture**: Utilizes a Siamese network to learn discriminative descriptors for cross-modality matching.

### Performance
Our experiments on real patient data demonstrate that our approach significantly outperforms state-of-the-art methods, achieving an average matching precision of 80.35%.

### Repository Contents
- **Code**: Implementation of the keypoint descriptor network, image synthesis pipeline, and matching algorithms.
- **Training Data**: Scripts for generating synthetic US images from MR images and creating the training dataset.
- **Evaluation**: Tools for evaluating the performance of the descriptors on real patient data with ground truth.

### Citation
If you use this code in your research, please cite our paper.

### License
This project is licensed under the [MIT License](LICENSE).
