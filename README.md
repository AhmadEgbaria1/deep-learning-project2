# deep-learning-project2
# Deep Learning: Assignment #2

## Submission Details
- **Submission Date:** 24/12/2025, 23:59
- **Submitted by:**
  - Ahmad Egbaria
  - Morad Kutt

## Topics Covered
This assignment explores several key concepts in Deep Learning:
- Regularization
- Batch Normalization
- Convolutional Neural Networks (CNNs)
- Semantic Segmentation

## Project Overview
This repository contains the solution for Deep Learning Assignment #2, focusing on digit classification using CNNs on the SVHN dataset and semantic segmentation with an FC-DenseNet architecture on the CamVid dataset.

### Question 1: Convolutional Digit Classification on SVHN

#### Dataset: Street View House Numbers (SVHN)
- Real-world images of digits (0-9).
- Each image is 32x32 pixels with three color channels (RGB).
- Loaded and preprocessed using `torchvision.datasets.SVHN` with normalization.

#### Model: SVHN_CNN (AlexNet inspired)
- **Architecture:**
  - Initial Convolutional Layer (32 output channels, 3x3 kernel, stride=1, padding=1)
  - ReLU activation
  - MaxPooling Layer (3x3 kernel, stride=2)
  - Convolutional Layer (64 output channels, 3x3 kernel, stride=1, padding=1)
  - ReLU activation
  - Convolutional Layer (128 output channels, 3x3 kernel, stride=1, padding=1)
  - ReLU activation
  - MaxPooling Layer (3x3 kernel, stride=2)
  - Dropout Layer (p=0.5)
  - Fully Connected Layer (output 128)
  - ReLU activation
  - Dropout Layer (p=0.5)
  - Final Fully Connected Layer (output 10 classes)
- **Training:**
  - Loss Function: `nn.CrossEntropyLoss`
  - Optimizer: `torch.optim.Adam`
  - Trained for 6 epochs.

#### Architectural Modifications (SVHN_CNN_v2, SVHN_CNN_v3)
Two modified CNN architectures were implemented, trained, and evaluated to explore the impact of architectural changes on performance.

### Question 2: The One Hundred Layers Tiramisu (Semantic Segmentation)

#### Dataset: CamVid
- Urban driving scenes with pixel-wise annotations.
- Original annotations collapsed into 11 semantic classes + a void class (ignored during training).
- Custom `CamVidDataset` for loading, preprocessing (cropping, flipping), and converting RGB masks to integer class IDs.

#### Model: FCDenseNet (FC-DenseNet-lite)
- **Architecture:** A fully convolutional encoder-decoder network inspired by "The One Hundred Layers Tiramisu" (FC-DenseNet67).
  - Uses `DenseLayer`, `DenseBlock`, `TransitionDown`, and `TransitionUp` modules.
  - Growth rate `k=16` for dense layers.
  - **Encoder:** Initial 3x3 conv -> 4 Dense Blocks (4, 5, 7, 10 layers respectively) each followed by a Transition Down.
  - **Bottleneck:** A Dense Block of 15 layers.
  - **Decoder:** Symmetric Transition Up and Dense Blocks (10, 7, 5, 4 layers respectively) with skip connections from the encoder.
  - **Final Classifier:** 1x1 convolution to 11 class logits.
- **Training:**
  - Loss Function: `nn.CrossEntropyLoss(ignore_index=255)` (pixel-wise cross-entropy)
  - Optimizer: `torch.optim.RMSprop` with weight decay.
  - Scheduler: `CosineAnnealingLR`.
  - Early stopping based on validation mIoU.

#### Evaluation Metrics
- **Pixel-wise Accuracy:** Fraction of correctly classified pixels.
- **Intersection over Union (IoU):** Overlap between prediction and ground truth for each class.
- **Mean IoU (mIoU):** Average IoU across all classes, a key metric for segmentation.

#### Visualization
- Feature maps visualization for the SVHN CNN to understand internal representations.
- Visual inspection of semantic segmentation predictions against ground truth and input images for the CamVid dataset to identify model strengths and weaknesses.

## Setup and Usage
1.  **Clone the repository.**
2.  **Install dependencies:** Ensure you have PyTorch, torchvision, numpy, matplotlib, and PIL installed. If running in Google Colab, most are pre-installed.
3.  **Download CamVid Dataset:** Upload `CamVid.zip` (provided separately) to your Google Drive and follow the instructions in the notebook to mount your drive and extract the dataset.
4.  **Run the notebook:** Execute all cells in sequential order.

## Further Notes
- All figures, printed results, and outputs are retained in the notebook.
- The notebook adheres to strict coding guidelines for clarity and readability.
- Academic integrity is paramount; any plagiarism will result in a zero grade.
