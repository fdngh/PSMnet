# PSMnet
Codes for the following paper:
Beyond Singular Prototype: A Prototype Splitting Strategy for Few-Shot Medical Image Segmentation
# Abstract
In the realm of medical image semantic segmentation, few-shot learning, characterized by its efficient
data utilization and flexible generalization capabilities, has been garnering increasing attention. The
mainstream methods currently employ prototype-based approaches, which extract semantic knowledge from the annotated support images to guide the segmentation of the query image via masked
global average pooling. However, such masked global average pooling leads to severe information
loss, which is more problematic for medical images with large numbers of highly heterogeneous
background categories. In this work, we propose a Prototype Splitting Module (PSM) to effectively
address the issue of semantic information loss in few-shot medical image segmentation. Specifically,
PSM iteratively splits the support image masks into set of sub-masks containing segmented regions
and unsegmented regions in a self-guided manner. This maximally retains the information within
the original semantic classes and better extracts the representations of those classes. Additionally,
we devise a Multi-level Cross Attention Module (MCAM) that transfers the foreground information
from the support images to the query images across different levels to facilitate final segmentation
prediction. We validate our method on multiple modal and multi-semantic medical image datasets.
Results demonstrate that our approach achieves superior performance over existing state-of-the-art
methods.
# Dependencies<br>
json5                          0.9.6<br>
jupyter-server                 1.11.0
jupyterlab                     3.1.12
matplotlib                     3.4.3
matplotlib-inline              0.1.3
numpy                          1.21.2
opencv-python                  4.9.0.80
Pillow                         8.3.2
sacred                         0.8.5
scipy                          1.10.1
SimpleITK                      2.3.1
tensorboard                    2.6.0
torch                          1.8.1+cu111
torchvision                    0.9.1+cu111
tornado                        6.1
tqdm                           4.61.2
# Datasets and pre-processing
Download:
# Training
# Testing
# Acknowledgement
