# H-Sets: Uncovering Feature Interactions in Image Classification using Hessian

Repository for "H-Sets: Uncovering Feature Interactions in Image Classification using Hessian," presented at the ATTRIB Workshop at Conference on Neural Information Processing Systems (NeurIPS) 2024. 

## Introduction
Feature attribution methods have emerged as a key tool for model explainability, assigning importance scores to individual input features in order to explain a model's decision for a specific input. However, these methods typically assume additive effects and focus solely on the marginal contribution of individual features, overlooking a critical aspect of model behavior: _feature interaction_.

Feature interactions occur when a set of features jointly influences a model’s prediction in a way that cannot be captured by their individual effects. These interactions are especially prevalent in image classification, where pixel interdependencies often carry significant semantic information. Yet, most existing methods either fail to capture such interactions or rely on coarse approximations using superpixels or regularization masks. Moreover, they do not satisfy the key axioms of feature attribution, such as completeness and implementation invariance.

In this work, we address this gap by introducing _H-Sets (Hessian Sets)_, a principled framework for discovering and attributing higher-order feature interactions in image classifiers. Our method leverages the Hessian matrix to detect pairwise feature dependencies and recursively merges them into semantically coherent higher-order interaction sets. To ensure interpretability, we anchor this process in perceptually meaningful regions using masks derived from the Segment Anything Model (SAM).  To attribute importance scores to these sets, we extend _Integrated Directional Gradients (IDG)_, originally developed for text classifiers, to the image domain, formulating a set-level attribution approach grounded in cooperative game theory and satisfying a comprehensive set of attribution axioms.

## Requirements
The current implementation is in Pytorch and Python 3. Below is the list of libraries that we used:
 - quantus (for sparsity and ROAD metrics)
 - opencv (for image processing)
 - captum (for saliency maps)
 - [SAM](https://github.com/facebookresearch/segment-anything.git) (for image segmentation)

You can install these libraries using 
```pip install -r requirements.txt```

## Models and Baselines Used
We use MobileNetV3 (Large), DenseNet121, ResNet101, and VGG16 on ImageNet and Caltech-UCSD Birds-200-2011. See models folder for the training script. On each model, we run 4 baselines: Integrated Gradients, Archipelago, Context-Aware First-Order (CAFO) explanations, and Context-Aware Second-Order (CASO) explanations. We provide the code for each baseline in the baselines folder. 

## Demo
We provide a Jupyter Notebook to demonstrate our algorithm H-Sets on Caltech-UCSD Birds-200-2011. You will need to download the CUB dataset, which you can get [here](https://data.caltech.edu/records/65de6-vp158).
