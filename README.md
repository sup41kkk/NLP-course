# Multimodal Movie Genre Classification

This project implements a deep learning model for classifying movie genres based on both their posters (images) and plot summaries (text). The model utilizes pre-trained ResNet50 for image feature extraction and BERT for text feature extraction, combining these modalities for a final multi-label genre prediction.

**Project Code Repository:** [https://github.com/sup41kkk/NLP-course](https://github.com/sup41kkk/NLP-course)

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Experiments](#experiments)
  - [Metrics](#metrics)
  - [Setup](#setup)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Related Work](#related-work)
- [Conclusion](#conclusion)

## Introduction
Movie genre classification is a valuable task for recommendation systems and content organization. This project explores a multimodal approach, leveraging the rich information present in both visual movie posters and textual plot descriptions. The model aims to predict multiple genres for a given movie by effectively fusing these two modalities.

The core of the project involves:
-   Using pre-trained ResNet50 and BERT as feature extractors.
-   Employing SwiGLU activation functions in projection and classifier heads for enhanced feature transformation.
-   Implementing a partial fine-tuning strategy for the pre-trained backbones.
-   Utilizing advanced training techniques:
    -   **Focal Loss**: To address class imbalance inherent in movie genre datasets.
    -   **Label Smoothing**: To prevent model overconfidence and improve generalization.
    -   **Learning Rate Warmup**: For more stable training convergence.
    -   **Early Stopping**: To prevent overfitting and save the best performing model.

## Model Architecture
The `MultimodalClassifier` consists of:
1.  **Image Backbone (ResNet50)**: Extracts 2048-dimensional features from movie posters. The last block (`layer4`) is fine-tuned.
2.  **Text Backbone (BERT - `bert-base-uncased`)**: Extracts 768-dimensional features (`pooler_output`) from plot summaries. The last 2 encoder layers, pooler, and embeddings are fine-tuned.
3.  **Projection Layers (SwiGLU)**: Both image and text features are projected to a 512-dimensional common space using SwiGLU activation.
    ```
    SwiGLU(x, W1, W3, W2) = (SiLU(xW1) ⊙ (xW3))W2
    ```
4.  **Feature Fusion & Self-Attention**: Projected features are concatenated and then processed by a multi-head self-attention layer (8 heads) with a residual connection and LayerNorm.
    ```
    AttnOutput = LayerNorm(SelfAttention(FusedFeatures) + FusedFeatures)
    ```
5.  **Classifier Head (SwiGLU)**: The attention output is passed through LayerNorm, another SwiGLU FFN layer, dropout (0.3), and finally a linear layer for multi-label genre logits (26 genres).

## Dataset
-   **Name**: MM-IMDb (Multimodal IMDb)
-   **Source**: Accessed via Hugging Face Datasets (`sxj1215/mmimdb`).
-   **Content**: Movie posters (images) and plot summaries/metadata (text).
-   **Task**: Multi-label movie genre classification.
-   **Details**:
    -   The original 'train' split of 15,552 samples was used.
    -   Split into 13,219 training and 2,333 validation samples (0.15 validation ratio, seed 42).
    -   26 unique genres were dynamically extracted for classification.
-   **Preprocessing**:
    -   **Text**: Plot extraction via regex, BERT tokenization (max length 128).
    -   **Image**: Resized to 224x224. Augmentations for training (RandomResizedCrop, HorizontalFlip, ColorJitter, Rotation, Affine). Normalization using ImageNet stats.
    -   **Labels**: MultiLabelBinarizer for genre encoding.

## Experiments

### Metrics
-   **Macro F1-score**: Primary metric for evaluation, early stopping, and LR scheduling.
-   **Micro F1-score**: Global F1-score.

### Setup
-   **Hardware**: CUDA-enabled GPU.
-   **Frameworks**: PyTorch, Hugging Face Transformers, Datasets, Scikit-learn.
-   **Key Hyperparameters**:
    -   Batch Size: 16
    -   Epochs: 20
    -   Optimizer: AdamW (LR: $1 \times 10^{-4}$, Weight Decay: $1 \times 10^{-5}$)
    -   Loss: Focal Loss ($\alpha = 0.25, \gamma = 2.0$)
    -   Label Smoothing: $\epsilon = 0.1$
    -   Warmup: 3 epochs (from $1 \times 10^{-6}$)
    -   LR Scheduler: ReduceLROnPlateau (factor 0.2, patience 2)
    -   Early Stopping: Patience 5, Min Delta 0.001

## Results
The model was trained for 20 epochs.
-   **Best Validation Macro F1-score**: 0.5230 (achieved at epoch 17)
-   **Validation Micro F1-score (at best Macro F1)**: 0.6390

Training showed consistent improvement in early epochs with the learning rate warmup. The validation loss started to increase in later epochs, suggesting some overfitting despite early stopping mechanisms (which did not trigger within the 20 epochs but would have if training continued without improvement).

A comparison with other models on the MM-IMDb dataset (as reported by Seo et al., 2022) is provided in the full project report (`Supkhankulov_report.pdf`). Our model's performance is competitive, especially considering it does not use the graph-based structure of SOTA models like MM-GATBT.

| Model                     | Micro F1 | Macro F1 |
| :------------------------ | :------- | :------- |
| **Our Model (Notebook)** | **0.639**| **0.523**|
| EfficientNet (Image-only) | 0.395    | 0.314    |
| BERT (Text-only)          | 0.645    | 0.587    |
| MM-GATBT (SOTA)           | 0.685    | 0.645    |


## How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/sup41kkk/NLP-course.git](https://github.com/sup41kkk/NLP-course.git)
    cd NLP-course
    ```
2.  Ensure you have the required dependencies installed (see `requirements.txt` if available, or install based on the notebook imports).
3.  Open and run the Jupyter Notebook: `nlp-coursev2_best.ipynb`.
    - The notebook handles data downloading, preprocessing, model training, and evaluation.
    - Ensure a GPU environment is available for training.

## Dependencies
Key Python libraries used:
-   PyTorch
-   Transformers (Hugging Face)
-   Datasets (Hugging Face)
-   scikit-learn
-   Matplotlib
-   NumPy
-   Pillow
-   Requests

## Related Work
The field of multimodal learning for tasks like genre classification is extensive. Approaches range from simple fusion techniques to complex attention-based models and graph neural networks.
-   **Multimodal Representation**: Involves early, late, or hybrid fusion of features from different modalities (e.g., MMBT by Kiela et al., 2020).
-   **Graph Neural Networks (GNNs)**: Models like GCN, GraphSAGE, and GAT (Veličković et al., 2018) are used for relational data. MM-GATBT (Seo et al., 2022) specifically applies this to the MM-IMDb dataset by constructing an entity graph.
-   **Attention Mechanisms**: Self-attention (Vaswani et al., 2017) is a key component in models like BERT and is also used in this project for fusing multimodal features.

For a detailed overview, please refer to Section \ref{sec:related} in `Supkhankulov_report.pdf`.

## Conclusion
This project demonstrated the development of a multimodal classifier for movie genre prediction using ResNet50 and BERT. The model achieved a validation Macro F1-score of 0.5230 by employing several advanced techniques. While effective, there is room for improvement compared to SOTA graph-based approaches. Future work could explore more complex fusion mechanisms or graph-based relational learning.
