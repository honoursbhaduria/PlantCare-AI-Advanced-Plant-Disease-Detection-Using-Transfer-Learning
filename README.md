# PlantCare AI — Advanced Plant Disease Detection Using Transfer Learning

Rice Leaf Disease Image Classification using Convolutional Neural Networks and Transfer Learning techniques to accurately detect and classify three major rice leaf diseases.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/honoursbhadauria/riceleafdisease-cnn-accuracy-86-67.2fc7a359-8315-4138-b6f6-3eda0f98d091.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20260309/auto/storage/goog4_request%26X-Goog-Date%3D20260309T114425Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3Dabb3ac4d60464dfe753a1fbd43858a35f44f78c598f379efdf3cd81f0872379cd9a9585832dee4e405d9301d87eb48a2f54ce1672c90612e7b2c19f602779fde6554c5d5e8f4ba33cba3d760b229cd7dcb6b9190d56d612cc119bfb7647a556bc5126be9b113afaa199117af9c6a5aba876acccabdd4b22b1c3dfdfa28fff0a8067a6dd8c1ba64d6510c241e9ab9b43880f363c632166a9c86d4eebfa0407b5b81f456c95e761a5b5cf966e945a581647a44ce48ffd0e0de3d67e67ab0f1c477e05bf03fd6cb7d5d11a91ed5c245ca93ecf34ff46bf539b49817d4345a3d35f13ead78892197ad3dc910c7425a91669d2e45a1f679346de546a7ad54a9f6462c)

> **Notebook Gist:** [View on GitHub Gist](https://gist.github.com/honoursbhaduria/e1e3e5e344e56e6e43f3872863913646)
>
> **Group Capstone Project** — Developed collaboratively as part of the training program.

---

## Table of Contents

- [Overview](#overview)
- [Disease Classes](#disease-classes)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Pipeline](#project-pipeline)
- [Models & Results](#models--results)
- [Transfer Learning](#transfer-learning)
- [Data Augmentation](#data-augmentation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Challenges](#challenges)
- [Getting Started](#getting-started)
- [Project Deliverables](#project-deliverables)
- [References](#references)

---

## Overview

Rice is one of the most important staple crops in the world and a major source of food for millions of people. However, rice plants are susceptible to various diseases that can cause significant losses in yield and quality. This project develops a machine learning model that accurately classifies the three major attacking diseases of rice plants based on leaf images using advanced deep learning and transfer learning techniques.

By leveraging pre-trained models and data augmentation, the system provides fast and accurate disease detection, enabling farmers to take prompt and effective measures to control the spread of diseases and minimize crop losses.

---

## Disease Classes

| Disease | Cause | Symptoms |
|---|---|---|
| **Bacterial Leaf Blight** | *Xanthomonas oryzae pv. oryzae* | Water-soaked lesions on leaves, turning brown and dry; wilting and death in severe cases |
| **Brown Spot** | *Cochliobolus miyabeanus* | Small, oval to elliptical brown spots with yellow halo; leaves wither and die in severe cases |
| **Leaf Smut** | *Entyloma oryzae* | Small, round, reddish-brown spots turning black with powdery spores |

---

## Dataset

- **Source:** [Rice Leaf Diseases — Kaggle](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)
- **Total Images:** 119 (across 3 classes)
- **Image Size (resized):** 224 x 224 x 3 (RGB)
- **Split:** 75% Training / 25% Testing (`random_state=1`)
- **Classes:** Balanced distribution across Bacterial Blight, Brown Spot, and Leaf Smut

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy, Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Hyperparameter Tuning | Keras Tuner (RandomSearch) |
| Transfer Learning Models | InceptionV3, Xception, VGG16, ResNet152V2 |
| Environment | Python 3, Google Colab / Kaggle Notebooks |

---

## Project Pipeline

```
1. Data Loading & Exploration
       |
2. Data Preprocessing
   - Resizing to 224x224
   - Normalization (pixel values / 255)
   - One-hot Encoding of labels
   - Train-Test Split
       |
3. Model Building & Training
   - Custom CNN
   - Custom CNN + Data Augmentation
   - Keras Tuner Hyperparameter Search
   - Transfer Learning (InceptionV3, Xception)
       |
4. Evaluation & Comparison
       |
5. Best Model Selection (Xception + Augmentation)
       |
6. Kernel Visualization & Prediction
```

---

## Models & Results

| Model | Augmentation | Key Highlights |
|---|---|---|
| Custom CNN | No | Baseline model with 3 Conv2D layers (16, 32, 64 filters) |
| Custom CNN | Yes | Improved generalization with rotation, shift & flip augmentation |
| Tuned CNN (Keras Tuner) | Yes | Best hyperparameters found via RandomSearch (3 trials) |
| InceptionV3 | No | Fine-tuned from `mixed9_0` layer onward |
| InceptionV3 | Yes | Augmented training on fine-tuned InceptionV3 |
| Xception | No | Fine-tuned from `add_8` layer onward |
| **Xception** | **Yes** | **Best model — Validation Accuracy: 96.67%** |

> The **Xception model with data augmentation** was selected as the final model based on the best balance of accuracy, loss, and efficiency.

---

## Transfer Learning

Pre-trained models (trained on ImageNet with 1000 classes) were adapted for rice leaf disease classification using transfer learning:

- **Frozen Layers:** Early convolutional layers retain learned general features (edges, textures, shapes).
- **Fine-Tuned Layers:** Later layers are unfrozen and retrained on the rice leaf dataset for domain-specific feature extraction.
- **Custom Head:** A `Flatten` → `Dense(128, relu)` → `Dense(3, softmax)` classification head replaces the original classifier.

### Architecture Highlights

- **InceptionV3:** Layers unfrozen from `mixed9_0` onward
- **Xception:** Layers unfrozen from `add_8` onward (depthwise separable convolutions)

---

## Data Augmentation

To combat the limited dataset size (only 119 images), the following augmentations were applied using `ImageDataGenerator`:

| Augmentation | Value |
|---|---|
| Rotation Range | 40 degrees |
| Width Shift | 0.3 |
| Height Shift | 0.3 |
| Horizontal Flip | Yes |
| Vertical Flip | Yes |

---

## Hyperparameter Tuning

Keras Tuner (`RandomSearch`) was used to optimize the custom CNN architecture:

| Hyperparameter | Search Space |
|---|---|
| Conv Layer Filters | 8–32 (layer 1), 16–64 (layer 2), 32–64 (layer 3) |
| Kernel Size | 2 or 3 |
| Padding | `same` or `valid` |
| Dense Units | 50–250 (step 50) |
| Learning Rate | 0.1, 0.01, 0.001 |

- **Objective:** Maximize `val_accuracy`
- **Max Trials:** 3

---

## Challenges

- **Limited Data:** Only 119 images — mitigated with data augmentation.
- **Model Complexity:** Deep models prone to overfitting — addressed with early stopping (`patience=5`) and augmentation.
- **Hardware Constraints:** Leveraged cloud GPU environments (Google Colab / Kaggle).
- **Model Selection:** Systematic comparison across multiple architectures using accuracy, loss, and training time.

---

## Getting Started

### Run on Google Colab (Recommended)

Click the badge below to open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/honoursbhadauria/riceleafdisease-cnn-accuracy-86-67.2fc7a359-8315-4138-b6f6-3eda0f98d091.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20260309/auto/storage/goog4_request%26X-Goog-Date%3D20260309T114425Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3Dabb3ac4d60464dfe753a1fbd43858a35f44f78c598f379efdf3cd81f0872379cd9a9585832dee4e405d9301d87eb48a2f54ce1672c90612e7b2c19f602779fde6554c5d5e8f4ba33cba3d760b229cd7dcb6b9190d56d612cc119bfb7647a556bc5126be9b113afaa199117af9c6a5aba876acccabdd4b22b1c3dfdfa28fff0a8067a6dd8c1ba64d6510c241e9ab9b43880f363c632166a9c86d4eebfa0407b5b81f456c95e761a5b5cf966e945a581647a44ce48ffd0e0de3d67e67ab0f1c477e05bf03fd6cb7d5d11a91ed5c245ca93ecf34ff46bf539b49817d4345a3d35f13ead78892197ad3dc910c7425a91669d2e45a1f679346de546a7ad54a9f6462c)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/honoursbhaduria/PlantCare-AI-Advanced-Plant-Disease-Detection-Using-Transfer-Learning.git
cd PlantCare-AI-Advanced-Plant-Disease-Detection-Using-Transfer-Learning

# Install dependencies
pip install tensorflow keras keras_tuner kagglehub matplotlib seaborn scikit-learn pandas numpy

# Download dataset
python -c "import kagglehub; kagglehub.dataset_download('vbookshelf/rice-leaf-diseases')"

# Launch notebook
jupyter notebook
```

### Training Configuration

| Parameter | Value |
|---|---|
| Input Shape | 224 x 224 x 3 |
| Batch Size | 32 |
| Epochs | 30 |
| Early Stopping Patience | 5 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |

---

## References

1. Mew, T.W., Alvarez, A.M., and Leach, J.E. (1993). *Bacterial leaf blight.* In: Compendium of rice diseases (ed. R.K. Webster), 25-36.
2. Singh, U.S., Singh, D.P., and Chaube, H.S. (2015). *Rice brown spot disease: A review.* Journal of Plant Pathology and Microbiology, 6(4), 1-8.
3. Acedo, A.L., and Daquioag, R.D. (2013). *Leaf smut disease of rice.* Philippine Agricultural Scientist, 96(4), 354-364.
4. Savary, S., et al. (2000). *Rice pest and disease constraints in the Philippines.* Plant Disease, 84(6), 565-572.
5. Sladojevic, S., et al. (2016). *Deep neural networks based recognition of plant diseases by leaf image classification.* Computational Intelligence and Neuroscience, 2016, 1-11.
6. Yosinski, J., et al. (2014). *How transferable are features in deep neural networks?* In Advances in Neural Information Processing Systems (pp. 3320-3328).
7. Szegedy, C., et al. (2016). *Rethinking the inception architecture for computer vision.* In Proceedings of IEEE CVPR (pp. 2818-2826).
8. Chollet, F. (2016). *Xception: Deep Learning with Depthwise Separable Convolutions.* arXiv:1610.02357.
9. [Rice Leaf Diseases Dataset — Kaggle](https://www.kaggle.com/datasets/raihan150146/rice-leaf-diseases-dataset)

---

## Project Deliverables

This is a **Group Capstone Project** completed as part of the training program. The following deliverables are included:

| Deliverable | Status |
|---|---|
| Source Code (Notebook) | Uploaded to GitHub |
| Dataset | Kaggle — Rice Leaf Diseases |
| Trained Model Weights | Generated during training (best checkpoints saved) |
| Results & Visualizations | Included in notebook (loss/accuracy plots, predictions) |
| Project Report | Completed using the provided project document template |
| Demo Video | Prepared for mentor review |
| GitHub Repository | [PlantCare-AI-Advanced-Plant-Disease-Detection-Using-Transfer-Learning](https://github.com/honoursbhaduria/PlantCare-AI-Advanced-Plant-Disease-Detection-Using-Transfer-Learning) |

---

<p align="center"><b>Made with dedication towards sustainable agriculture</b></p>
