# Signature Verification with Siamese Neural Networks

This project implements a robust Signature Verification system using a Siamese Neural Network (SNN) architecture. The model uses feature extraction backbones like ResNet and EfficientNet for offline signature verification and leverages a contrastive loss function to differentiate between genuine and forged signatures.

## Key Features:
- **Siamese Neural Network** with ResNet and EfficientNet for accurate feature extraction.
- **Contrastive loss function** to optimize the model for signature verification.
- **Dynamic learning rate adjustment** and **threshold tuning** to maximize F1 score performance.
- **Scalable solution** for real-world signature forgery detection.

## Data:
- **Crawl Data**: [Download from Google Drive](https://drive.google.com/file/d/1xwjj_kims4cGjwZY7jySnJX3UYBpWFb-/view?usp=sharing)  
- **Public Data**: [Download from Kaggle](https://www.kaggle.com/datasets/mallapraveen/signature-matching)

## Instructions:
1. Clone the repository:
    ```bash
    git clone https://github.com/lbyyybaby/signature_verification.git
    ```
2. Navigate to the project folder:
    ```bash
    cd signature_verification
    ```
3. Open and run the main notebook (signature_verification.ipynb) for training, validation, and testing.
    ```bash
    jupyter notebook signature_verification.ipynb
    ```
## Requirements:
- Python 3.x
- TensorFlow or PyTorch
- Necessary dependencies

## License:
This project is licensed under the MIT License.
