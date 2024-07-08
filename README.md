# FedFV: Federated Learning Framework for Finger Vein Image Classification

FedFV is a federated learning-based framework for classifying Finger Vein images using computer vision. This project utilizes the power of federated learning to create robust and privacy-preserving machine learning models, built on top of the pre-trained MobileNet V2 architecture, for finger vein classification.

## Table of Contents
- [Introduction](#introduction)
- [What is Federated Learning?](#what-is-federated-learning)
- [Advantages of Federated Learning](#advantages-of-federated-learning)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

FedFV is designed to facilitate federated learning for finger vein image classification, leveraging computer vision techniques. The framework supports two aggregation strategies: FedAvg and FedWPR, enabling personalized federated learning for non-iid datasets.

## What is Federated Learning?

Federated learning is a distributed machine learning approach that enables model training across multiple decentralized devices holding local data samples, without exchanging them. This method ensures data privacy and security, as the data remains localized.

## Advantages of Federated Learning

- **Data Privacy**: Keeps sensitive data on local devices, reducing the risk of data breaches.
- **Efficiency**: Reduces the need for large-scale data transfers, saving bandwidth and improving efficiency.
- **Personalization**: Enables the creation of personalized models that are fine-tuned to the specific data of each client, enhancing performance for diverse datasets.

## Project Structure

- **Data Conversion code**: Contains Jupyter notebooks for data preprocessing.
  - `convert_datasets.ipynb`: Converts your datasets into subfolders label-wise.
  - `create_non_iid_datasets.ipynb`: Splits your data into non-iid datasets.
  - `split_train_test.ipynb`: Splits the data into training and testing sets.
- **GPU based**: Contains a notebook to check the available GPUs on your device.
  - `gpu_test.ipynb`: Checks and lists available GPUs.
- `**FedAvg.ipynb**`: Main implementation of the `FedAvg` aggregation algorithm.
- `**FedWPR.ipynb**`: Main implementation of the `FedWPR` aggregation algorithm.
- `**requirements.txt**`: Lists all the necessary libraries for the project.
- `**Lian_et_al_2023_FedFV_Paper.pdf**`: Base paper used for the implementation.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/FedFV.git
   cd FedFV
   ```

2. Create and activate a virtual environment:
   
    **Windows:**
    ```sh
    python -m venv venv
    .\venv\Scripts\activate
    ```
  
    **macOS/Linux:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
**Note**: It is highly recommended to use GPUs for faster execution. For executing `FedAvg.ipynb` and `FedWPR.ipynb`, Google Colab's T4 GPU has been used. Data is imported using Google Drive. If you have a good GPU, you can skip the cells involving Google Drive mounting and directly add your dataset paths.

## Data Preparation

- If you have raw data, use the notebooks in the Data Conversion code folder to preprocess your datasets.
    -`convert_datasets.ipynb`
    -`create_non_iid_datasets.ipynb`
    -`split_train_test.ipynb`
  
-If you already have `non-iid datasets`, you can skip this step.

## Run Federated Learning:
1. Choose the aggregation strategy (FedAvg or FedWPR).
2. Update the dataset paths in the respective notebooks (FedAvg.ipynb or FedWPR.ipynb).
3. Run the notebooks to start the federated learning process.

## Results

## FedAvg
After 10 rounds of training, impressive accuracies were achieved:

Client 1: 85%

Client 2: 87%

Client 3: 81%

Client 4: 88%

## FedWPR
The baseline architecture is completed, and hyperparameter tuning is in progress.

## Contributing
Contributions are welcome! Please read the contributing guidelines for more details.

## Licence
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any inquiries or issues, please contact the project maintainers at [ssnfs26@gmail.com].
