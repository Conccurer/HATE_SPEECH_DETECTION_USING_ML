# Hate Speech Detection

This project aims to detect hate speech patterns in comments and posts on social media platforms, with a focus on Facebook. The goal is to develop a machine learning model that can classify content as hate speech or not, providing insights into how such speech manifests online.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Description](#model-description)
- [Results](#results)
- [Contributing](#contributing)

## Overview

The project utilizes a dataset of labeled comments to train a machine learning model capable of identifying hate speech. It involves various steps, including data preprocessing, feature extraction using TF-IDF, and training using Support Vector Machine (SVM) classifiers.

## Features

- **Data Preprocessing:** Clean and preprocess text data to remove noise, stop words, and other irrelevant elements.
- **Feature Extraction:** Use TF-IDF to convert text data into numerical features suitable for machine learning.
- **Model Training:** Train an SVM classifier to distinguish between hate speech and non-hate speech.
- **Evaluation:** Assess model performance using metrics such as accuracy, F1-score, and confusion matrix.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/hate-speech-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd hate-speech-detection
    ```
3. Install the required dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn nltk
    ```

## Usage

1. Ensure that you have the required dataset (`labeled_data.csv`) in the project directory.
2. Open the notebook `HATE_SPEECH_DETECTION.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the cells sequentially to execute the code, train the model, and evaluate its performance.

## Model Description

- **Algorithm:** Support Vector Machine (SVM)
- **Feature Extraction:** TF-IDF Vectorization
- **Libraries Used:** 
  - Scikit-learn for model training and evaluation
  - NLTK for text preprocessing
  - Pandas and NumPy for data manipulation
  - Matplotlib and Seaborn for visualization

## Results

The performance of the model is evaluated using:
- **Accuracy Score:** Measures the overall correctness of the model.
- **F1 Score:** Balances precision and recall, particularly important for imbalanced datasets.
- **Confusion Matrix:** Provides detailed insight into the model’s predictions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs, improvements, or suggestions.
