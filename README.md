# ANN Classification for Churn Prediction

## Introduction

This repository implements a deep learning solution using Artificial Neural Networks (ANN) for customer churn prediction and classification. The project leverages neural networks to identify patterns in customer behavior and predict which customers are likely to churn, enabling proactive retention strategies.

## Project Overview

Customer churn prediction is a critical business problem in various industries such as telecommunications, finance, and subscription-based services. This project demonstrates how to build, train, and evaluate ANN models to accurately classify customers as likely to churn or not churn.

**Key Features:**
- Implementation of Artificial Neural Networks for binary classification
- Data preprocessing and feature engineering
- Model training and evaluation
- Performance metrics and visualization
- Hyperparameter tuning for optimal results

## Technologies & Languages

- **Primary Language:** Jupyter Notebook (96.5%)
- **Supporting Language:** Python (3.5%)
- **Key Libraries:**
  - TensorFlow / Keras (for neural networks)
  - Pandas (data manipulation)
  - NumPy (numerical computing)
  - Scikit-learn (preprocessing & metrics)
  - Matplotlib / Seaborn (visualization)

## Project Structure

```
ANN-classification-churn/
├── README.md                 # Project documentation
├── notebooks/                # Jupyter notebooks
│   ├── data_preprocessing.ipynb      # Data cleaning and preparation
│   ├── eda.ipynb                     # Exploratory data analysis
│   ├── model_training.ipynb          # ANN model building and training
│   └── evaluation.ipynb              # Model evaluation and results
└── data/                     # Dataset directory
    └── churn_data.csv       # Customer churn dataset
```

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required libraries (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ssampathkumar104/ANN-classification-churn.git
   cd ANN-classification-churn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage

1. Start with **data_preprocessing.ipynb** to understand data cleaning
2. Review **eda.ipynb** for exploratory analysis
3. Follow **model_training.ipynb** to see ANN implementation
4. Check **evaluation.ipynb** for model performance metrics

## Model Architecture

The ANN model consists of:
- **Input Layer:** Features from preprocessed customer data
- **Hidden Layers:** Multiple dense layers with ReLU activation for feature extraction
- **Output Layer:** Sigmoid activation for binary classification (Churn/No Churn)
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam or SGD

## Results

The model achieves competitive performance on churn prediction tasks with metrics including:
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC

## Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## License

This project is open source and available under the MIT License.

## Author

**S. Sampath Kumar**
- GitHub: [@ssampathkumar104](https://github.com/ssampathkumar104)

## Acknowledgments

- TensorFlow/Keras documentation and tutorials
- Scikit-learn community
- Open-source data science community

---

**Last Updated:** 2026-07-01
