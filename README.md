# Power System Security Assessment using Machine Learning

## Overview
This is the implementation of the [paper](https://arxiv.org/abs/2407.08886) Semi-Supervised Multi-Task Learning Based Framework for Power System Security Assessment.

## Directory Structure
```
.
├── src
│   ├── DT.py                    # Implementation of Decision Tree algorithm
│   ├── RF.py                    # Implementation of Random Forest algorithm
│   ├── svm.py                   # Implementation of Support Vector Machine
│   ├── xgboost.py               # Implementation of XGBoost
│   ├── ffnn.py                  # Implementation of Feedforward Neural Network
│   ├── rnn.py                   # Implementation of Recurrent Neural Network
│   ├── ssmtl.py                 # Implementation of Conditional Variational Autoencoder
├── utils
│   ├── data.py                  # Data loading and preprocessing
│   ├── eval.py                  # Evaluation metrics (Accuracy and F1 score)
├── main.py                      # Main script to run the models
├── config.json                  # Configuration file for models and data
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Prerequisites
- Python 3.8+
- Install dependencies using:

```bash
pip install -r requirements.txt
```

## Configuration
Modify the `config.json` file to specify:
- Path to the dataset (`file_path`)
- Target column name (`target_column`)
- List of models to run (`models_to_run`)
- Latent dimensions for Conditional Variational Autoencoder (`latent_dim`)

## Running the Project
1. Prepare your dataset and update the `config.json` file with its path and target column.
2. Run the main script:

```bash
python main.py
```
3. The output will display the accuracy and F1 score for each specified model.

## Algorithms Implemented
- **Decision Tree**: Classical machine learning algorithm.
- **Random Forest**: Ensemble learning technique.
- **Support Vector Machine (SVM)**: Effective for smaller datasets.
- **XGBoost**: Gradient boosting for classification tasks.
- **Feedforward Neural Network (FFNN)**: Fully connected deep learning model.
- **Recurrent Neural Network (RNN)**: Handles sequential data effectively.
- **Conditional Variational Autoencoder (CVAE)**: Deep generative model with a classification head.

## Example Output
```text
Running decision_tree...
decision_tree - Accuracy: 0.8723, F1 Score: 0.8602

Running random_forest...
random_forest - Accuracy: 0.9021, F1 Score: 0.8905

Running svm...
svm - Accuracy: 0.8456, F1 Score: 0.8327
...
```
