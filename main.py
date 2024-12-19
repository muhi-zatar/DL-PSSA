import json
from utils.data import DataLoader
from utils.eval import Evaluation
from src.DT import DecisionTree
from src.RF import RandomForest
from src.svm import SVM
from src.xgboost import XGBoost
from src.ffnn import FeedforwardNN
from src.rnn import RecurrentNN
from src.ssmtl import SSMTL

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Initialize data loader
data_loader = DataLoader(config['data']['file_path'], config['data']['target_column'])
X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data()

# Model mapping
models = {
    "decision_tree": DecisionTree(),
    "random_forest": RandomForest(),
    "svm": SVM(),
    "xgboost": XGBoost(),
    "feedforward_nn": FeedforwardNN(input_dim=X_train.shape[1]),
    "recurrent_nn": RecurrentNN(input_dim=X_train.shape[1]),
    "conditional_vae": SSMTL(input_dim=X_train.shape[1], latent_dim=config['cvae']['latent_dim'])
}

# Run models
for model_name in config['models_to_run']:
    print(f"Running {model_name}...")
    model = models[model_name]
    model.run(X_train, y_train)
    y_pred = model.inference(X_val)

    # Evaluate results
    accuracy, f1 = Evaluation.evaluate(y_val, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")