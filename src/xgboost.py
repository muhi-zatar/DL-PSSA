from xgboost import XGBClassifier
from model import BaseModel

class XGBoost(BaseModel):
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def run(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def inference(self, X):
        return self.model.predict(X)