from model import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def run(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def inference(self, X):
        return self.model.predict(X)