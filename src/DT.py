from model import BaseModel
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(BaseModel):
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def run(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def inference(self, X):
        return self.model.predict(X)