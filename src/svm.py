from model import BaseModel
from sklearn.svm import SVC

class SVM(BaseModel):
    def __init__(self):
        self.model = SVC()

    def run(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def inference(self, X):
        return self.model.predict(X)