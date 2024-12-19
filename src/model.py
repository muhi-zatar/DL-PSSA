class BaseModel:
    def run(self, X_train, y_train):
        raise NotImplementedError

    def inference(self, X):
        raise NotImplementedError