from sklearn.metrics import accuracy_score, f1_score

class Evaluation:
    @staticmethod
    def evaluate(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return accuracy, f1