import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle

class DataLoader:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data

    def preprocess_data(self):
        data = self.load_data()
        data = shuffle(data)
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test