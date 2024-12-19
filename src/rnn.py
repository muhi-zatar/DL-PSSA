from model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class RecurrentNN(BaseModel):
    def __init__(self, input_dim):
        self.model = Sequential([
            LSTM(64, activation='relu', input_shape=(input_dim, 1)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def run(self, X_train, y_train):
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    def inference(self, X):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return (self.model.predict(X) > 0.5).astype(int)