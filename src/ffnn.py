from model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class FeedforwardNN(BaseModel):
    def __init__(self, input_dim):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def run(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    def inference(self, X):
        return (self.model.predict(X) > 0.5).astype(int)