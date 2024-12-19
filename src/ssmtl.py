from model import BaseModel
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras import backend as K

class SSMTL(BaseModel):
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder, self.decoder = self._build_vae()
        self.classifier = self._build_classifier()

    def _build_vae(self):
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        h = Dense(128, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        # Decoder
        latent_inputs = Input(shape=(self.latent_dim,))
        h_decoded = Dense(128, activation='relu')(latent_inputs)
        outputs = Dense(self.input_dim, activation='sigmoid')(h_decoded)

        encoder = Model(inputs, [z_mean, z_log_var, z])
        decoder = Model(latent_inputs, outputs)

        return encoder, decoder

    def _build_classifier(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.latent_dim,)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def run(self, X_train, y_train):
        # Train VAE
        z_mean, z_log_var, z = self.encoder.predict(X_train)
        reconstructed = self.decoder.predict(z)

        # Train classifier
        self.classifier.fit(z, y_train, epochs=10, batch_size=32, verbose=0)

    def inference(self, X):
        z_mean, z_log_var, z = self.encoder.predict(X)
        return (self.classifier.predict(z) > 0.5).astype(int)