import flwr as fl
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_keras import get_model, get_images_data
from load_MITBIH import load_mit_db
import settings
print("🚀 client B started")
# Load local data (client-specific)
def load_data():
    tr_features, tr_labels, _, _ = load_mit_db(
        'DS2', 90, 90, True, True, True, True, {'raw'},settings.db_path, False, [1, 0]
    )

    x, y, _ = get_images_data(tr_features, tr_labels, 5000, "train_client")
    y = tf.keras.utils.to_categorical(y, num_classes=6)
    return x, y


class ECGClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = get_model()
        self.x_train, self.y_train = load_data()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("📡 Client training started")
        self.model.set_weights(parameters)
        if len(self.x_train) == 0 or len(self.y_train) == 0:
            return self.model.get_weights(), 0, {}

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1,
            batch_size=32,
            verbose=1
        )

        # 🔐 Light Differential Privacy (noise injection)
        noisy_weights = [
            w + np.random.normal(0, 0.001, w.shape)
            for w in self.model.get_weights()
        ]

        return noisy_weights, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(
            self.x_train, self.y_train, verbose=0
        )
        return loss, len(self.x_train), {"accuracy": acc}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8081",
        client=ECGClient()
    )