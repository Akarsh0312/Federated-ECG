from xml.parsers.expat import model

import flwr as fl
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from train_keras import get_model
import os

MODEL_DIR = "../model"
MODEL_PATH = "../model/global_model.h5"

os.makedirs(MODEL_DIR, exist_ok=True)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated is not None:
            print(f"💾 Saving global model after round {rnd}")

            weights = aggregated[0]
            model = get_model()
            #model.set_weights(weights)
            from flwr.common import parameters_to_ndarrays
            model.set_weights(parameters_to_ndarrays(weights))
            model.save(MODEL_PATH)

        return aggregated


def main():
    print("🌐 Federated Server Starting...")

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1
    )

    port = int(os.environ.get("PORT", 10000))

    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )    


if __name__ == "__main__":
    main()