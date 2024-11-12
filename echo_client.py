import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import cifar
import flwr as fl
import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

disable_progress_bar()

USE_FEDBN: bool = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create directory to store parameters
SAVE_DIR = "local_parameters"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class AdversarialClient(fl.client.NumPyClient):
    """Flower client implementing adversarial behavior for federated learning."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: DataLoader,
        testloader: DataLoader,
        epochs: int = 1,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs  # Number of epochs for training (unused in adversarial client)
        self.global_parameters_1 = None  # Storage for global parameters from two rounds ago
        self.global_parameters_2 = None  # Storage for global parameters from one round ago

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Store received global parameters
        if self.global_parameters_1 is None:
            # First two rounds: store received parameters in global_parameters_1
            self.global_parameters_1 = parameters
            print("Storing initial global parameters.")
        else:
            # Shift stored parameters for the next rounds
            self.global_parameters_1 = self.global_parameters_2
            self.global_parameters_2 = parameters
            print("Shifting global parameters for delayed transmission.")

        # Determine the parameters to send back to the server
        if self.global_parameters_1 is not None:
            print("Sending stored parameters from two rounds ago.")
            parameters_to_send = self.global_parameters_1
        else:
            print("Sending initialized parameters (first round).")
            parameters_to_send = [np.random.randn(*param.shape) for param in parameters]

        # Save the parameters that are being sent to the server
        round_number = int(config.get("round_number", 0))  # Get round number
        self.save_parameters(parameters_to_send, round_number)

        return parameters_to_send, len(self.trainloader.dataset), {}

    def save_parameters(self, parameters: List[np.ndarray], round_number: int) -> None:
        """Save the current parameters to a JSON file."""
        try:
            # Convert parameters to lists for JSON serialization
            parameters_list = [param.tolist() for param in parameters]
            save_path = os.path.join(SAVE_DIR, f"parameters_round_{round_number}.json")
            with open(save_path, "w") as f:
                json.dump(parameters_list, f)
            print(f"Parameters saved to {save_path}")
        except Exception as e:
            print(f"Failed to save parameters: {e}")

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def main() -> None:
    """Load data, start AdversarialClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition-id", type=int, required=True, choices=range(0, 10))
    parser.add_argument("--round-number", type=int, default=0, help="Round number for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    args = parser.parse_args()

    # Load data
    trainloader, testloader = cifar.load_data(args.partition_id)

    # Load model
    model = cifar.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))["img"].to(DEVICE))

    # Start client with specified number of epochs
    client = AdversarialClient(model, trainloader, testloader, epochs=args.epochs).to_client()
    fl.client.start_client(server_address="192.168.0.40:8080", client=client)

if __name__ == "__main__":
    main()
