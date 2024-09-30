import argparse
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import flwr as fl
import torch
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from flask import Flask, request, jsonify  # Flask import
import utils

warnings.filterwarnings("ignore")

app = Flask(__name__)
client_counter = 0  # 클라이언트 ID를 부여하기 위한 카운터
client_mapping = {}  # IP 주소와 클라이언트 ID 매핑
client_parameters = {}  # 각 클라이언트의 파라미터를 저장하기 위한 딕셔너리

def assign_client_id(client_ip):
    """클라이언트 IP에 따라 고유한 클라이언트 ID를 부여합니다."""
    global client_counter
    if client_ip not in client_mapping:
        client_mapping[client_ip] = f"client_{client_counter}"
        client_counter += 1
    return client_mapping[client_ip]

@app.route('/send_params', methods=['POST'])
def receive_parameters():
    """클라이언트로부터 로컬 파라미터를 수신하여 저장합니다."""
    data = request.json
    client_ip = request.remote_addr  # 클라이언트 IP 주소
    client_id = assign_client_id(client_ip)
    
    # 수신된 파라미터 저장
    client_parameters[client_id] = data['parameters']
    
    # 로그 출력
    print(f"Received parameters from {client_id} with IP {client_ip}")
    
    return jsonify({"status": "parameters received", "client_id": client_id}), 200

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""
    centralized_data = utils.load_centralized_data()
    if toy:
        centralized_data = centralized_data.select(range(10))

    val_loader = DataLoader(centralized_data, batch_size=16)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, val_loader)
        return loss, {"accuracy": accuracy}

    return evaluate

def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. \
             If you want to achieve differential privacy, please use the Alexnet model",
    )

    args = parser.parse_args()

    if args.model == "alexnet":
        model = utils.load_alexnet(classes=10)
    else:
        model = utils.load_efficientnet(classes=10)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )

if __name__ == "__main__":
    # Flask 서버 시작
    app.run(host='0.0.0.0', port=5000)
    main()
