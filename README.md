---
<<<<<<< HEAD
tags: [advanced, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Advanced Flower Example (PyTorch)

This example demonstrates an advanced federated learning setup using Flower with PyTorch. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) and it differs from the quickstart example in the following ways:

- 10 clients (instead of just 2)
- Each client holds a local dataset of 5000 training examples and 1000 test examples (note that using the `run.sh` script will only select 10 data samples by default, as the `--toy` argument is set).
- Server-side model evaluation after parameter aggregation
- Hyperparameter schedule using config functions
- Custom return values
- Server-side parameter initialization
=======
tags: [basic, vision, fds]
dataset: [CIFAR-10]
framework: [torch]
---

# PyTorch: From Centralized To Federated

This example demonstrates how an already existing centralized PyTorch-based machine learning project can be federated with Flower.

This introductory example for Flower uses PyTorch, but you're not required to be a PyTorch expert to run the example. The example will help you to understand how Flower can be used to build federated learning use cases based on existing machine learning projects. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.
>>>>>>> f9786dc (update)

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
<<<<<<< HEAD
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/advanced-pytorch . && rm -rf flower && cd advanced-pytorch
```

This will create a new directory called `advanced-pytorch` containing the following files:
=======
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/pytorch-from-centralized-to-federated . && rm -rf flower && cd pytorch-from-centralized-to-federated
```

This will create a new directory called `pytorch-from-centralized-to-federated` containing the following files:
>>>>>>> f9786dc (update)

```shell
-- pyproject.toml
-- requirements.txt
<<<<<<< HEAD
-- client.py
-- server.py
-- README.md
-- run.sh
=======
-- cifar.py
-- client.py
-- server.py
-- README.md
>>>>>>> f9786dc (update)
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

<<<<<<< HEAD
## Run Federated Learning with PyTorch and Flower

The included `run.sh` will start the Flower server (using `server.py`),
sleep for 2 seconds to ensure that the server is up, and then start 10 Flower clients (using `client.py`) with only a small subset of the data (in order to run on any machine),
but this can be changed by removing the `--toy` argument in the script. You can simply start everything in a terminal as follows:

```shell
# After activating your environment
./run.sh
```

The `run.sh` script starts processes in the background so that you don't have to open eleven terminal windows. If you experiment with the code example and something goes wrong, simply using `CTRL + C` on Linux (or `CMD + C` on macOS) wouldn't normally kill all these processes, which is why the script ends with `trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT` and `wait`. This simply allows you to stop the experiment using `CTRL + C` (or `CMD + C`). If you change the script and anything goes wrong you can still use `killall python` (or `killall python3`) to kill all background processes (or a more specific command if you have other Python processes running that you don't want to kill).

You can also manually run `python3 server.py` and `python3 client.py --client-id <ID>` for as many clients as you want but you have to make sure that each command is run in a different terminal window (or a different computer on the network). In addition, you can make your clients use either `EfficienNet` (default) or `AlexNet` (but all clients in the experiment should use the same). Switch between models using the `--model` flag when launching `client.py` and `server.py`.
=======
## From Centralized To Federated

This PyTorch example is based on the [Deep Learning with PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) tutorial and uses the CIFAR-10 dataset (a RGB image classification task). Feel free to consult the tutorial if you want to get a better understanding of PyTorch. The file `cifar.py` contains all the steps that are described in the tutorial. It loads the dataset, trains a convolutional neural network (CNN) on the training set, and evaluates the trained model on the test set.

You can simply start the centralized training as described in the tutorial by running `cifar.py`:

```shell
python3 cifar.py
```

The next step is to use our existing project code in `cifar.py` and build a federated learning system based on it. The only things we need are a simple Flower server (in `server.py`) and a Flower client that connects Flower to our existing model and data (in `client.py`). The Flower client basically takes the already defined model and training code and tells Flower how to call it.

Start the server in a terminal as follows:

```shell
python3 server.py
```

Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py
```

Start client 2 in the second terminal:

```shell
python3 client.py
```

You are now training a PyTorch-based CNN image classifier on CIFAR-10, federated across two clients. The setup is of course simplified since both clients hold the same dataset, but you can now continue with your own explorations. How about using different subsets of CIFAR-10 on each client? How about adding more clients?
>>>>>>> f9786dc (update)
