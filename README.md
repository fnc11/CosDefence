# CosDefence: A defence mechanism against Data Poisoning attacks in Federated Learning (FL)

This projects aims to develop a defence mechanism against data poisoning attacks in FL. It employs techniques like cosine similarity to check similarity between the clients, kmeans clustering to detect important neural connection and EigenTrust to score different clients.

## Overview

1. Project Setup
2. Project Structure
3. Running A Basic Experiment
4. Configuration Settings
5. Visualizing Results

## 1. Project Setup

To run this project mainly you need to intall python3 and PyTorch deep learning framework. To create the environment just use the environment.yml file. Currently experiments were performed on three datasets **mnist, fmnist and cifar10**.

## 2. Project Structure

- `config` folder contains configuration settings for different datasets.
- `data` folder contains raw and federated data for 100 clients.
- `federated_learning` folder contains main modules of this project like preparing data, running FL and visualizing results.
- `logs` folder contains logs of the experiments.
- `results` folder contains plots and results of the experiment as json files.
- `run_scripts` contains script to run the experiments on rwth compute cluster.

## 3. Running A Basic Experiment

In a terminal first go to `federated_learning` folder and then type the following command (assuming you already did the setup and activated the environment):

```bash
python3 run_config.py mnist_default.yaml
```

If you want to run a config file with different variations for comparison purposes, use the below command:

```bash
python3 run_config_variations.py mnist_default.yaml
```

## 4. Configuration Settings

Let's look at one of the configuration files `mnist_default.yaml`

```yaml
# FL_DATA
DATASET: "mnist"
TOTAL_CLIENTS: 100 # total clients 10|100
CLIENT_FRAC: 0.2 # client fraction to select in each round 0.1|0.2|0.4
POISON_FRAC: 0.2 # poisoned fraction for environment 0.0|0.1|0.2|0.3|0.4|0.5
CREATE_DATASET: False # if true it creates FL Dataset everytime
RANDOM_DATA: False # to create data and poison data randomly or with fixed seed 42 
CLASS_RATIO: 10 # class ratio in client data distribution, choose 1|4|10


# FL_SETTINGS
MODEL: "nnet" # model to be used
LEARNING_RATE: 0.001 # client's model learning rate
BATCH_SIZE: 32 # client model parameter
LOCAL_EPOCHS: 10 # epochs to run before clients send back the model update
OPTIMIZER: "adam" # client model parameter
FED_ROUNDS: 30 # federated rounds to run
TEST_EVERY: 1 # test the model for accuracy after every 3rd round
RANDOM_PROCESS: False # to select candidates randomly or with fixed seed 42 


# COS_DEFENCE_SETTINGS
COS_DEFENCE: True # to turn on cos_defence
SEL_METHOD: 1  # Client selection method, 0: total_random, 1: system_trust as probability, 2: top k clients according to system_trust
REDUCE_LR: 1  # By what factor selected learning rate should be reduced in the starting before the cos_defence mechanism starts trust based gathering.
COLLAB_MODE: True  # collaboration mode on means, all validation clients jointly assign trust
COLLAB_ALL: True # whether all client's joint vector is used for validation or top half
CONSIDER_LAYERS: "l1"  # choose layers for grad collection f2|f1|f1l1|l1|l2|all
GRAD_AGG: True   # Collect all gradients over the iterations or just use last iteration grads
FEATURE_FINDING_ALGO: auror  # use Auror's cluster algo to find important neural units in the layers, other options none|auror_plus
GRAD_COLLECTION_START: 0   # when to start collecting grads 
GRAD_COLLECT_FOR: 20   #for how many fed rounds grads are collected, -1 means selected based on client_frac 
CLUSTER_SEP: 0.01   # parameter for Auror's cluster algo, how much seperation between clusters to consider them important
ALPHA: 0.6   # parameter for updating system trust vector
BETA: 0.8    # parameter for updating system trust matrix
GAMMA: 0.1   # decides how much initial trust worthy clients in the environment
TRUST_INC: 2
TRUST_MODIFY_STRATEGY: 1  # 0 for using clustering method to segregate, 1 for using median, mean, std method, 2 for none
HONEST_PARDON_FACTOR: 1.0 # it decides how much you want to pardon good clients getting bad trust values
RESET_AXIS: False # for updating trust of assumed malicious clients , 0 for normal, 1 for resetting trust axis
TRUST_SAMPLING: True # to fill matix by sampling
TRUST_NORMALIZATION: False # to normalize trust values before filling in the matrix


# Other Settings
LOG_LEVEL: 'ERROR' # To make logs based on logging level, DEBUG|INFO|WARNING|ERROR|CRITICAL
JSON_RESULTS: True  # whether to save results, they are saved in json files for later visualization
GEN_PLOTS: True # whehter to generate plot for experiment or not, this will also control generating plot dfs
SAVE_HTML_PLOTS: False # whether to save interactive html plots or just png image plots.
```

## 5. Visualizing Results

By default when _cos_defence_ mechanism is on it generates three type of plots:

- Accuracy F1 Poison Plot: Shows you total class accuracy (avg of all classes), poisoned class accuracy and poisoned data selected in that round.
- Trust Histogram Plot: Shows you how much trust honest and malicious(minor offender and major offender) client got during similarity calculations.
- Trust Score Curves Plot: Shows how the trust score of different clients evolved during the training.

When _cos_defence_ mechanism is off only Accuracy F1 Poison Plot is generated.
