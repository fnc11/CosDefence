# CosDefence: A defence mechanism against Data Poisoning attacks in Federated Learning (FL)

In this projects a novel defence mechanism called CosDefence against data poisoning attacks in FL was developed. It employs techniques like cosine similarity to check similarity between the clients, kmeans clustering to detect important neural connection and EigenTrust to keep track of trust scores of different clients.

## Overview

1. Project Setup
2. Project Structure
3. Running A Basic Experiment
4. Running Multiple Complex Experiments
5. Running Experiments on RWTH HPC Cluster
6. Configuration Settings
7. Visualizing Results
8. Further Improvements and Research Directions

## 1. Project Setup

To run this project mainly you need to intall python3 and PyTorch deep learning framework. To create the environment just use the environment.yml file. Currently experiments were performed on three datasets **mnist, fmnist and cifar10**.

## 2. Project Structure

- `config` folder contains configuration settings for different datasets.
- `data` folder contains raw and federated data for 100 clients. When you first clone this might be empty, you need to generate data using a data preparation script.
- `federated_learning` folder contains main modules of this project like preparing data, running FL and visualizing results.
- `notebooks` folder contain some notebooks to further analyse the summary results.
- `logs` folder contains logs of the experiments.
- `results` folder contains plots, plot_dfs and results of the experiment as json files.
- `run_scripts` contains script to run the experiments on RWTH HPC compute cluster and log of those experiment.

## 3. Running A Basic Experiment
There are two options (assuming you already did the setup and activated the environment):
### A. Prepare data first:
If you choose this option then you can create the federated data first for 5 or more distributions for the selected client data ratio (10:1, 4:1, 1:1). In config you can select the CLASS_RATIO:10 or 4 or 1.
Run below instruction to prepare data for 5 distributions for each type of poisonous environments (0, 10, 20, 30, 40, 50, 60) and then run the experiment. Just keep the cofig setting CREATE_DATASET:False. It basically creates one distribution everytime if it's True.
```bash
cd federated_learning/
python3 prepare_data.py mnist_modified.yaml 5
python3 run_config.py mnist_modified.yaml
```
### B. Not prepare data separately:
If you don't want to create more distributions and just want to run experiment on single distribution, then in the config set CREATE_DATASET:True.
And you can turn it False after running one time, if you want to test with random distribution everytime, keep it True and in the config set RANDOM_DATA:True. This will randomize image distribution and poisoning process of the clients.
```bash
cd federated_learning/
python3 run_config.py mnist_default.yaml
```
## 4. Running Multiple Complex Experiments
If you want to run a config file with different variations in the config for comparison purposes, use the _run_config_variations.py_ file.
```bash
cd federated_learning/
python3 run_config_variations.py mnist_default.yaml
```

## 5. Running Experiments on RWTH HPC Cluster
- If you are in RWTH Network then you don't need VPN, otherwise first use RWTH VPN.
- Then make one folder called _repos_ and copy/clone this project _CosDefence_ in that folder.
- For first time you might need to run below commands in terminal, you might need to update version of different modules to the newer versions or according to hardware compatibility.
```
module load python/3.8.7
pip3 install --user tensorboard
pip3 install --user -U scikit-learn
pip3 install --user plotly
pip3 install --user -U kaleido
pip3 install --user pyyaml
pip3 install --user torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
- After installing these one time module, go to _run_scripts_ folder and use any of the scripts mnist_script.sh, fmnist_script.sh, cifar10_script.sh. These scripts contains instruction on how to run your batch job. Make sure the python module version you are loading in the script is same where you installed all the above packages like tensorboard, scikit-learn pytorch, plotly, pytorch etc. These commands are explained on RWTH HPC Cluster website also.
- The two important commands in this file are below:
```
cd $HOME/repos/CosDefence/federated_learning
python3 run_config_variations.py mnist_modified.yaml
```
- If you don't want to store the _CosDefence_ project in repos folder then change the location in this file also. The other command is to what experiments to be run on the batch. 
- Change these script file variables according to the dataset and experiement numbers. For example in the case of mnist, fmnist, cifar10, a single experiment takes ~6, ~20, ~50 minutes respectively, so if you are running 5 experiments for fmnist give little more time than 100 minutes i.e.  02:00:00 . You can request different size mem-per-cpu, change job name, logs name, request gpu etc. You will be notified to the given mail ID if your job starts, ends, cancels etc.

## 6. Configuration Settings

Let's look at one of the configuration files `mnist_modified.yaml`

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

## 7. Visualizing Results

By default when _CosDefence_ mechanism is on it generates three type of plots:

- Accuracy F1 Poison Plot: Shows you total class accuracy (avg of all classes), F1, poisoned class accuracy, F1 and poisoned data selected in that round.
- Trust Histogram Plot: Shows you how much trust honest and malicious (minor offender and major offender) client got during similarity calculations.
- Trust Score Curves Plot: Shows how the trust score of different clients evolved during the training.

When _CosDefence_ mechanism is off only Accuracy F1 Poison Plot is generated.
For further visualization use jupyter notebooks in the notebooks folder.

## 8. Further Improvements and Research Directions
- Running parallelized experiments.
- Find better Hyper-parameter Tuning Strategy to find best value in less time.
- Research on how Matrix and Vector can be updated in a better way.
- Better Client Selection Strategy and better Feature Finding Algorithm.
- Making code more modular, readable and configurable.
