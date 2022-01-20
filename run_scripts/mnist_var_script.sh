#!/usr/local_rwth/bin/zsh
### ask for 4 GB memory
#SBATCH --mem-per-cpu=4G   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=mnist_ffa_none
### job run time
#SBATCH --time=06:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=mnist_ffa_logs.%J.txt
###
#SBATCH --mail-type=ALL
###
#SBATCH --mail-user=praveen.yadav@rwth-aachen.de
### request a GPU
#SBATCH --gres=gpu:volta:1

### begin of executable commands
cd $HOME/repos/CosDefence/federated_learning
### load modules
module switch intel gcc
module load python/3.8.7
module load cuda/11.1
module load cudnn/8.0.5
# pip3 install --user tensorboard
# pip3 install --user -U scikit-learn
# pip3 install plotly
# pip3 install -U kaleido
# pip3 instal pyyaml

python3 run_config_variations.py mnist_modified.yaml
