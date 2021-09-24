#!/usr/local_rwth/bin/zsh
### ask for 5 GB memory
#SBATCH --mem-per-cpu=5G   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=fmnist
### job run time
#SBATCH --time=20:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=fmnist_logs.%J.txt
### request a GPU
#SBATCH --gres=gpu:pascal:1

### begin of executable commands
cd $HOME/repos/CosDefence/federated_learning
### load modules
module switch intel gcc
module load python
module load cuda/111
module load cudnn/8.0.5
# pip3 install --user tensorboard
# pip3 install --user -U scikit-learn
# pip3 install plotly
# pip3 install -U kaleido
# pip3 install pyyaml

python3 run_config_variations.py fmnist_modified.yaml
